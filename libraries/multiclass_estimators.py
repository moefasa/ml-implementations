"""
This module contains tools for multiclassification problems.
Currently only supports implementation of one-vs-rest strategy.

All classes support any estimator that is scikit-learn compatible.
"""
import os
import sys
import copy
import itertools
import logging
import multiprocessing
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

import base_estimators as best

log = logging.getLogger(__name__)

class MyMultiClassifier(BaseEstimator, ClassifierMixin):
    """ Base class for multi-classification.

        Args:
        base_model: instance of scikit-learn compatible model.
        n_jobs: number of parallel jobs. Uses same logic as sklearn's
            n_jobs (-1 uses all cores, -2 all but one, etc.)

        Additional Fields:
        models_: list of models (one for every label set)
        labels_: list of labels
    """
    def __init__(self, base_model, n_jobs=-1):
        self.base_model = base_model
        self.n_jobs = n_jobs
        self.models_ = []
        self.labels_ = []

    @abstractmethod
    def fit(self, *args):
        yield

    @abstractmethod
    def predict(self, *args):
        yield

    @property
    def num_labels_(self):
        return len(self.labels_)

    def create_pool(self):
        """ Method for creating a pool.

            Returns:
            pool: object with map, close, and join methods.

            NOTE: See constructor docstrings for notes on n_jobs.
        """
        if self.n_jobs == 1:
            pool = NullPool()
        elif self.n_jobs <= 0:
            n_cpus = multiprocessing.cpu_count()
            n_processes = n_cpus + 1 + self.n_jobs
            pool = multiprocessing.Pool(processes=n_processes)
        else:
            pool = multiprocessing.Pool(processes=self.n_jobs)
        return pool

class MyOneVsRestClassifier(MyMultiClassifier):
    """ Class for one-vs-rest multi-classification.

        Args:
        base_model: instance of scikit-learn compatible model.
        n_jobs: number of parallel jobs. Uses same logic as sklearn's
            n_jobs (-1 uses all cores, -2 all but one, etc.)
        grid: if not None will perform grid search for every label.
            grid should be in same format as in scikit-learn's GridSearchCV.
        grid_n_jobs: number of jobs if using grid search. Will be forced to 1 if n_jobs=-1.
        cv (int): number of cross-validation folds of grid is not None.
        params_list (list): optional list of dictionaries to set model parameters.
            E.g. params_list = [{"reg_coef": 1}, {"reg_coef": 2}, {"reg_coef": 3}]
            fits 3 models with model 1 having reg_coef=1, etc.

        Additional Fields:
        binarizer: Scikit-learn's LabelBinarizer for converting labels into -1/+1

    """
    def __init__(self, base_model=best.MyLinearSVM(), n_jobs=1, grid=None, grid_n_jobs=1, cv=3, params_list=[]):
        """
        grid is for base_model
        params_list is for setting any parameters for the set of models.
        """
        if grid is not None:
            if n_jobs == -1:
                grid_n_jobs = 1
            base_model = GridSearchCV(base_model, param_grid=grid, cv=cv, n_jobs=grid_n_jobs)

        super().__init__(base_model=base_model, n_jobs=n_jobs)
        self.params_list = params_list
        self.binarizer = LabelBinarizer(neg_label=-1)


    def fit(self, X, y):
        """ Method fits num_labels_ models.
        """
        self.binarizer.fit(y)
        self.labels_ = self.binarizer.classes_

        if self.num_labels_ == 2:
            raise ValueError('The MyOneVsRestClassifier does not support \
                binary label classification.')

        # Instanciate models
        self.models_ = [copy.deepcopy(self.base_model) for _ in range(self.num_labels_)]

        if self.params_list:
            set_params(self.models_, self.params_list)

        # Create list of replicas (to pass into parallel function)
        xtrain_list = [X]*self.num_labels_
        ytrain_list = [y]*self.num_labels_
        binarizer_list = [self.binarizer]*self.num_labels_

        count = 0
        configs_list = []
        for label in self.labels_:
            d = {}
            d['model'] = self.models_[count]
            d['label'] = label
            d['X'] = xtrain_list[count]
            d['y'] = ytrain_list[count]
            d["binarizer"] = binarizer_list[count]
            configs_list.append(d)
            count += 1

        self.configs_list = configs_list

        pool = super().create_pool()
        results = pool.map(fit_model_set, configs_list)
        pool.close()
        pool.join()
        self.models_ = results
        return self

    def predict(self, X, y=None, beta=None):
        """ Method returns labels for X with list
            of previously fit models.
        """
        results_list = []
        for model in self.models_:
            vals = model.predict_proba(X)
            results_list.append(vals)
        results = pd.DataFrame(np.stack(results_list, axis=1))
        yhat = self.labels_[results.idxmax(axis=1).astype(int)]
        return yhat


class NullPool():
    """ Convenience class for non-parallel
        compute with interface similar to
        multiprocessing.Pool.
    """

    def map(self, func, iter_inputs):
        """ Args:

            func: callable function.
            iter_inputs: list of inputs to function.

            Returns:
            list of results from calling func on inputs.
        """
        results = []
        for input in iter_inputs:
            results.append(func(input))
        return results

    def join(self):
        pass

    def close(self):
        pass


def fit_model_set(configs):
    """ Functions for fitting a single 'model set'
        given configurations.

        Args:
        configs (dict): has keys: [X, y, model, label, binarizer]

        Returns:
        model fit to X and y (after y is transformed by binarizer).
    """
    X = configs["X"]
    y = configs["y"]
    model = configs["model"]
    label = configs["label"]
    binarizer = configs["binarizer"]

    label_indx = np.where(binarizer.classes_ == label)[0][0]
    y_train = binarizer.transform(y)[:, label_indx]

    log.info("Fitting label {}".format(label))

    mod = model.fit(X, y_train)
    return mod

def set_params(list_of_models, list_of_params):
    """ Modifies the original list of models.

        Args:
        list_of_models: list of model objects.
        list_of_params: list of dictionaries holding attributes
            contained in model objects.
    """
    if len(list_of_models) != len(list_of_params):
        raise ValueError("params_list should be the same size as the \
            number of models to fit.")
    for i, model in enumerate(list_of_models):
        params = list_of_params[i]
        print(model)
        print(params)
        for key, value in params.items():
            setattr(model, key, value)
    return None
