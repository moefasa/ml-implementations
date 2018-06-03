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
        base_model (model object of list of model objects): instance of scikit-learn compatible model.
            (has .fit, .predict, and .predict_proba methods). Can also be list of models, but must be
            the same length as the number of labels in the training set.
        n_jobs: number of parallel jobs. Uses same logic as sklearn's
            n_jobs (-1 uses all cores, -2 all but one, etc.)
        grid: if not None will perform grid search on the base model for every label.
            grid should be in same format as in scikit-learn's GridSearchCV.
        grid_n_jobs: number of jobs if using grid search. Will be forced to 1 if n_jobs=-1.
        cv (int): number of cross-validation folds of grid is not None.

        Additional Fields:
        binarizer: Scikit-learn's LabelBinarizer for converting labels into -1/+1

    """
    def __init__(self, base_model=best.MyLinearSVM(), n_jobs=1, grid=None, grid_n_jobs=1, cv=3, params_list=[]):
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
        if hasattr(self.base_model, '__iter__'):
            if len(self.base_model) < self.num_labels_:
                raise ValueError('If base_model is a list it must be the same \
                    size as the number of labels in the training set.')
            self.models_ = self.base_model
        else:
            self.models_ = [copy.deepcopy(self.base_model) for _ in range(self.num_labels_)]


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
