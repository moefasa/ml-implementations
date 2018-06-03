"""
This module holds base estimators. Each class is scikit-learn compatible
and implements .fit and .predict methods (at a minimum).

Currently only holds 'MyLinearSVM' with squared-hinge loss.
"""
import os
import sys
import copy
import itertools
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

import gradient_utils as grad_utils

log = logging.getLogger(__name__)

class MyLinearSVM(BaseEstimator, ClassifierMixin):
    """ Class for linear support vector machine trained using
        gradient descent.

        Args:
        reg_coef (int/float): regularization parameter.
        max_iter (int): maximum number of iterations for gradient descent.
        eta_init (int/float): starting learning rate.
        loss_function (str): type of loss function.
            Currently only supports 'squared-hinge'.

    """
    def __init__(self, reg_coef=1, max_iter=100, eta_init=1, loss_function='squared-hinge'):
        self.reg_coef = reg_coef
        self.max_iter = max_iter
        self.eta_init = eta_init
        self.loss_function = loss_function

    def fit(self, X, y=None):
        """ Performs gradient descent and returns betas.

            Args:
            X (numpy ndarray): matrix
            y (numpy array): binary labels must be (-1/+1)

            Returns:
            self
        """
        self._gradient_optimizer = \
            grad_utils.GradientOptimizer(
                loss_function=self.loss_function,
                max_iter=self.max_iter,
                eta_init=self.eta_init,
                )

        self._gradient_optimizer.optimize(X, y, self.reg_coef)
        self.betas_ = self._gradient_optimizer.betas_
        self.beta_ = self.betas_[-1]

        return self

    def predict_proba(self, X, y=None, beta=None):
        """ Returns values. Name predict_proba used
            for compatibility with other classifiers.
            Allows passage of optional beta values.
        """
        if beta is None:
            beta = self.beta_
        return X @ beta

    def predict(self, X, y=None, beta=None):
        """ Returns predicted labels (-1/+1).
            Allows passage of optional beta values.
        """
        if beta is None:
            beta = self.beta_
        vals = self.predict_proba(X=X, beta=beta)
        result = pd.Series([-1]*len(vals))
        result[vals > 0.0] = 1
        return result
