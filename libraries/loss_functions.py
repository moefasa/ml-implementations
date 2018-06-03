"""
This module contains loss/cost/objective function classes to be referenced
by the gradient_utils library.

To add a custom loss function simply inherit from the base LossFunction()
class and follow the instructions in the docstrings.

You must also add the name of the function (as referenced by user input)
in the available_functions dictionary along with the class (not instance).
"""

import sys
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin

def get_available_functions():
    available_functions =  {
        "squared-hinge": SquaredHingeLoss
        }
    return available_functions


class LossFunction():
    """ Base class for any loss function. An instance (1) maintains state of
        current point, objective, and gradient, and (2) can query objective and
        gradient given a candidate point.

        Required fields for subclasses:
        * self.n: The number of rows in X.
        * self.d: The number of columns in X.

        Required methods for subclasses.
        * set_space: is used to pass in X and y, along with any additional
          parameters that affect the space (regularization coeficient).
          Constants should be cached , used for queries of gradient and
          objective at any point.
        * obj: returns the objective at some point (beta).
        * computegrad: returns the gradient at some point (beta).

        Example:
        loss = MyLossFunction()
        loss.set_space(X, y)
        loss.update_pos(beta).get_current_obj() # Updates location and returns objective
        loss.get_current_grad() # returns gradient for beta

        loss.obj(beta2) # returns objective at location beta2, but does not update
        loss.computeobj(beta2) # returns gradient at location beta2, but does not update

        NOTES: * Any attribute with trailing '_' means it is
                 an attribute of the current point/position/beta.
               * The constructor should not be modified and should simply
                 call the super class constructor (e.g. super().__init__()).


    """
    def __init__(self):
        self.beta_ = None

    @abstractmethod
    def set_space(self, X, y, *args):
        """ Function for setting X and y and cacheing any
            values.
        """
        yield

    @abstractmethod
    def obj(self, beta, *args):
        """ Method for computing objective
            for some point in space.
        """
        yield

    @abstractmethod
    def computegrad(self, beta, *args):
        """ Method for computing gradient for some
            point in space.
        """
        yield

    def update_pos(self, beta):
        """ Updates position of 'current' point.
            Note that in subclasses, any attribute with
            trailing '_' is a property of the current point.
            That is, it has been updated using update_pos.

            Args:
            beta (numpy array): vector of updated location.
        """
        self.beta_ = beta
        return self

    @property
    def obj_(self):
        return self.obj(self.beta_)

    @property
    def grad_(self):
        return self.computegrad(self.beta_)

    def get_current_obj(self):
        return self.obj_

    def get_current_grad(self):
        return self.grad_

    def get_current_point(self):
        return self.beta_

    def get_shape(self):
        """ Method returns tuple (n, d) where
            n is the number of rows in X
            and d is the number of dimensions.
        """
        return self.n, self.d

class SquaredHingeLoss(LossFunction):
    """ Loss function for squared-hinge loss. Inherits from
        LossFunction() class. See super class docstrings for
        additional details.
    """
    def __init__(self):
        super().__init__()

    def set_space(self, X, y, reg_coef):
        # Cache constants
        self.yx = y[:, np.newaxis] * X
        self.reg_coef = reg_coef
        self.n, self.d = X.shape

    def obj(self, beta):
        reg_coef = self.reg_coef
        yx = self.yx
        n, d = self.n, self.d

        hinge = np.square(np.maximum(0, 1-yx@beta)).sum()
        obj = (1.0/n)*hinge + reg_coef * norm(beta)**2
        return obj

    def computegrad(self, beta):
        reg_coef = self.reg_coef
        yx = self.yx
        n, d = self.n, self.d

        dhinge = (yx.T * np.maximum(0, 1-yx@beta)).sum(axis=1)
        grad = (-2.0/n)*dhinge + 2*reg_coef*beta
        return grad
