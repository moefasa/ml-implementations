"""
This module contains tools for performing gradient descent optimization.
Currently only supports the fast/accelerated gradient descent algorithm
for convex optimization problems.
"""
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

log = logging.getLogger(__name__)

class GradientOptimizer():
    """ Interface class to be used by base estimators.

        Args:
        loss_function (str): type of loss function to use.
            Currently only supports 'squared-hinge'
        eta_init (int/float): initial step-size.
        max_iter (int): maximum number of gradient steps.
        eps (float): tolerance/stopping criteria.

    """
    def __init__(self, loss_function, eta_init=1, max_iter=1000, eps=1e-3):
        self.max_iter = max_iter
        self.eta_init = eta_init
        self.eps = eps
        self.betas_ = []
        self.beta_ = []

        if loss_function == 'squared-hinge':
            self.loss_function = SquaredHingeLoss()
        else:
            raise ValueError("Loss function '{}' \
                is not supported.".format(loss_function))

    def optimize(self, X, y, *args):
        """ Method sets solution space for LossFunction instance
            and performs fast gradient descent.
        """
        self.loss_function.set_space(X, y, *args)
        self.betas_ = fastgradalgo(
            self.loss_function,
            t_init=self.eta_init,
            max_iter=self.max_iter
            )
        return self.betas_

def fastgradalgo(loss_function, t_init=1, eps=1e-3, max_iter=100):
    """ Fast/accelerated gradient descent algorithm.

        Args:
        loss_function: object that inherits from LossFunction(). See class
            for additional information.
        t_init (int/float): initial step-size/learning rate.
        max_iter (int): maximum number of gradient steps.
        eps (float): tolerance/stopping criteria.
    """
    n, d = loss_function.get_shape()
    beta_init = np.array([0]*d)
    theta_init = np.array([0]*d)
    grad = loss_function.update_pos(beta_init).get_current_grad()
    grad_norm = norm(grad)

    beta_vals  = [beta_init]
    theta_vals = [theta_init]
    eta = t_init
    i = 0

    log.info("Starting fast gradient descent. Initial gradient norm: {}".format(grad_norm))
    while grad_norm > eps and i < max_iter:
        grad = loss_function.update_pos(theta_vals[i]).get_current_grad()
        eta = backtracking(loss_function, eta)

        beta_vals.append(theta_vals[i] - eta*grad)
        theta_vals.append(beta_vals[i+1] + (i/(i+3))*(beta_vals[i+1]-beta_vals[i]))
        grad_norm = norm(grad)
        i += 1
        log.debug("Iteration: {}, Grad Norm: {}".format(i, grad_norm))
    log.info("Ending fast gradient descent. Final gradient norm: {}".format(grad_norm))
    return np.array(beta_vals)

def backtracking(loss_function, t=1, alpha=0.5, beta=0.5, max_iter=50):
    """ Function that performs backtracking line-search to find an appropriate
        learning rate given the current position.

        This code was partly taken from Corinne Jones from the
        University of Washington.

        Args:
        loss_function: object that inherits from LossFunction(). Maintains state
            of current location, objective, and gradient. See class docstrings for
            additional information.
        t (int/float): max/starting step-size.
        alpha: Constant used to define sufficient decrease condition.
        beta: Fraction by which we decrease t if the previous t doesn't work.

        Return:
        t: the step-size to take.
    """
    b = loss_function.get_current_point()
    grad_b = loss_function.get_current_grad()
    norm_grad_b = norm(grad_b)
    found_t = False
    i = 0  # Iteration counter

    while (found_t is False and i < max_iter):
        log.debug("Backtracking i: {}, eta: {}".format(i, t))
        if (loss_function.obj(b-t*grad_b) <
                    (loss_function.obj(b)-alpha*t*(norm_grad_b**2))):
            found_t = True
        elif i == max_iter - 1:
            log.debug("Backtracking hasn't converged. Last t: {}".format(t))
            return t
        else:
            t *= beta
            i += 1
    return t

class LossFunction():
    """ Base class for any loss function. An instance (1) maintains state of
        current point, objective, and gradient, and (2) can query objective and
        gradient given a candidate point.

        Required fields for subclasses:
        * The number of rows in X is self.n.
        * The number of columns in X is self.d.

        Required methods for subclasses.
        * set_space is used to pass in X and y, along with any additional
          parameters that affect the space (regularization coeficient).
          Constants should be cached and cache constants, used for queries of
          gradient and objective at any point.
        * obj returns the objective at some point (beta).
        * computegrad returns the gradient at some point (beta).

        Example:
        loss = MyLossFunction()
        loss.set_space(X, y)
        loss.update_pos(beta).get_current_obj() # Updates location and returns objective
        loss.get_current_grad() # returns gradient for beta

        loss.obj(beta2) # returns objective at location beta2, but does not update
        loss.computeobj(beta2) # returns gradient at location beta2, but does not update

        NOTE: Any attribute with trailing '_' means it is
              an attribute of the current point/position/beta.


    """

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

    @abstractmethod
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

    @abstractmethod
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
        self.beta_ = None

    def set_space(self, X, y, reg_coef):
        self.reset(space=True)
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

    def reset(self, space=False):
        """ Resets current point.

            Args:
            space (boolean): if True, will also reset solution space (X, y).

            Return:
            self
        """
        self.beta_ = None
        if space:
            self.yx = None
            self.n, self.d = None, None
        return self
