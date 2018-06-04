"""
This module contains tools for performing gradient descent optimization.
Currently only supports the fast/accelerated gradient descent algorithm
for convex optimization problems.

To use the gradient_utils library simply instanciate the GradientOptimizer
and pass it the appropriate loss function (must be in loss_functions library).

Given a matrix X of size (n, d), the optimizer will return a list of arrays (1 for each iteration),
each with parameters with dimension (d,).

"""
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

import loss_functions as lfuncs

log = logging.getLogger(__name__)

# Set constants
available_functions = lfuncs.get_available_functions()

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

        if loss_function in available_functions.keys():
            self.loss_function = available_functions[loss_function]()
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
