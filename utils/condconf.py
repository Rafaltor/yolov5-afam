import cvxpy as cp
import numpy as np

from functools import partial
from scipy.optimize import linprog
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable

FUNCTION_DEFAULTS = {"kernel": None, "gamma" : 1, "lambda": 1}

class CondConf:
    def __init__(
            self,
            n_conf : int,
            n_size : int,
            Phi_fn : Callable,

        ):
        """
        Constructs the CondConf object that caches relevant information for
        generating conditionally valid prediction sets.

        We define the score function and set of conditional guarantees
        that we care about in this function.

        Parameters
        ---------

        Phi_fn : Callable[np.ndarray] -> np.ndarray
            Function that defines finite basis set that we provide
            exact conditional guarantees over
        """
        self.Phi_fn = Phi_fn
        self.n_conf = n_conf
        self.n_size = n_size

    def setup_problem(
            self,
            scores_calib : np.ndarray,
            conf: np.ndarray,
            size: np.ndarray,

    ):
        """
        setup_problem sets up the final fitting problem for a 
        particular calibration set

        The resulting cvxpy Problem object is stored inside the CondConf parent.

        Arguments
        ---------
        x_calib : np.ndarray
            Covariate data for the calibration set

        y_calib : np.ndarray
            Labels for the calibration set
        """
        self.phi_calib = self.Phi_fn(conf, size, self.n_conf, self.n_size)
        self.scores_calib = scores_calib.numpy()

        

    def predict(
            self,
            quantile : float,
            x_test : np.ndarray,
            x_conf : int,
            x_size : int,
            S_min : float = None,
            S_max : float = None
    ):
        """
        Returns the (conditionally valid) prediction set for a given 
        test point

        Arguments
        ---------
        quantile : float
            Nominal quantile level
        x_test : np.ndarray
            Single test point
        score_inv_fn : Callable[float, np.ndarray] -> .
            Function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        S_min : float = None
            Lower bound (if available) on the conformity scores
        S_max : float = None
            Upper bound (if available) on the conformity scores

        Returns
        -------
        prediction_set
        """
        scores_calib = self.scores_calib
        if S_min is None:
            S_min = np.min(scores_calib)
        if S_max is None:
            S_max = np.max(scores_calib)
        phi_test = self.Phi_fn(x_conf, x_size, self.n_conf, self.n_size)

        _solve = partial(_solve_dual, gcc=self, phi_test=phi_test, quantile=quantile)

        #lower, upper = binary_search(_solve, S_min, S_max * 2)

        if quantile < 0.5:
            threshold = self._get_threshold(lower, phi_test, quantile)
            #threshold = np.quantile((self.phi_calib*np.expand_dims(scores_calib,1))[:, self.Phi_fn(x_test).argmax()][np.nonzero(self.phi_calib[:, self.Phi_fn(x_test).argmax()])], 0.05)
        else:
            threshold = self._get_threshold(upper, phi_test, quantile)
            #threshold = np.quantile((self.phi_calib*np.expand_dims(scores_calib,1))[:, phi_test.argmax()][np.nonzero(self.phi_calib[:, phi_test.argmax()])], 0.95)

        #print(threshold, threshold_1)
        return threshold



    def _get_threshold(
        self,
        S : float,
        phi_test : np.ndarray,
        quantile : float
    ):


        S = np.concatenate([self.scores_calib, [S]])
        Phi = np.concatenate([self.phi_calib, phi_test], axis=0)
        zeros = np.zeros((Phi.shape[1],))
        bounds = [(quantile - 1, quantile)] * (len(self.scores_calib) + 1)
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds,
                      method='highs-ds', options={'presolve': False})
        beta = -1 * res.eqlin.marginals

        threshold = phi_test @ beta

        return threshold
    

def binary_search(func, min, max, tol=0.5):
    min, max = float(min), float(max)
    assert (max + tol) > max
    while (max - min) > tol:
        mid = (min + max) / 2
        if np.isclose(func(mid), 0):
            max = mid
        else:
            min = mid
    return min, max


def _solve_dual(S, gcc, phi_test, quantile):


    S = np.concatenate([gcc.scores_calib, [S]], dtype=float)
    Phi = np.concatenate([gcc.phi_calib, phi_test], axis=0, dtype=float)
    zeros = np.zeros((Phi.shape[1],))

    bounds = [(quantile - 1, quantile)] * (len(gcc.scores_calib) + 1)
    res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds,
                  method='highs', options={'presolve': False})
    weights = res.x

    if quantile < 0.5:
        return weights[-1] + (1 - quantile)
    return weights[-1] - quantile




def _get_kernel_matrix(x_calib, kernel, gamma):
    K = pairwise_kernels(
        X=x_calib,
        metric=kernel,
        gamma=gamma
    ) + 1e-5 * np.eye(len(x_calib))

    K_chol = np.linalg.cholesky(K)
    return K, K_chol


def finish_dual_setup(
    prob : cp.Problem,
    S : np.ndarray, 
    X : np.ndarray,
    quantile : float,
    Phi : np.ndarray,
    x_calib : np.ndarray,
    infinite_params = {}
):
    prob.param_dict['S_test'].value = np.asarray([[S]])
    prob.param_dict['Phi_test'].value = Phi.reshape(1,-1)
    prob.param_dict['quantile'].value = quantile

    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])

    if kernel is not None:
        K_12 = pairwise_kernels(
            X=np.concatenate([x_calib, X.reshape(1,-1)], axis=0),
            Y=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )

        if 'K_12' in prob.param_dict:
            prob.param_dict['K_12'].value = K_12[:-1]
            prob.param_dict['K_21'].value = K_12.T

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
        K_22 = pairwise_kernels(
            X=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )
        L_21 = np.linalg.solve(L_11, K_12[:-1]).T
        L_22 = K_22 - L_21 @ L_21.T
        L_22[L_22 < 0] = 0
        L_22 = np.sqrt(L_22)    
        prob.param_dict['L_21_22'].value = np.hstack([L_21, L_22])
    
        prob.param_dict['radius'].value = radius

        # update quantile definition for silly cvxpy reasons
        prob.param_dict['quantile'].value *= radius / (len(x_calib) + 1)
    
    return prob

