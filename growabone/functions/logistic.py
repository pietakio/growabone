#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Logistic functions for modeling human growth characteristics**

'''

import numpy as np

def growth_potential(ti, ta, alpha):
    '''
    An equation to compute the Logistic-curve growth potential given time points ti and the
    necessary parameters.
    '''
    phi =  alpha*(1 - (1 / (1 + np.exp(-alpha * (ti - ta)))))

    return phi


def growth_vel(ti, ta, alpha, Lmax):
    '''
    An equation to compute the Logistic growth velocity given time points ti and the
    necessary parameters.
    '''
    logi = 1 / (1 + np.exp(-alpha * (ti - ta)))

    Vt = logi*alpha*Lmax*(1 - logi)

    return Vt


def growth_len(ti, ta, alpha, Lmax=1.0):
    '''
    An equation to compute the Logistic growth curve given time points ti and the
    necessary parameters. Computes in terms of the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    ta : float
        Center of growth curve
    alpha : float
        Decline in initial rate of growth
    Lmax : float
        Specifies the maximum growth that the segment reaches. If Lmax is 1, the
        growth curve is normalized.

    '''
    Lt = Lmax / (1 + np.exp(-alpha * (ti - ta)))

    return Lt

def growth_len_fitting(ti, ta, alpha):
    '''
    An equation to compute the Logistic growth curve given time points ti and the
    necessary parameters. Computes in terms of the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    ta : float
        Center of growth curve
    alpha : float
        Decline in initial rate of growth

    '''
    Lt = 1 / (1 + np.exp(-alpha * (ti - ta)))

    return Lt

def growth_fboost(ta, alpha):

    Fboost = np.log(1 + np.exp(alpha*ta))

    return Fboost

def growth_mapper(A_o, alpha_o, A_i, alpha_i):
    '''
    Given two sets of growth potential parameters, find the scaling factor lambda that
    will map curve 'o' to curve 'i'.

    Parameters
    ---------------
    :param A_o:
    :param B_o:
    :param A_i:
    :param alpha_i:

    Returns
    ----------------
    '''

    # Calculate the boost factor for the i-data:
    Fboost_i = growth_fboost(A_i, alpha_i)

    # The scaling equation is trivial: lamb*(A_o/alpha_o) = Fboost_i
    lamb = Fboost_i*(alpha_o/A_o)

    return lamb
