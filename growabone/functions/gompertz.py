#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Gompertz functions for modeling human growth characteristics**

'''

import numpy as np

def growth_potential(ti, Ao, alpha):
    '''
    An equation to compute the Gompertz-curve growth potential given time points ti and the
    necessary parameters.
    '''
    phi = Ao * np.exp(-alpha * (ti))

    return phi


def growth_vel(ti, Ao, alpha, Lmax=1):
    '''
    An equation to compute the Gompertz growth velocity given time points ti and the
    necessary parameters.
    '''
    Vt = Ao*Lmax*np.exp(-alpha*ti)*np.exp(-(Ao / alpha)*np.exp(-alpha * (ti)))

    return Vt


def growth_len(ti, Ao, alpha, Lmax=1.0):
    '''
    An equation to compute the Gompertz growth curve given time points ti and the
    necessary parameters. Computes in terms of the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Lmax : float
        Specifies the maximum growth that the segment reaches. If Lmax is 1, the
        growth curve is normalized.

    '''
    Lt = Lmax * np.exp(-(Ao / alpha) * np.exp(-alpha * (ti)))

    return Lt

def growth_len_fitting(ti, Ao, alpha):
    '''
    An equation to compute the Gompertz growth curve given time points ti and the
    necessary parameters. Computes in terms of the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Lmax : float
        Specifies the maximum growth that the segment reaches. If Lmax is 1, the
        growth curve is normalized.

    '''
    Lt = np.exp(-(Ao / alpha) * np.exp(-alpha * (ti)))

    return Lt

def growth_fboost(Ao, alpha):

    Fboost = Ao/alpha

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
