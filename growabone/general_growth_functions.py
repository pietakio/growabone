#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**General functions for modeling human growth characteristics**

'''

def gomp_potential(ti, Ao, alpha):
    '''
    An equation to compute the Gompertz-curve growth potential given time points ti and the
    necessary parameters.
    '''
    phi = Ao * np.exp(-alpha * (ti))

    return phi


def gomp_velocity(ti, Ao, alpha):
    '''
    An equation to compute the Gompertz growth velocity given time points ti and the
    necessary parameters.
    '''
    phi = np.exp(Ao * np.exp(-alpha * (ti)))

    return phi


def gomp_Lmax(ti, Ao, alpha, Lmax=1.0):
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