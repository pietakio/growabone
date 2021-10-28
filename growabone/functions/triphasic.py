#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Triphasic growth functions (with unscaled Gaussians) for modeling human growth characteristics**

Leaving Gaussians unscaled leads to a messy final growth curve; however, it makes
working with 'time scaling', which maps one growth curve to another using a single multiplication
constant, much more feasible.

'''

import numpy as np
from scipy.special import erf


def growth_potential(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc):
    '''
    Computes the growth potential curve given time ti and the set of required
    parameters. Designed for data fitting. .

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Bo : float
        Related to the height of the first Gaussian growth pulse
    beta : float
        Related to the width of the first Gaussian growth pulse
    tb : float
        Specifies the centre (wrt time) of the first Gaussian growth pulse.
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''


    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    phi = (Ao*np.exp(-alpha*(ti)) +
           Bo*np.exp(-(1/(beta**2))*(ti - tb)**2) +
           Co*np.exp(-(1/(gamma**2))*(ti - tc)**2)
           )

    return phi


def growth_potential_components(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc):
    '''
    Computes the three components of the growth potential curve given time ti and the set of required
    parameters.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Bo : float
        Related to the height of the first Gaussian growth pulse
    beta : float
        Related to the width of the first Gaussian growth pulse
    tb : float
        Specifies the centre (wrt time) of the first Gaussian growth pulse.
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''

    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    gomp = Ao*np.exp(-alpha * (ti))
    gauss1 = Bo*np.exp(-(1 / (beta ** 2)) * (ti - tb) ** 2)
    gauss2 = Co*np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)

    return gomp, gauss1, gauss2


def growth_vel(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc):
    '''
    Computes the growth velocity curve given time ti and the set of required
    parameters.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Bo : float
        Related to the height of the first Gaussian growth pulse
    beta : float
        Related to the width of the first Gaussian growth pulse
    tb : float
        Specifies the centre (wrt time) of the first Gaussian growth pulse.
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''

    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    dLdt = np.exp(Ao*np.exp(-alpha * (ti)) +
                  Bo*np.exp(-(1 / (beta ** 2)) * (ti - tb) ** 2) +
                  Co*np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)
                  )

    return dLdt


def growth_len(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc, Lmax=1):
    '''
    Computes the growth curve given time ti and the set of required
    parameters given Lmax as a final parameter. Computes in terms of
    the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Bo : float
        Related to the height of the first Gaussian growth pulse
    beta : float
        Related to the width of the first Gaussian growth pulse
    tb : float
        Specifies the centre (wrt time) of the first Gaussian growth pulse.
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.
    Lmax : float
        Specifies the maximum growth that the segment reaches. If Lmax is 1, the
        growth curve is normalized.

    '''

    B = ((np.sqrt(np.pi*beta)*Bo)/2)
    C = ((np.sqrt(np.pi*gamma)*Co)/2)

    g_Lm = (-(Ao / alpha) * np.exp(-alpha * ti) +
            B*erf((ti - tb) / beta) +
            C*erf((ti - tc) / gamma) +
            np.log(Lmax) - B - C
            )

    Lt = np.exp(g_Lm)

    return Lt

def growth_len_fitting(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc):
    '''
    Computes the growth curve given time ti and the set of required
    parameters given Lmax as a final parameter. Computes in terms of
    the maximum (saturated) length.

    Parameters
    ----------
    ti : np.array
        Time points to evaluate the curve
    Ao : float
        Initial rate of growth
    alpha : float
        Decline in initial rate of growth
    Bo : float
        Related to the height of the first Gaussian growth pulse
    beta : float
        Related to the width of the first Gaussian growth pulse
    tb : float
        Specifies the centre (wrt time) of the first Gaussian growth pulse.
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.
    Lmax : float
        Specifies the maximum growth that the segment reaches. If Lmax is 1, the
        growth curve is normalized.

    '''

    B = ((np.sqrt(np.pi*beta)*Bo)/2)
    C = ((np.sqrt(np.pi*gamma)*Co)/2)

    g_Lm = (-(Ao / alpha) * np.exp(-alpha * ti) +
            B*erf((ti - tb) / beta) +
            C*erf((ti - tc) / gamma)  - B - C
            )

    Lt = np.exp(g_Lm)

    return Lt

def growth_fboost(Ao, alpha, Bo, beta, Co, gamma):

    Fboost = (Ao/alpha) + np.sqrt(np.pi*beta)*Bo + np.sqrt(np.pi*gamma)*Co

    return Fboost

def growth_mapper(A_o, alpha_o, B_o, beta_o, C_o, gamma_o, A_i, alpha_i, B_i, beta_i, C_i, gamma_i):
    '''
    Given two sets of growth potential parameters, find the scaling factor lambda that
    will map curve 'o' to curve 'i'.

    Parameters
    ---------------
    :param A_o:
    :param B_o:
    :param beta_o:
    :param C_o:
    :param gamma_o:
    :param A_i:
    :param B_i:
    :param beta_i:
    :param C_i:
    :param gamma_i:

    Returns
    ----------------
    '''

    # Calculate the boost factor for the i-data:
    Fboost_i = growth_fboost(A_i, alpha_i, B_i, beta_i, C_i, gamma_i)

    # For convenience, define some renormalized parameters
    Ao_n = A_o/alpha_o
    Bo_n = np.sqrt(np.pi*beta_o)*B_o
    Co_n = np.sqrt(np.pi*gamma_o)*C_o

    # The scaling equation is quadradic wrt the scaling parameter lamb; therefore solve
    # using the quadradic formula:
    # FIXME: I believe only the positive solution (lamb > 0) applies, but will return both for a bit
    lamb_p = ((Bo_n + Co_n) + np.sqrt((Bo_n + Co_n)**2 + 4*Fboost_i*Ao_n))/(2*Fboost_i)

    lamb_n = ((Bo_n + Co_n) - np.sqrt((Bo_n + Co_n)**2 + 4*Fboost_i*Ao_n))/(2*Fboost_i)

    return lamb_p, lamb_n

