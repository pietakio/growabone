#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Growth function models based on the Preece-Baines mathematical models of human growth curves**

Preece-Baines model (Preece, Baines, Annals of Human Biology, V5, 1978) is based on a logistic-equation differential
generating equation and appears to produce excellent fits to growth curves with fewer parameters.

'''

import numpy as np
import sympy as sp

def growth_potential(ti, h_1, h_theta, s_o, s_1, theta):
    '''
    An equation to compute the Gompertz-curve growth potential given time points ti and the
    necessary parameters.
    '''
    h_1o, h_thetao, s_oo, s_1o, thetao, p_oo, p_1o, q_1o, to = sp.symbols('h_1, h_theta, s_o, s_1, theta, p_o, p_1, q_1, t',
                                                                 real=True, positive=True)
    h_m1 = h_1o - ((2 * (h_1o - h_thetao)) / (sp.exp(s_oo * (to - thetao)) + sp.exp(s_1o * (to - thetao))))  # Growth curve

    gp_m1 = sp.diff(sp.log(h_m1), to)  # Growth potential, analytical

    gp_m1_f = sp.lambdify([h_1o, h_thetao, s_oo, s_1o, thetao, to], gp_m1)

    phi = gp_m1_f(h_1, h_theta, s_o, s_1, theta, ti)

    return phi


def growth_vel(ti, h_1, h_theta, s_o, s_1, theta):
    '''
    An equation to compute the Gompertz growth velocity given time points ti and the
    necessary parameters.
    '''

    h_1o, h_thetao, s_oo, s_1o, thetao, p_oo, p_1o, q_1o, to = sp.symbols('h_1, h_theta, s_o, s_1, theta, p_o, p_1, q_1, t',
                                                                 real=True, positive=True)
    h_m1 = h_1o - ((2 * (h_1o - h_thetao)) / (sp.exp(s_oo * (to - thetao)) + sp.exp(s_1o * (to - thetao))))  # Growth curve

    v_m1 = sp.diff(h_m1, to)  # Growth velocity, analytical

    v_m1_f = sp.lambdify([h_1o, h_thetao, s_oo, s_1o, thetao, to], v_m1)

    Vt = v_m1_f(h_1, h_theta, s_o, s_1, theta, ti)

    return Vt


def growth_len(ti, h_1, h_theta, s_o, s_1, theta):
    '''
    An equation to compute the Preece-Baines model 1 growth curve given time points ti and the
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
    Lt = h_1 - ((2*(h_1 - h_theta))/(np.exp(s_o*(ti-theta)) + np.exp(s_1*(ti-theta)))) # Growth curve

    return Lt

def growth_len_fitting(ti, h_1, h_theta, s_o, s_1, theta):
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
    Lt = h_1 - ((2*(h_1 - h_theta))/(np.exp(s_o*(ti-theta)) + np.exp(s_1*(ti-theta)))) # Growth curve

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