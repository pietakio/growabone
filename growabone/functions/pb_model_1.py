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

h_1o, h_thetao, s_oo, s_1o, thetao, to = sp.symbols('h_1, h_theta, s_o, s_1, theta, t',
                                                                      real=True, positive=True)
h_m1 = h_1o - ((2 * (h_1o - h_thetao)) / (sp.exp(s_oo * (to - thetao)) + sp.exp(s_1o * (to - thetao))))  # Growth curve

v_m1 = sp.diff(h_m1, to)  # Growth velocity, analytical

v_m1_f = sp.lambdify([h_1o, h_thetao, s_oo, s_1o, thetao, to], v_m1)

def growth_vel(ti, h_1, h_theta, s_o, s_1, theta):
    '''
    An equation to compute the Gompertz growth velocity given time points ti and the
    necessary parameters.
    '''

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

def growth_len_fitting(params, ti, ydata):
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
    h_1 = params[0]
    h_theta = params[1]
    s_o = params[2]
    s_1= params[3]
    theta = params[4]

    Lt = growth_len(ti, h_1, h_theta, s_o, s_1, theta)

    rmse = np.sqrt(np.mean((Lt - ydata)**2))
    # ss = np.sum((Lt - ydata)**2)

    return rmse

def growth_vel_fitting(params, ti, ydata):
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
    h_1 = params[0]
    h_theta = params[1]
    s_o = params[2]
    s_1= params[3]
    theta = params[4]

    Vt = growth_vel(ti, h_1, h_theta, s_o, s_1, theta)

    rmse = np.sqrt(np.mean((Vt - ydata)**2))
    # ss = np.sum((Vt - ydata)**2)
    return rmse