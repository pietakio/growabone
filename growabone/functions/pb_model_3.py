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


def growth_vel(ti, h_1, h_theta, p_o, p_1, q_1, theta):
    '''
    An equation to compute the Gompertz growth velocity given time points ti and the
    necessary parameters.
    '''

    h_1o, h_thetao, thetao, p_oo, p_1o, q_1o, to = sp.symbols('h_1, h_theta, s_o, s_1, theta, p_o, p_1, q_1, t',
                                                                 real=True, positive=True)
    h_m3 = (h_1o -
            ((4*(h_1o - h_thetao))/((sp.exp(p_oo*(to-thetao)) +
                                     sp.exp(p_1o*(to-thetao)))*(1 + sp.exp(q_1o*(to-thetao))))))

    v_m3 = sp.diff(h_m3, to)  # Growth velocity, analytical

    v_m3_f = sp.lambdify([h_1o, h_thetao, p_oo, p_1o, q_1o, thetao, to], v_m3)

    Vt = v_m3_f(h_1, h_theta, p_o, p_1, q_1, theta, ti)

    return Vt


def growth_len(ti, h_1, h_theta, p_o, p_1, q_1, theta):
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

    Lt = h_1 - ((4 * (h_1 - h_theta)) / (
                (sp.exp(p_o * (ti - theta)) + sp.exp(p_1 * (ti - theta))) * (1 + sp.exp(q_1 * (ti - theta)))))

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
    p_o = params[2]
    p_1= params[3]
    q_1 = params[4]
    theta = params[5]

    Lt = h_1 - ((4 * (h_1 - h_theta)) / (
                (sp.exp(p_o * (ti - theta)) + sp.exp(p_1 * (ti - theta))) * (1 + sp.exp(q_1 * (ti - theta)))))

    rmse = np.sqrt(np.mean((Lt - ydata)**2))


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
    p_o = params[2]
    p_1= params[3]
    q_1 = params[4]
    theta = params[5]

    Vt = growth_vel(ti, h_1, h_theta, p_o, p_1, q_1, theta)

    rmse = np.sqrt(np.mean((Vt - ydata)**2))

    return rmse
