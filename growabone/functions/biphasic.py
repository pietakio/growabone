#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Biphasic growth functions (with unscaled Gaussians) for modeling human growth characteristics**

Leaving Gaussians unscaled leads to a messy final growth curve; however, it makes
working with 'time scaling', which maps one growth curve to another using a single multiplication
constant, much more feasible.

'''

import numpy as np
from scipy.special import erf


def growth_potential(ti, Ao, alpha, Co, gamma, tc):
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
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''


    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    phi = (Ao*np.exp(-alpha*(ti)) +
           Co*np.exp(-(1/(gamma**2))*(ti - tc)**2)
           )

    return phi

def growth_pot_fitting(params, ti, ydata):
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
    Ao = params[0]
    alpha = params[1]
    Co = params[5]
    gamma = params[6]
    tc = params[7]

    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    phi = (Ao*np.exp(-alpha*(ti)) +
           Co*np.exp(-(1/(gamma**2))*(ti - tc)**2)
           )

    rmse = np.sqrt(np.mean((phi - ydata)**2))
    # ss = np.sum((phi - ydata)**2)

    return rmse

def growth_potential_components(ti, Ao, alpha, Co, gamma, tc):
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
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''

    gomp = Ao*np.exp(-alpha * (ti))
    gauss2 = Co*np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)

    return gomp, gauss1, gauss2

def growth_vel(ti, Ao, alpha, Co, gamma, tc, Lmax=1):
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
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''

    Bp = (np.sqrt(np.pi)*Bo*beta)/2
    Cp = (np.sqrt(np.pi)*Co*gamma)/2

    Vt = (Lmax*(Ao*np.exp(-alpha*(ti)) +
                Co*np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2))*(
                np.exp(-(Ao/alpha)*np.exp(-alpha*(ti)) +
                Cp*(erf((ti - tc) / gamma) - 1)
                )
                )
          )


    return Vt

def growth_vel_fitting(ti, Ao, alpha, Co, gamma, tc):
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
    Co : float
        Related to the height of the second Gaussian growth pulse
    gamma : float
        Related to the width of the second Gaussian growth pulse
    tc : float
        Specifies the centre (wrt time) of the second Gaussian growth pulse.

    '''

    Cp = (np.sqrt(np.pi)*Co*gamma)/2

    Vt = ((Ao*np.exp(-alpha*(ti)) +
                Co*np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2))*(
                np.exp(-(Ao/alpha)*np.exp(-alpha*(ti)) +
                Cp*(erf((ti - tc) / gamma) - 1)
                )
                )
          )


    return Vt

def growth_len(ti, Ao, alpha, Co, gamma, tc, Lmax=1):
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

    Cp = ((np.sqrt(np.pi)*Co*gamma)/2)

    g_Lm = (-(Ao/alpha)*np.exp(-alpha*ti) +
            Cp*erf((ti - tc)/gamma) +
            np.log(Lmax) - Cp
            )

    Lt = np.exp(g_Lm)

    return Lt

def growth_len_fitting(ti, Ao, alpha, Co, gamma, tc):
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

    Cp = ((np.sqrt(np.pi)*Co*gamma)/2)

    g_Lm = (-(Ao/alpha)*np.exp(-alpha*ti) +
            Cp*erf((ti - tc)/gamma)  - Cp
            )

    Lt = np.exp(g_Lm)

    return Lt

def growth_len_fitting_f2(params, ti, ydata):
    '''
    Computes the growth curve given time ti and the set of required
    parameters given Lmax as a final parameter. Computes in terms of
    the maximum (saturated) length. Function is formatted for stochastic
    optimization methods such as basinhopping in scipy optimize.

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

    Ao = params[0]
    alpha = params[1]
    Co = params[5]
    gamma = params[6]
    tc = params[7]

    Cp = ((np.sqrt(np.pi)*Co*gamma)/2)

    g_Lm = (-(Ao/alpha)*np.exp(-alpha*ti) +
            Cp*erf((ti - tc)/gamma) - Cp
            )

    Lt = np.exp(g_Lm)

    rmse = np.sqrt(np.mean((Lt - ydata) ** 2)) # These functions fit better to rmse
    # ss = np.sum(np.sqrt((Lt - ydata) ** 2))
    # mse = np.sum((Lt - ydata) ** 2)

    return rmse

def growth_fboost(Ao, alpha, Co, gamma, tc):

    Cp = ((np.sqrt(np.pi)*Co*gamma)/2)

    Fboost = (Ao/alpha) + Cp*(1 + erf(tc/gamma))

    Fa = (Ao/alpha) # first component
    Fc = Cp*(1 + erf(tc/gamma)) # third component

    return Fboost, Fa, Fc

def error_fboost(Ao, alpha, Co, gamma, tc, cov):

    Cp = ((np.sqrt(np.pi)*Co*gamma)/2)

    Fa = (Ao/alpha) # first component
    Fc = Cp*(1 + erf(tc/gamma)) # third component

    sd_A = np.sqrt(cov[0, 0])
    sd_a = np.sqrt(cov[1, 1])
    sd_C = np.sqrt(cov[5, 5])
    sd_gamma = np.sqrt(cov[6, 6])

    cov_Aa = cov[0, 1]
    cov_Cc = cov[5, 6]

    sd_Fa = np.sqrt((Fa ** 2) * ((sd_A / Ao) ** 2 + (sd_a / alpha) ** 2 - 2 * (cov_Aa / (Ao * alpha))))
    sd_Fc = np.sqrt((Fc ** 2) * ((sd_C / Co) ** 2 + (sd_gamma / gamma) ** 2 + 2 * (cov_Cc / (Co * gamma))))
    sd_F = np.sqrt(sd_Fa ** 2 + sd_Fc ** 2)

    return sd_F, sd_Fa, sd_Fc

def growth_jacobian(t, A, a, C, c, t_c):

    # L_max = kwargs['L_max']
    L_max = 1.0
    # t = kwargs['t']

    Cp = (np.sqrt(np.pi)/2)*C*c

    dLdA = (-(L_max / a) * np.exp(-a * t) *
            np.exp(-(A / a) * np.exp(-a * t)  + Cp * (erf((t - t_c) / c) - 1)))

    dLda = (L_max * (((A * t * np.exp(-a * t)) / a + (A * np.exp(-a * t)) / a ** 2)) *
            np.exp(-(A / a) * np.exp(-a * t) +  Cp * (erf((t - t_c) / c) - 1)))

    dLdC = ((np.sqrt(np.pi)) * (L_max / 2) * c * (erf((t - t_c) / c) - 1) *
            np.exp(-(A / a) * np.exp(-a * t) +  Cp * (erf((t - t_c) / c) - 1)))

    dLdc = (L_max * ((((np.sqrt(np.pi) * C) / 2) * (erf((t - t_c) / c) - 1)) -
                     ((C * (t - t_c)) / (c)) * np.exp(-(1 / c ** 2) * (t - t_c) ** 2)) *
            np.exp(-(A / a) * np.exp(-a * t) + Cp * (erf((t - t_c) / c) - 1)))

    dLdtc = (-C * L_max * np.exp(-(1 / c ** 2) * (t - t_c) ** 2) *
             np.exp(
                 -(A / a) * np.exp(-a * t) + Cp * (erf((t - t_c) / c) - 1)))

    jac = np.asarray([dLdA, dLda, dLdC, dLdc, dLdtc]).T

    return jac

def growth_jacobian_f2(x, t, L_max):

    A = x[0]
    a = x[1]
    C = x[5]
    c = x[6]
    t_c = x[7]

    Cp = (np.sqrt(np.pi)/2)*C*c

    dLdA = (-(L_max / a) * np.exp(-a * t) *
            np.exp(-(A / a) * np.exp(-a * t) + Cp * (erf((t - t_c) / c) - 1)))

    dLda = (L_max * (((A * t * np.exp(-a * t)) / a + (A * np.exp(-a * t)) / a ** 2)) *
            np.exp(-(A / a) * np.exp(-a * t)  + Cp * (erf((t - t_c) / c) - 1)))


    dLdC = ((np.sqrt(np.pi)) * (L_max / 2) * c * (erf((t - t_c) / c) - 1) *
            np.exp(-(A / a) * np.exp(-a * t)  + Cp * (erf((t - t_c) / c) - 1)))

    dLdc = (L_max * ((((np.sqrt(np.pi) * C) / 2) * (erf((t - t_c) / c) - 1)) -
                     ((C * (t - t_c)) / (c)) * np.exp(-(1 / c ** 2) * (t - t_c) ** 2)) *
            np.exp(-(A / a) * np.exp(-a * t)  + Cp * (erf((t - t_c) / c) - 1)))

    dLdtc = (-C * L_max * np.exp(-(1 / c ** 2) * (t - t_c) ** 2) *
             np.exp(
                 -(A / a) * np.exp(-a * t)  + Cp * (erf((t - t_c) / c) - 1)))

    jac = np.asarray([dLdA, dLda, dLdC, dLdc, dLdtc]).T

    return jac

# FIXME: GROWTH MAPPER needs to be updated for the unscaled growth potentials...
# def growth_mapper(A_o, alpha_o, B_o, beta_o, C_o, gamma_o, A_i, alpha_i, B_i, beta_i, C_i, gamma_i):
#     '''
#     Given two sets of growth potential parameters, find the scaling factor lambda that
#     will map curve 'o' to curve 'i'.
#
#     Parameters
#     ---------------
#     :param A_o:
#     :param B_o:
#     :param beta_o:
#     :param C_o:
#     :param gamma_o:
#     :param A_i:
#     :param B_i:
#     :param beta_i:
#     :param C_i:
#     :param gamma_i:
#
#     Returns
#     ----------------
#     '''
#
#     # Calculate the boost factor for the i-data:
#     Fboost_i = growth_fboost(A_i, alpha_i, B_i, beta_i, C_i, gamma_i)
#
#     # For convenience, define some renormalized parameters
#     Ao_n = A_o/alpha_o
#     Bo_n = np.sqrt(np.pi*beta_o)*B_o
#     Co_n = np.sqrt(np.pi*gamma_o)*C_o
#
#     # The scaling equation is quadradic wrt the scaling parameter lamb; therefore solve
#     # using the quadradic formula:
#     # FIXME: I believe only the positive solution (lamb > 0) applies, but will return both for a bit
#     lamb_p = ((Bo_n + Co_n) + np.sqrt((Bo_n + Co_n)**2 + 4*Fboost_i*Ao_n))/(2*Fboost_i)
#
#     lamb_n = ((Bo_n + Co_n) - np.sqrt((Bo_n + Co_n)**2 + 4*Fboost_i*Ao_n))/(2*Fboost_i)
#
#     return lamb_p, lamb_n

