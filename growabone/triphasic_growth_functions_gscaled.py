#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Triphasic growth functions (with scaled Gaussians) for modeling human growth characteristics**

'''


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
    phi = (Ao * np.exp(-alpha * (ti)) +
           ((2 * Bo) / (beta * np.sqrt(np.pi))) * np.exp(-(1 / (beta ** 2)) * (ti - tb) ** 2) +
           ((2 * Co) / (gamma * np.sqrt(np.pi))) * np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)
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
    gomp = Ao * np.exp(-alpha * (ti))
    gauss1 = ((2 * Bo) / (beta * np.sqrt(np.pi))) * np.exp(-(1 / (beta ** 2)) * (ti - tb) ** 2)
    gauss2 = ((2 * Co) / (gamma * np.sqrt(np.pi))) * np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)

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
    dLdt = np.exp(Ao * np.exp(-alpha * (ti)) +
                  ((2 * Bo) / (beta * np.sqrt(np.pi))) * np.exp(-(1 / (beta ** 2)) * (ti - tb) ** 2) +
                  ((2 * Co) / (gamma * np.sqrt(np.pi))) * np.exp(-(1 / (gamma ** 2)) * (ti - tc) ** 2)
                  )

    return dLdt


def growth_Lmax(ti, Ao, alpha, Bo, beta, tb, Co, gamma, tc, Lmax=1):
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

    # use my normalization for the gaussian, which gives the cleanest expression for L(t)
    g_Lm = (-(Ao / alpha) * np.exp(-alpha * ti) +
            Bo * erf((ti - tb) / beta) +
            Co * erf((ti - tc) / gamma) +
            np.log(Lmax) - Bo - Co
            )

    Lt = np.exp(g_Lm)

    return Lt


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