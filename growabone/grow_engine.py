#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Main script for modeling human growth characteristics**

'''

# ....................{ IMPORTS                           }....................
from beartype import beartype
import numpy as np
import sympy as sp
from sympy.solvers.ode.systems import dsolve_system
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from scipy.misc import derivative
from scipy.integrate import odeint, cumulative_trapezoid
from scipy.special import erf


class Grower(object):

    def __init__(self, dat_name, dat_o, init_params=None):
        '''
        '''

        self.dat_name = str(dat_name)
        self.dat_o = np.asarray(dat_o)

        # Set initial guess for parameter values:
        if init_params is None:
            self.init_params = [0.32, 0.31, 0.15, 5, 7.5, 0.10, 3.1, 13]

        else:
            self.init_params = init_params

        self.process_data()

    def process_data(self):
        '''
        '''

        self.dat_n = self.dat_o / self.dat_o.max()  # normalize the raw growth curve data
        self.mults = 1 / self.dat_n  # Obtain the multiplier curve

        self.vel_o = np.gradient(self.dat_o, edge_order=2)  # Calculate the growth velocity of the raw data
        self.vel_n = np.gradient(self.dat_n, edge_order=2)  # Calculate the growth velocity of the normed data

        self.ln_dat_o = np.log(self.dat_o)  # Natural log of the raw data
        self.ln_dat_n = np.log(self.dat_n)  # Natural log of the normed data

        self.ln_vel_o = np.gradient(self.ln_dat_o, edge_order=2)  # Growth velocity of raw log data
        self.ln_vel_n = np.gradient(self.ln_dat_n, edge_order=2)  # Growth velocity of normed log data

        self.dat_Lmax = self.dat_o.max()  # store the Lmax value of the data curve

    def fit_growth_potential(self):
        '''
        '''
        popt, pcov = curve_fit(growth_potential,
                               dat_age,
                               ydata,  # Fit to male femur first
                               p0=self.init_params,
                               sigma=None,
                               absolute_sigma=False,
                               check_finite=True,
                               method='trf',  # lm’, ‘trf’, ‘dogbox’
                               jac=None)

        phi_fit = growth_potential(dat_age, popt[0],
                                   popt[1],
                                   popt[2],
                                   popt[3],
                                   popt[4],
                                   popt[5],
                                   popt[6],
                                   popt[7]
                                   )

        sum_resid = np.sum((phi_fit - ydata) ** 2)
        rmse = np.sqrt(sum_resid)

        error = np.sqrt(np.diag(pcov))

        phi_vect.append(phi_fit)
        params_vect.append(popt)
        error_vect.append(error)
        rmse_vect.append(rmse)

        lboost = np.exp((popt[0] / popt[1]) + 2 * popt[2] + 2 * popt[5])
        Lboost_vect.append(lboost)
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        lo = ldat[-1] / lboost
        Lo_vect.append(lo)

    def fit_gomperts(self):
        '''
        '''
        pass