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
from growabone.functions import triphasic as gf
from growabone.functions import gompertz as gomp


class Grower(object):

    def __init__(self, dat_time, dat_name, dat_o, init_params=None):
        '''
        '''

        self.dat_time = dat_time
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
        Compute data series normalizations, derivatives and their logarithmic counterparts
        from the raw data series growth curve.

        '''
        self.dat_Lmax = np.max(self.dat_o)  # store the Lmax value of the data curve

        self.dat_n = self.dat_o / self.dat_Lmax  # normalize the raw growth curve data
        self.mults = 1 / self.dat_n  # Obtain the multiplier curve

        self.vel_o = np.gradient(self.dat_o, edge_order=2)  # Calculate the growth velocity of the raw data
        self.vel_n = np.gradient(self.dat_n, edge_order=2)  # Calculate the growth velocity of the normed data

        self.ln_dat_o = np.log(self.dat_o)  # Natural log of the raw data
        self.ln_dat_n = np.log(self.dat_n)  # Natural log of the normed data

        self.ln_vel_o = np.gradient(self.ln_dat_o, edge_order=2)  # Growth velocity of raw log data
        self.ln_vel_n = np.gradient(self.ln_dat_n, edge_order=2)  # Growth velocity of normed log data

    def fit_growth_potential(self):
        '''
        '''

        fit_params, param_covariance = curve_fit(gf.growth_potential,
                               self.dat_time,
                               self.ln_vel_o,  # Fit to derivative of log of unnormalized growth curve
                               p0=self.init_params,
                               sigma=None,
                               absolute_sigma=False,
                               check_finite=True,
                               method='trf',  # lm’, ‘trf’, ‘dogbox’
                               jac=None)

        self.phi_fit = gf.growth_potential(self.dat_time, fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7]
                                   )

        phi_sum_resid = np.sum((self.phi_fit - self.ln_vel_o) ** 2)
        self.phi_rmse = np.sqrt(phi_sum_resid)
        self.phi_params_error = np.sqrt(np.diag(param_covariance))
        self.phi_params = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o = gf.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7],
                                   self.dat_Lmax
                                   )
        sum_resid = np.sum((self.fit_o - self.dat_o)**2)
        self.rmse = np.sqrt(sum_resid)

        self.fboost = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2],fit_params[3],
                                       fit_params[5], fit_params[6])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo = self.dat_Lmax/self.fboost

    def fit_growth_velocity(self):
        '''
        '''

        fit_params, vel_covariance = curve_fit(gf.growth_vel,
                                                 self.dat_time,
                                                 self.vel_o,  # Fit to derivative of unnormalized growth curve
                                                 p0=self.init_params,
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 jac=None)

        self.vel_fit = gf.growth_vel(self.dat_time, fit_params[0],
                                           fit_params[1],
                                           fit_params[2],
                                           fit_params[3],
                                           fit_params[4],
                                           fit_params[5],
                                           fit_params[6],
                                           fit_params[7]
                                           )

        vel_sum_resid = np.sum((self.vel_fit - self.vel_o) ** 2)
        self.vel_rmse = np.sqrt(vel_sum_resid)
        self.vel_params_error = np.sqrt(np.diag(vel_covariance))
        self.vel_params = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_byvel = gf.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7],
                                   self.dat_Lmax
                                   )
        sum_resid = np.sum((self.fit_o_byvel - self.dat_o) ** 2)
        self.rmse_byvel = np.sqrt(sum_resid)

        self.fboost_byvel = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3],
                                       fit_params[5], fit_params[6])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_byvel = self.dat_Lmax / self.fboost_byvel

    def fit_gompertz_potential(self):
        '''
        '''
        fit_params, gomp_covariance = curve_fit(gomp.growth_potential,
                                                 self.dat_time,
                                                 self.ln_vel_o,  # Fit to derivative of log of unnormalized growth curve
                                                 # p0=self.init_params,
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 jac=None)

        self.gomp_fit = gomp.growth_potential(self.dat_time, fit_params[0],
                                           fit_params[1],
                                           )

        gomp_sum_resid = np.sum((self.gomp_fit - self.ln_vel_o) ** 2)
        self.gomp_rmse = np.sqrt(gomp_sum_resid)
        self.gomp_params_error = np.sqrt(np.diag(gomp_covariance))
        self.gomp_params = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_bygomp = gomp.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax
                                   )
        sum_resid = np.sum((self.fit_o_bygomp - self.dat_o) ** 2)
        self.rmse_bygomp = np.sqrt(sum_resid)

        self.fboost_bygomp = gomp.growth_fboost(fit_params[0], fit_params[1])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bygomp = self.dat_Lmax / self.fboost_bygomp

    def fit_gompertz_velocity(self):
        '''
        '''
        fit_params, gomp_covariance = curve_fit(gomp.growth_vel,
                                                 self.dat_time,
                                                 self.vel_o,  # Fit to derivative of unnormalized growth curve
                                                 # p0=self.init_params,
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 jac=None)

        self.gomp_vel_fit = gomp.growth_vel(self.dat_time, fit_params[0],
                                           fit_params[1],
                                           )

        gomp_vel_sum_resid = np.sum((self.gomp_vel_fit - self.vel_o) ** 2)
        self.gomp_vel_rmse = np.sqrt(gomp_vel_sum_resid)
        self.gomp_vel_params_error = np.sqrt(np.diag(gomp_covariance))
        self.gomp_vel_params = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_bygompvel = gomp.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax
                                   )
        sum_resid = np.sum((self.fit_o_bygompvel - self.dat_o) ** 2)
        self.rmse_bygompvel = np.sqrt(sum_resid)

        self.fboost_bygompvel = gomp.growth_fboost(fit_params[0], fit_params[1])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bygompvel = self.dat_Lmax / self.fboost_bygompvel