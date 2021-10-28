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
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from growabone.functions import triphasic as gf
from growabone.functions import triphasic_gscaled as gfs
from growabone.functions import gompertz as gomp
from growabone.functions import logistic as logi
from warnings import warn


class Grower(object):

    def __init__(self, dat_time, dat_name, dat_o, init_params=None, init_params_gscaled=None):
        '''
        '''

        self.dat_time = dat_time
        self.dat_name = str(dat_name)
        self.dat_o = np.asarray(dat_o)

        # Set initial guess for parameter values:
        if init_params is None:
            # self.init_params = [0.40, 0.40, 0.045, 4.8, 7.5, 0.03, 2.5, 12.3] # bone data, unscaled gaussians
            self.init_params = [0.42, 0.5, 0.12, 4.96, 6.31, 0.06, 3.11, 12.05]
        else:
            self.init_params = init_params

        if init_params_gscaled is None:
            self.init_params_gscaled = [0.42, 0.5, 0.25, 4.96, 6.31, 0.09, 3.11, 12.05] # bone data, scaled gaussians
        else:
            self.init_params_gscaled = init_params_gscaled

        self.process_data()

        try:
            self.fit_triphasic()
        except RuntimeError as exception:
            warn(str(exception), UserWarning)

        try:
            self.fit_gompertz()
        except RuntimeError as exception:
            warn(str(exception), UserWarning)

        try:
            self.fit_logistic()
        except RuntimeError as exception:
            warn(str(exception), UserWarning)

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

    def fit_triphasic(self):
        '''
        '''

        fit_params, param_covariance = curve_fit(gf.growth_len_fitting,
                                                 self.dat_time,
                                                 self.dat_n,  # Fit to normalized growth curve
                                                 p0=self.init_params,
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 bounds=(0.0, np.inf),
                                                 jac=None)

        self.params_bytri = fit_params
        self.param_error_bytri = np.sqrt(np.diag(param_covariance))

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_bytri = gf.growth_len(self.dat_time,
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
        sum_resid = np.sum((self.fit_o_bytri - self.dat_o) ** 2)
        self.rmse_bytri = np.sqrt(sum_resid)

        self.fboost_bytri = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3],
                                       fit_params[5], fit_params[6])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bytri = self.dat_Lmax / np.exp(self.fboost_bytri)

    def fit_triphasic_gscaled(self):
        '''
        '''

        fit_params, param_covariance = curve_fit(gfs.growth_len_fitting,
                                                 self.dat_time,
                                                 self.dat_n,  # Fit to normalized growth curve
                                                 p0=self.init_params_gscaled,
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 bounds=(0.0, np.inf),
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 jac=None)

        self.params_bytri_gs = fit_params
        self.param_error_bytri_gs = np.sqrt(np.diag(param_covariance))

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_bytri_gs = gfs.growth_len(self.dat_time,
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
        sum_resid = np.sum((self.fit_o_bytri_gs - self.dat_o) ** 2)
        self.rmse_bytri_gs = np.sqrt(sum_resid)

        self.fboost_bytri_gs = gfs.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3],
                                       fit_params[5], fit_params[6])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bytri_gs = self.dat_Lmax / np.exp(self.fboost_bytri_gs)

    def fit_gompertz(self):
        '''
        '''
        fit_params, gomp_covariance = curve_fit(gomp.growth_len_fitting,
                                                 self.dat_time,
                                                 self.dat_n,  # Fit to normalized growth curve
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 bounds=(0.0, np.inf),
                                                 jac=None)

        self.params_bygomp = fit_params
        self.param_error_bygomp = np.sqrt(np.diag(gomp_covariance))

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
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param?
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bygomp = self.dat_Lmax / np.exp(self.fboost_bygomp)



    def fit_logistic(self):
        '''
        '''
        fit_params, logi_covariance = curve_fit(logi.growth_len_fitting,
                                                 self.dat_time,
                                                 self.dat_n,  # Fit to normalized growth curve
                                                 sigma=None,
                                                 absolute_sigma=False,
                                                 check_finite=True,
                                                 method='trf',  # lm’, ‘trf’, ‘dogbox’
                                                 jac=None)

        # Compute the fit for use of these parameters on the original growth curve data:
        self.fit_o_bylogi = logi.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax
                                   )
        sum_resid = np.sum((self.fit_o_bylogi - self.dat_o) ** 2)
        self.rmse_bylogi = np.sqrt(sum_resid)

        self.fboost_bylogi = logi.growth_fboost(fit_params[0], fit_params[1])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param?
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bylogi = self.dat_Lmax / np.exp(self.fboost_bylogi)

        self.params_bylogi = fit_params
        self.param_error_bylogi = np.sqrt(np.diag(logi_covariance))