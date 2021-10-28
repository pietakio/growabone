#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Things we tested that don't seem to work very well...**

'''
from beartype import beartype
import numpy as np
from scipy.optimize import curve_fit
from growabone.functions import triphasic as gf
# from growabone.functions import triphasic_gscaled as gf
from growabone.functions import gompertz as gomp

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
    sum_resid = np.sum((self.fit_o - self.dat_o) ** 2)
    self.rmse = np.sqrt(sum_resid)

    self.fboost = gf.growth_fboost(fit_params[0], fit_params[1],
                                   fit_params[2], fit_params[3],
                                   fit_params[5], fit_params[6])
    #     error_lboost =
    ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
    ## AND can propegate it...

    # predict initial length from lboost and final length:
    self.Lo = self.dat_Lmax / self.fboost