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
from scipy.optimize import curve_fit, minimize, basinhopping
from growabone.functions import triphasic as gf
from growabone.functions import triphasic_gscaled as gfs
from growabone.functions import gompertz as gomp
from growabone.functions import logistic as logi
from warnings import warn


class Grower(object):

    def __init__(self, dat_time, dat_name, dat_o, init_params=None):
        '''
        '''

        self.dat_time = dat_time
        self.dat_name = str(dat_name)
        self.dat_o = np.asarray(dat_o)

        self.del_t = np.gradient(self.dat_time, edge_order=2)

        # Set initial guess for parameter values:
        if init_params is None:
            # self.init_params = [0.42, 0.5, 0.12, 4.96, 6.31, 0.06, 3.11, 12.05]
            # self.init_params = [0.369,  0.355,  0.034,  3.343,  6.912,  0.04 ,  3.369, 11.662] # bone data
            self.init_params = [0.029, 0.52, 0.0037, 3.0, 6.5, 0.0039, 2.3, 11.662] # Height data
        else:
            self.init_params = init_params

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

        try:
            self.fit_growth_potential()
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

        #FIXME: WE need an RBF gradient for the data s the numpy version SUCKS!
        self.vel_o = np.gradient(self.dat_o, edge_order=2)/self.del_t  # Calculate the growth velocity of the raw data
        self.vel_n = np.gradient(self.dat_n, edge_order=2)/self.del_t  # Calculate the growth velocity of the normed data

        self.ln_dat_o = np.log(self.dat_o)  # Natural log of the raw data
        self.ln_dat_n = np.log(self.dat_n)  # Natural log of the normed data

        self.ln_vel_o = np.gradient(self.ln_dat_o, edge_order=2)/self.del_t  # Growth velocity of raw log data
        self.ln_vel_n = np.gradient(self.ln_dat_n, edge_order=2)/self.del_t  # Growth velocity of normed log data

    def fit_triphasic(self):
        '''
        '''

        # Minimize method: ----------------------------------------
        params = self.init_params

        # Bounds on parameter values
        minb = 0.0
        maxb = np.inf
        boundsv = [(minb, maxb) for i in range(len(params))]
        boundsv[-1] = (2.0, 18.0)  # Constrain the puberty peak to acheive a better fit

        sol0 = minimize(gf.growth_len_fitting_f2,
                        params,
                        args=(self.dat_time, self.dat_n),
                        method='trust-constr',
                        bounds=boundsv,
                        )

        # sol0 = minimize(gf.growth_pot_fitting,
        #                 params,
        #                 args=(self.dat_time, self.ln_vel_n),
        #                 method='trust-constr',
        #                 bounds=boundsv,
        #                 )

        fit_params = sol0.x
        self.rmse = sol0.fun

        # Calculate parameter covariance matrix using the Jacobian:
        jac = gf.growth_jacobian_f2(fit_params, self.dat_time)
        param_covariance = np.linalg.pinv(jac.T.dot(jac))*self.rmse**2

        #---------------------------------------------------------
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
        self.mse_bytri = sum_resid
        self.rmse_bytri = np.sqrt(sum_resid)

        self.fboost_bytri = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3], fit_params[4],
                                       fit_params[5], fit_params[6], fit_params[7])
        #     error_lboost =
        ## FIXME: WHAT IS THE ERROR ON THE Lboost value given we have error on each param
        ## AND can propegate it...

        # predict initial length from lboost and final length:
        self.Lo_bytri = self.dat_Lmax / np.exp(self.fboost_bytri)

        # Now that we have the parameters, compute components of the triphasic model's growth potential:
        self.phi_gomp1, self.phi_gauss1, self.phi_gauss2 = gf.growth_potential_components(self.dat_time,
                                                                                           fit_params[0],
                                                                                           fit_params[1],
                                                                                           fit_params[2],
                                                                                           fit_params[3],
                                                                                           fit_params[4],
                                                                                           fit_params[5],
                                                                                           fit_params[6],
                                                                                           fit_params[7],
                                                                                           )

        self.fit_vel_bytri = gf.growth_vel(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7],
                                   self.dat_Lmax)


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

        # Predict growth velocity:
        self.fit_vel_bygomp = gomp.growth_vel(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax)


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

        # Predict growth velocity:
        self.fit_vel_bylogi = logi.growth_vel(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax)

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
                                                 bounds=(0.0, np.inf),
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

        # # Now that we have the parameters, compute components of the triphasic model's growth potential:
        # self.phi_gomp1, self.phi_gauss1, self.phi_gauss2 = gf.growth_potential_components(self.dat_time,
        #                                                                                    fit_params[0],
        #                                                                                    fit_params[1],
        #                                                                                    fit_params[2],
        #                                                                                    fit_params[3],
        #                                                                                    fit_params[4],
        #                                                                                    fit_params[5],
        #                                                                                    fit_params[6],
        #                                                                                    fit_params[7],
        #                                                                                    )

