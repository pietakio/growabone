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
from scipy.optimize import curve_fit, minimize, dual_annealing
from scipy.interpolate import CubicSpline
from growabone.functions import triphasic as gf
from growabone.functions import gompertz as gomp
from growabone.functions import pb_model_1 as pb1
from growabone.functions import pb_model_3 as pb3
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
            # self.init_params = [0.369,  0.355,  0.034,  3.343,  6.912,  0.04 ,  3.369, 11.662] # bone data
            # self.init_params = [0.029, 0.52, 0.0037, 3.0, 6.5, 0.0039, 2.3, 11.662] # Height data
            self.init_params = [0.42, 0.5, 0.05, 1.0, 8.0, 0.05, 1.0, 13.5]
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
        # params = np.asarray([0.5, 1.3, 0.06, 3.0, 3.0, 0.04, 2.0, 13.0])
        params = np.asarray([0.1, 0.1, 0.06, 3.0, 3.0, 0.04, 2.0, 13.0])

        # Bounds on parameter values
        minb = 0.0
        maxb = 100.0
        boundsv = [(minb, maxb) for i in range(len(params))]

        ## Constraints:
        boundsv[0] = (0.0, 1.0)  # Exponential phase max
        boundsv[1] = (0.0, 20.0)  # Exponential phase decay
        boundsv[2] = (0.0, 0.2)  # Childhood Gaussian peak height
        boundsv[3] = (0.1, 10.0)  # Childhood Gaussian width
        boundsv[4] = (0.0, 10.0)  # Childhood Gaussian peak center
        boundsv[5] = (0.0, 0.2)  # Teenage Gaussian peak height
        boundsv[6] = (0.1, 5.0)  # Teenage Gaussian peak width
        boundsv[7] = (8.0, 20.0)  # Teenage Gaussian center

        # Step #1: Use dual annealing to find optimal starting parameters
        sol00 = dual_annealing(gf.growth_len_fitting,
                       boundsv,
                       args=(self.dat_time, self.dat_n),
                       maxiter=2000,
                       minimizer_kwargs=None,
                       initial_temp=5230.0,
                       restart_temp_ratio=2e-05,
                       visit=2.62,
                       accept=-5.0,
                       no_local_search=False,
                       callback=None,
                       x0=params,
                       local_search_options=None)

        # Try a two-phase search where the solution of the growth potential fitting is used as the
        # initial conditions of a second fit to the real data:

        sol0 = minimize(gf.growth_len_fitting,
                        sol00.x,
                        args=(self.dat_time, self.dat_n),
                        method='trust-constr',
                        bounds=boundsv,
                        tol=1.0e-9
                        )

        fit_params = sol0.x

        self.params_bytri = fit_params

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

        self.rmse =  np.sqrt(np.mean((self.fit_o_bytri - self.dat_o) ** 2))

        # Calculate parameter covariance matrix using the Jacobian:
        jac = gf.growth_jacobian_f2(fit_params, self.dat_time, self.dat_Lmax)

        vari = np.sum((self.fit_o_bytri - self.dat_o)**2)/(len(self.fit_o_bytri) - len(fit_params))
        param_covariance = np.linalg.pinv(jac.T.dot(jac))*vari

        self.cov = param_covariance # save the covariance matrix
        self.df = (len(self.fit_o_bytri) - len(fit_params)) # save degrees of freedom
        self.N = len(self.fit_o_bytri)  # save total samples
        self.n = len(fit_params) # save fit params

        self.param_error_bytri = np.sqrt(np.diag(param_covariance))*self.rmse # Stand error of mean parameters
        self.param_sd_bytri = np.sqrt(np.diag(param_covariance)) # Stand deviation of parameters
        sum_resid = np.sum((self.fit_o_bytri - self.dat_o) ** 2)
        self.mse_bytri = sum_resid
        self.rmse_bytri = self.rmse

        # Calculate the fboost and its components:
        self.fboost_bytri, self.Fa, self.Fb, self.Fc = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3], fit_params[4],
                                       fit_params[5], fit_params[6], fit_params[7])

        # Calculate the error on the fboost parameters:
        self.sd_fboost, self.sd_Fa, self.sd_Fb, self.sd_Fc = gf.error_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3], fit_params[4],
                                       fit_params[5], fit_params[6], fit_params[7], param_covariance)

        # predict initial length from lboost and final length:
        self.Lo_bytri = self.dat_Lmax / np.exp(self.fboost_bytri)
        self.sd_Lo = self.Lo_bytri*self.sd_fboost

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

    def fit_whole_triphasic(self):
        '''
        '''

        # Minimize method: ----------------------------------------
        params = [0.42, 0.5, 0.05, 1.0, 8.0, 0.05, 1.0, 13.5, 160]

        # Bounds on parameter values
        minb = 0.0
        maxb = 100.0
        boundsv = [(minb, maxb) for i in range(len(params))]

        ## Constraints:
        boundsv[0] = (0.0, 100.0)  # Exponential phase max
        boundsv[1] = (0.0, 100.0)  # Exponential phase decay
        boundsv[2] = (0.0, 10.0)  # Childhood Gaussian peak height
        boundsv[3] = (0.1, 20.0)  # Childhood Gaussian width
        boundsv[4] = (0.0, 10.0)  # Childhood Gaussian peak center
        boundsv[5] = (0.0, 10.0)  # Teenage Gaussian peak height
        boundsv[6] = (0.1, 10.0)  # Teenage Gaussian peak width
        boundsv[7] = (6.0, 20.0)  # Teenage Gaussian center
        boundsv[8] = (0.0, 300.0)

        # Step #1: Use dual annealing to find optimal starting parameters
        sol00 = dual_annealing(gf.growth_len_fitting_complete,
                       boundsv,
                       args=(self.dat_time, self.dat_o),
                       maxiter=1000,
                       minimizer_kwargs=None,
                       initial_temp=5230.0,
                       restart_temp_ratio=2e-05,
                       visit=2.62,
                       accept=-5.0,
                       no_local_search=False,
                       callback=None,
                       x0=params,
                       local_search_options=None)

        # Try a two-phase search where the solution of the growth potential fitting is used as the
        # initial conditions of a second fit to the real data:

        sol0 = minimize(gf.growth_len_fitting_complete,
                        sol00.x,
                        args=(self.dat_time, self.dat_o),
                        method='trust-constr',
                        bounds=boundsv,
                        tol=1.0e-9
                        )

        fit_params = sol0.x

        self.params_bytri2 = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:

        self.fit_o_bytri2 = gf.growth_len(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7],
                                   fit_params[8]
                                   )

        self.rmse_bytri2 =  np.sqrt(np.mean((self.fit_o_bytri - self.dat_o) ** 2))

        # Calculate parameter covariance matrix using the Jacobian:
        # jac = gf.growth_jacobian_f2(fit_params, self.dat_time, self.dat_Lmax)

        # vari = np.sum((self.fit_o_bytri - self.dat_o)**2)/(len(self.fit_o_bytri) - len(fit_params))
        # param_covariance = np.linalg.pinv(jac.T.dot(jac))*vari

        # self.cov = param_covariance # save the covariance matrix
        self.df_bytri2 = (len(self.fit_o_bytri) - len(fit_params)) # save degrees of freedom
        self.N_bytri2 = len(self.fit_o_bytri)  # save total samples
        self.n_bytri2 = len(fit_params) # save fit params

        # self.param_error_bytri = np.sqrt(np.diag(param_covariance))*self.rmse # Stand error of mean parameters
        # self.param_sd_bytri = np.sqrt(np.diag(param_covariance)) # Stand deviation of parameters
        sum_resid = np.sum((self.fit_o_bytri - self.dat_o) ** 2)
        self.mse_bytri2 = sum_resid
        self.rmse_bytri2 = self.rmse

        # Calculate the fboost and its components:
        self.fboost_bytri2, self.Fa2, self.Fb2, self.Fc2 = gf.growth_fboost(fit_params[0], fit_params[1],
                                       fit_params[2], fit_params[3], fit_params[4],
                                       fit_params[5], fit_params[6], fit_params[7])

        # Calculate the error on the fboost parameters:
        # self.sd_fboost, self.sd_Fa, self.sd_Fb, self.sd_Fc = gf.error_fboost(fit_params[0], fit_params[1],
        #                                fit_params[2], fit_params[3], fit_params[4],
        #                                fit_params[5], fit_params[6], fit_params[7], param_covariance)

        # predict initial length from lboost and final length:
        self.Lo_bytri2 = fit_params[8] / np.exp(self.fboost_bytri2)
        # self.sd_Lo2 = self.Lo_bytri2*self.sd_fboost2

        # Now that we have the parameters, compute components of the triphasic model's growth potential:
        self.phi_gomp12, self.phi_gauss12, self.phi_gauss22 = gf.growth_potential_components(self.dat_time,
                                                                                           fit_params[0],
                                                                                           fit_params[1],
                                                                                           fit_params[2],
                                                                                           fit_params[3],
                                                                                           fit_params[4],
                                                                                           fit_params[5],
                                                                                           fit_params[6],
                                                                                           fit_params[7],
                                                                                           )

        self.fit_vel_bytri2 = gf.growth_vel(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   fit_params[2],
                                   fit_params[3],
                                   fit_params[4],
                                   fit_params[5],
                                   fit_params[6],
                                   fit_params[7],
                                   fit_params[8])

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

        # predict initial length from lboost and final length:
        self.Lo_bygomp = self.dat_Lmax / np.exp(self.fboost_bygomp)

        # Predict growth velocity:
        self.fit_vel_bygomp = gomp.growth_vel(self.dat_time,
                                   fit_params[0],
                                   fit_params[1],
                                   self.dat_Lmax)

    def fit_pb1(self):
        '''
        '''

        # Minimize method: ----------------------------------------
        params = np.asarray([174.6, 162.9, 0.1124, 1.2397, 14.6])

        # Bounds on parameter values
        minb = 0.0
        maxb = 300.0
        boundsv = [(minb, maxb) for i in range(len(params))]

        boundsv[2] = (0, 100.0)
        boundsv[3] = (0, 100.0)
        boundsv[4] = (0, 100.0)

        # Step #1: Use dual annealing to find optimal starting parameters
        sol00 = dual_annealing(pb1.growth_len_fitting,
                               boundsv,
                               args=(self.dat_time, self.dat_o),
                               maxiter=1000,
                               minimizer_kwargs=None,
                               initial_temp=5230.0,
                               restart_temp_ratio=2e-05,
                               visit=2.62,
                               accept=-5.0,
                               no_local_search=False,
                               callback=None,
                               x0=params,
                               local_search_options=None)

        # Try a two-phase search where the solution of the growth potential fitting is used as the
        # initial conditions of a second fit to the real data:

        sol0 = minimize(pb1.growth_len_fitting,
                        sol00.x,
                        args=(self.dat_time, self.dat_o),
                        method='trust-constr',
                        bounds=boundsv,
                        tol=1.0e-9
                        )

        fit_params = sol0.x

        self.params_bypb1 = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:

        self.fit_o_bypb1 = pb1.growth_len(self.dat_time,
                                         fit_params[0],
                                         fit_params[1],
                                         fit_params[2],
                                         fit_params[3],
                                         fit_params[4]
                                         )

        rmse = np.sqrt(np.mean((self.fit_o_bypb1 - self.dat_o) ** 2))

        # self.cov = param_covariance  # save the covariance matrix
        self.df_pb1 = (len(self.fit_o_bypb1) - len(fit_params))  # save degrees of freedom
        self.N_pb1 = len(self.fit_o_bypb1)  # save total samples
        self.n_pb1 = len(fit_params)  # save fit params

        # self.param_error_bytri = np.sqrt(np.diag(param_covariance)) * self.rmse  # Stand error of mean parameters
        # self.param_sd_bytri = np.sqrt(np.diag(param_covariance))  # Stand deviation of parameters
        sum_resid = np.sum((self.fit_o_bypb1 - self.dat_o) ** 2)
        self.mse_bypb1 = sum_resid
        self.rmse_bypb1 = rmse


        self.fit_vel_bypb1 = pb1.growth_vel(self.dat_time,
                                           fit_params[0],
                                           fit_params[1],
                                           fit_params[2],
                                           fit_params[3],
                                           fit_params[4]
                                            )

    def fit_pb3(self):
        '''
        '''

        # Minimize method: ----------------------------------------
        params = np.asarray([174.6, 162.9, 0.088, 0.23, 1.37, 14.6])

        # Bounds on parameter values
        minb = 0.0
        maxb = 300.0
        boundsv = [(minb, maxb) for i in range(len(params))]

        boundsv[2] = (0, 100.0)
        boundsv[3] = (0, 100.0)
        boundsv[4] = (0, 100.0)
        boundsv[4] = (0, 100.0)

        # Step #1: Use dual annealing to find optimal starting parameters
        sol00 = dual_annealing(pb3.growth_len_fitting,
                               boundsv,
                               args=(self.dat_time, self.dat_o),
                               maxiter=1000,
                               minimizer_kwargs=None,
                               initial_temp=5230.0,
                               restart_temp_ratio=2e-05,
                               visit=2.62,
                               accept=-5.0,
                               no_local_search=False,
                               callback=None,
                               x0=params,
                               local_search_options=None)

        # Try a two-phase search where the solution of the growth potential fitting is used as the
        # initial conditions of a second fit to the real data:

        sol0 = minimize(pb3.growth_len_fitting,
                        sol00.x,
                        args=(self.dat_time, self.dat_o),
                        method='trust-constr',
                        bounds=boundsv,
                        tol=1.0e-9
                        )

        fit_params = sol0.x

        self.params_bypb3 = fit_params

        # Compute the fit for use of these parameters on the original growth curve data:

        self.fit_o_bypb3 = pb3.growth_len(self.dat_time,
                                         fit_params[0],
                                         fit_params[1],
                                         fit_params[2],
                                         fit_params[3],
                                         fit_params[4],
                                         fit_params[5]
                                         )

        rmse = np.sqrt(np.mean((self.fit_o_bypb3 - self.dat_o) ** 2))

        # self.cov = param_covariance  # save the covariance matrix
        self.df_pb3 = (len(self.fit_o_bypb3) - len(fit_params))  # save degrees of freedom
        self.N_pb3 = len(self.fit_o_bypb3)  # save total samples
        self.n_pb3 = len(fit_params)  # save fit params

        # self.param_error_bytri = np.sqrt(np.diag(param_covariance)) * self.rmse  # Stand error of mean parameters
        # self.param_sd_bytri = np.sqrt(np.diag(param_covariance))  # Stand deviation of parameters
        sum_resid = np.sum((self.fit_o_bypb3 - self.dat_o) ** 2)
        self.mse_bypb3 = sum_resid
        self.rmse_bypb3 = rmse


        self.fit_vel_bypb3 = pb3.growth_vel(self.dat_time,
                                           fit_params[0],
                                           fit_params[1],
                                           fit_params[2],
                                           fit_params[3],
                                           fit_params[4],
                                            fit_params[5]
                                            )

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

