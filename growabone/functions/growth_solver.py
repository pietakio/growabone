#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.

'''
**Growth functions for modeling human growth characteristics**

This module uses a symbolic math solver (Sympy) to solve analytic equations to provide loss functions,
Jacobians and Hessians derived from the analytic forms for non-linear least squares solving.

'''
from beartype import beartype
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from enum import Enum

class ModelType(Enum):
    '''

    '''
    full_TGM = 'full TGM'
    norm_TGM = 'norm TGM'
    pb1 = 'PB1'
    karleberg = 'Karleberg'
    logistic = 'logistic'
    gompertz = 'gompertz'
    trilog = 'tri-logistic'
    explogistic = 'exp-logistic'
    lomax = 'lomax'


@beartype
class GrowthSolver(object):
    '''

    '''

    def __init__(self, model_type: ModelType=ModelType.full_TGM):
        '''

        :return:
        '''

        # Set the system up for the desired equations to model the growth data:

        if model_type is ModelType.full_TGM:
            self._solve_analytical_full_TGM()

        elif model_type is ModelType.norm_TGM:
            self._solve_analytical_norm_TGM()

        elif model_type is ModelType.pb1:
            self._solve_analytical_PB1()

        elif model_type is ModelType.trilog:
            self._solve_analytical_trilogistic()

        elif model_type is ModelType.explogistic:
            self._solve_analytical_explogistic()

        elif model_type is ModelType.karleberg:
            self._solve_analytical_karleberg()

        elif model_type is ModelType.logistic:
            self._solve_analytical_logistic()

        elif model_type is ModelType.gompertz:
            self._solve_analytical_gompertz()

        elif model_type is ModelType.lomax:
            self._solve_analytical_lomax()

        else:
            raise Exception("Model type not supported.")

        self.model_type = model_type

    def _solve_analytical_full_TGM(self):
        '''

        :return:
        '''

        # Begin by defining some symbols to use in analytic math expressions:
        a, b, c, A, B, C, t_a, t_b, t_c, t_f, L_o, L_max, t, K = sp.symbols(
            'alpha, beta, gamma, A, B, C, t_a, t_b, t_c, t_f, L_o, L_max, t, K', real=True, positive=True)

        # Triphasic growth equation using unnormalized Gaussians (good for clear understanding of time-scaling effects,
        # but leads to a more complicated growth equation):
        f = A * sp.exp(-a * (t)) + B * sp.exp(-(1 / b) * (t - t_b) ** 2) + C * sp.exp(-(1 / c) * (t - t_c) ** 2)

        # Compute 'g', which is the logarithm of the growth curve:
        g = sp.integrate(f, t)

        # Integration of the growth driving potential, with added constant of integration 'K':
        g += K

        # for unscaled Gaussians, apply some simplifications:
        BB = ((sp.sqrt(sp.pi * b) * B) / 2)
        CC = ((sp.sqrt(sp.pi * c) * C) / 2)
        g_Lm = -(A / a) * sp.exp(-a * t) + BB * sp.erf((t - t_b) / b) + CC * sp.erf((t - t_c) / c) + sp.log(
            L_max) - BB - CC
        # g_Lo = -(A / a) * sp.exp(-a * t) + BB * sp.erf((t - t_b) / b) + CC * sp.erf((t - t_c) / c) + sp.log(L_o) + (
        #             A / a) + BB + CC

        # Symbolic expressions for the growth curve with time will be the expression for g_Lo or g_Lm raised to the
        # exponential:
        self._Lt_m = sp.exp(g_Lm) # in terms of final size Lmax

        self.params_s = [A, a, B, b, t_b, C, c, t_c, L_max] # symbolic parameters vector

        self._t = t

        self._set_numerical_components() # Finish up this model with common assignment & 'numerification' of functions

        # Set of suitable initial parameters:
        self.init_params_o = [0.42, 0.5, 0.05, 1.0, 8.0, 0.05, 1.0, 13.5, 200.0]

    def _solve_analytical_norm_TGM(self):
        '''

        :return:
        '''

        # Begin by defining some symbols to use in analytic math expressions:
        a, b, c, A, B, C, t_a, t_b, t_c, t_f, L_o, L_max, t, K = sp.symbols(
            'alpha, beta, gamma, A, B, C, t_a, t_b, t_c, t_f, L_o, L_max, t, K', real=True, positive=True)

        # Triphasic growth equation using unnormalized Gaussians (good for clear understanding of time-scaling effects,
        # but leads to a more complicated growth equation):
        f = A * sp.exp(-a * (t)) + B * sp.exp(-(1 / b) * (t - t_b) ** 2) + C * sp.exp(-(1 / c) * (t - t_c) ** 2)

        # Compute 'g', which is the logarithm of the growth curve:
        g = sp.integrate(f, t)

        # Integration of the growth driving potential, with added constant of integration 'K':
        g += K

        # for unscaled Gaussians, apply some simplifications:
        BB = ((sp.sqrt(sp.pi * b) * B) / 2)
        CC = ((sp.sqrt(sp.pi * c) * C) / 2)
        g_Lm = -(A / a) * sp.exp(-a * t) + BB * sp.erf((t - t_b) / b) + CC * sp.erf((t - t_c) / c) + sp.log(
            L_max) - BB - CC
        # g_Lo = -(A / a) * sp.exp(-a * t) + BB * sp.erf((t - t_b) / b) + CC * sp.erf((t - t_c) / c) + sp.log(L_o) + (
        #             A / a) + BB + CC

        # Symbolic expressions for the growth curve with time will be the expression for g_Lo or g_Lm raised to the
        # exponential:
        self._Lt_m = (sp.exp(g_Lm) / L_max).simplify() # in terms of final size Lmax, normalized to final height of 1.0

        self.params_s = [A, a, B, b, t_b, C, c, t_c] # symbolic parameters vector for this model

        self._t = t # save the symbolic time

        self._set_numerical_components() # Finish up this model with common assignment & 'numerification' of functions

        # Set of suitable initial parameters:
        self.init_params_o = [0.42, 0.5, 0.05, 1.0, 8.0, 0.05, 1.0, 13.5]

    def _solve_analytical_PB1(self):
        '''
        Solve the Preece-Bains model #1
        :return:
        '''
        h_1, h_theta, s_o, s_1, theta, t = sp.symbols('h_1, h_theta, s_o, s_1, theta, t',
                                                            real=True, positive=True)
        h_m1 = h_1 - ((2 * (h_1 - h_theta)) / (
                    sp.exp(s_o * (t - theta)) + sp.exp(s_1 * (t - theta))))  # Growth curve

        self._Lt_m = h_m1 # Symbolic growth curve
        self._t = t # symbolic time
        self.params_s = [h_1, h_theta, s_o, s_1, theta] # Symbolic parameters

        self._set_numerical_components() # Finish up this model with common assignments & 'numerification' of functions

        self.init_params_o = [172.0, 159.8, 0.115, 1.06, 12.81]

    def _solve_analytical_karleberg(self):
        '''
        Solve the Karleberg growth model.
        :return:
        '''
        ai, bi, ci, ac, bc, cc, ap, bp, tv, t = sp.symbols('a_i, b_i, c_i, a_c, b_c, c_c, a_p, b_p, t_v, t')

        g_inf = ai + bi*(1 - sp.exp(-ci*t)) # infant growth curve component
        g_chi = ac + bc*t + cc*t**2 # childhood growth component
        g_pub = ap/(1 + sp.exp(-bp*(t-tv))) # puberty growth component

        self._Lt_m = g_inf + g_chi + g_pub # symbolic growth curve
        self._t = t # symbolic time
        self.params_s = [ai, bi, ci, ac, bc, cc, ap, bp, tv] # Symbolic parameters

        self._set_numerical_components() # Finish up this model with common assignments & 'numerification' of functions

        self.init_params_o = [25.1, 26.1, -0.04, 44.17, 9.76, -0.20, 11.88, 2.27, 14.60]

    def _solve_analytical_logistic(self):
        '''

        :return:
        '''
        alpha, ta, Lmax, t = sp.symbols('alpha, ta, Lmax, t')
        Lt = Lmax / (1 + sp.exp(-alpha * (t - ta)))

        self._Lt_m = Lt # growth function

        self.params_s = [alpha, ta, Lmax] # symbolic parameters vector

        self._t = t

        self._set_numerical_components() # Finish up this model with common assignment & 'numerification' of functions

    def _solve_analytical_gompertz(self):
        '''

        :return:
        '''
        A, alpha, Lmax, t = sp.symbols('alpha, ta, Lmax, t')

        Lt = Lmax * sp.exp(-(A / alpha) * sp.exp(-alpha * (t)))

        self._Lt_m = Lt  # growth function

        self.params_s = [A, alpha, Lmax]  # symbolic parameters vector

        self._t = t

        self._set_numerical_components()  # Finish up this model with common assignment & 'numerification' of functions

    def _solve_analytical_lomax(self):
        '''

        :return:
        '''
        A, alpha, t = sp.symbols('A, alpha, t')

        # Lt = A * (1 - (1 + (t/lamb))**(-alpha))
        Lt = A * (1 - sp.exp(-alpha*t))

        self._Lt_m = Lt  # growth function

        self.params_s = [A, alpha]  # symbolic parameters vector

        self._t = t

        self._set_numerical_components()  # Finish up this model with common assignment & 'numerification' of functions

    def _solve_analytical_trilogistic(self):
        '''

        :return:
        '''
        alpha1, ta1, Lmax1, t = sp.symbols('alpha1, ta1, Lmax1, t')
        alpha2, ta2, Lmax2 = sp.symbols('alpha2, ta2, Lmax2')
        alpha3, ta3, Lmax3 = sp.symbols('alpha3, ta3, Lmax3')

        Lt1 = Lmax1 / (1 + sp.exp(-alpha1 * (t - ta1)))
        Lt2 = Lmax2 / (1 + sp.exp(-alpha2 * (t - ta2)))
        Lt3 = Lmax3 / (1 + sp.exp(-alpha3 * (t - ta3)))

        self._Lt_m = Lt1 + Lt2 + Lt3 # growth function

        self.params_s = [alpha1, ta1, Lmax1, alpha2, ta2, Lmax2, alpha3, ta3, Lmax3] # symbolic parameters vector

        self._t = t

        self._set_numerical_components() # Finish up this model with common assignment & 'numerification' of functions

        self.init_params_o = [1.24, 0.25, 90.48, 0.48, 8.17, 74.85, 0.32, 13.9, 17.2]

    def _solve_analytical_explogistic(self):
        '''

        :return:
        '''

        alpha1, Lmax1, t = sp.symbols('alpha1, Lmax1, t')
        alpha2, ta2, Lmax2 = sp.symbols('alpha2, ta2, Lmax2')
        alpha3, ta3, Lmax3 = sp.symbols('alpha3, ta3, Lmax3')

        Lt1 = Lmax1*(1 - sp.exp(-alpha1*1))
        Lt2 = Lmax2 / (1 + sp.exp(-alpha2 * (t - ta2)))
        Lt3 = Lmax3 / (1 + sp.exp(-alpha3 * (t - ta3)))

        self._Lt_m = Lt1 + Lt2 + Lt3 # growth function

        self.params_s = [alpha1, Lmax1, alpha2, ta2, Lmax2, alpha3, ta3, Lmax3] # symbolic parameters vector

        self._t = t

        self._set_numerical_components() # Finish up this model with common assignment & 'numerification' of functions

        self.init_params_o = [1.24, 0.25, 90.48, 0.48, 8.17, 74.85]

    def _set_numerical_components(self):
        '''

        :return:
        '''
        # Derive an optimization function for non-linear least-squares fitting:
        y_dat = sp.symbols('y_dat') # data curve
        Fopti = (self._Lt_m - y_dat) ** 2 # optimization function, symbolic version

        Fresids = self._Lt_m - y_dat # residuals

        jac_elements = [sp.diff(Fopti, pi, 1) for pi in self.params_s] # Symbolic Jacobian
        hess_elements = [[Fopti.diff(pj).diff(pi) for pj in self.params_s] for pi in self.params_s] # Symbolic Hessian

        self._jac_funko = sp.lambdify((self.params_s, self._t, y_dat), jac_elements)
        self._hess_funko = sp.lambdify((self.params_s, self._t, y_dat), hess_elements)

        # Numerical optimization function:
        self._opti_funko = sp.lambdify((self.params_s, self._t, y_dat), Fopti)

        self.growth_funk = sp.lambdify((self.params_s, self._t, y_dat), self._Lt_m)

        vt_m = sp.diff(self._Lt_m, self._t, 1)
        self.vel_funk = sp.lambdify((self.params_s, self._t, y_dat), vt_m)

        self.residuals = sp.lambdify((self.params_s, self._t, y_dat), Fresids)

    def jac_funk(self, para_v, ti, ydati):
        '''
        Numerical function to calculate the Jacobian matrix.
        :param para_v:
        :param ti:
        :param ydati:
        :return:
        '''

        jac_arr = np.sum(self._jac_funko(para_v, ti, ydati), axis=1)

        return jac_arr

    def hess_funk(self, para_v, ti, y_dati):
        '''
        Numerical function to calculate the Hessian matrix.
        :param para_v:
        :param ti:
        :param y_dati:
        :return:
        '''
        hess_arr = np.sum(self._hess_funko(para_v, ti, y_dati), axis=2)
        return hess_arr

    def opti_funk(self, para_v, ti, y_dati):
        '''
        Numerical function to calculate the optimization function for the system.
        :param para_v:
        :param ti:
        :param y_dati:
        :return:
        '''
        opti_val = np.sum(self._opti_funko(para_v, ti, y_dati))

        return opti_val

    def solve_system(self,
                     tt,
                     dat_array,
                     init_params,
                     tol: float=1.0e-9,
                     method: str='trust-krylov',
                     verbose: bool=True):
        '''

        :return:
        '''

        # Storage arrays
        fit_params_M = []
        fit_params_err_M = []
        fit_height_M = []
        fit_vel_M = []
        fit_rmse_M = []
        fit_resids_M = []
        vel_array = []

        del_t = np.gradient(tt, edge_order=2) # time differential

        for i, dat in enumerate(dat_array.T):


            if self.model_type is not ModelType.karleberg:

                sol0 = minimize(self.opti_funk,
                                init_params,
                                args=(tt, dat),
                                method=method,
                                jac=self.jac_funk,
                                hess=self.hess_funk,
                                tol=tol
                                )

                Hf = self.hess_funk(sol0.x, tt, dat)


            else:
                sol0 = minimize(self.opti_funk,
                                init_params,
                                args=(tt, dat),
                                method='CG',  # 3.7374
                                jac=self.jac_funk,
                                tol=tol
                                )

                Jf = self.jac_funk(sol0.x, tt, dat)
                Hf = np.outer(Jf, Jf) # Estimate for the hessian

            resids = self.residuals(sol0.x, tt, dat)
            var_resids = np.var(resids)
            param_err = np.sqrt(np.abs(np.diag(np.linalg.pinv(Hf)) * var_resids))

            rmse = np.sqrt(np.mean(resids ** 2))

            fit_params_M.append(sol0.x)
            fit_params_err_M.append(param_err)
            fit_height_M.append(self.growth_funk(sol0.x, tt, dat))
            fit_vel_M.append(self.vel_funk(sol0.x, tt, dat))
            fit_rmse_M.append(rmse)
            fit_resids_M.append(resids)

            # take the velocity of the data series:
            vel = np.gradient(dat, edge_order=2) / del_t  # Calculate the growth velocity of the raw data
            vel_array.append(vel)

            if verbose:
                print(f'Record {i} rmse: {rmse}, opti val: {sol0.fun}')

        self.fit_params_M = np.asarray(fit_params_M).T
        self.fit_params_err_M = np.asarray(fit_params_err_M).T
        self.fit_height_M = np.asarray(fit_height_M).T
        self.fit_vel_M = np.asarray(fit_vel_M).T
        self.fit_rmse_M = np.asarray(fit_rmse_M).T
        self.fit_resids_M = np.asarray(fit_resids_M).T
        self.dat_time = tt # save the original time vector
        self.dat_array = dat_array # save the original set of data series
        self.vel_array = np.asarray(vel_array).T # save the velocity of original data series

