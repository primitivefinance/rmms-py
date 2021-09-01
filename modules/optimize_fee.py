'''
Contains a set of functions used to find the optimal fee for a given set of parameters.
'''

import gc
import numpy as np
import scipy
from scipy.optimize import minimize_scalar

from joblib import Parallel, delayed

from modules.utils import generateGBM
from modules.simulate import simulate
from modules import cfmm

def returnErrors(fee, initial_tau, timestep_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters and a gbm, return the errors under 
    optimal arbitrage for that gbm
    '''
    np.random.seed()
    t, gbm = generateGBM(time_horizon, drift, volatility, initial_price, timestep_size)
    Pool = cfmm.CoveredCallAMM(0.5, strike, volatility, initial_tau, fee)
    _, _, mean_error, terminal_error = simulate(Pool, t, gbm)
    del Pool 
    del gbm 
    del t
    gc.collect()
    return mean_error, terminal_error

def findOptimalFee(initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters, return the fee that minimizes the maximum between the mean square
    error and the terminal square error (square of the error at the last step of the 
    simulation)
    '''

    def ErrorFromFee(fee): 
        '''
        Return the max of the average mse and average terminal square error from 100 
        simulations with different price actions given these parameters
        '''
        # DEBUGGING 
        # print("fee = ", fee)
        # results = []
        # for i in range(50): 
        #     print("STEP ", i)
        #     results.append(returnErrors(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price))

        results = Parallel(n_jobs=-1, verbose=0, backend='loky')(delayed(returnErrors)(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price) for i in range(50))
        average_error = np.mean([item[0] for item in results])
        average_terminal_error = np.mean([item[1] for item in results])
        del results
        gc.collect()
        # return max(average_error, average_terminal_error)
        return average_terminal_error

    #Look for the optimal fee with a tolerance of +/- 0.5% or 50 bps
    # sol = minimize_scalar(ErrorFromFee, bracket=(0.0001, 0.15), method='Golden', tol = 0.005)
    sol = scipy.optimize.fminbound(ErrorFromFee, 0.0001, 0.10, xtol = 0.0005)
    optimal_fee = sol
    return optimal_fee
