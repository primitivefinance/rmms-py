'''
Contains a set of functions used to find the optimal fee for a given set of parameters.
'''

import cfmm
import numpy as np
from scipy.optimize import minimize_scalar
from simulate import simulate
from time_series import generateGBM
from joblib import Parallel, delayed

def returnErrors(fee, initial_tau, timestep_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters and a gbm, return the errors under 
    optimal arbitrage for that gbm
    '''
    np.random.seed()
    t, gbm = generateGBM(time_horizon, drift, volatility, initial_price, timestep_size)
    Pool = cfmm.CoveredCallAMM(0.5, strike, volatility, initial_tau, fee)
    _, _, mean_error, terminal_error = simulate(Pool, t, gbm)
    return mean_error, terminal_error

def findOptimalFee(initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters, return the fee that minimizes the maximum between the mean square
    error and the terminal square error (square of the error at the last step of the 
    simulation)
    '''

    def maxErrorFromFee(fee): 
        '''
        Return the max of the average mse and average terminal square error from 100 
        simulations with different price actions given these parameters
        '''
        results = Parallel(n_jobs=-1, verbose=1, backend='multiprocessing')(delayed(returnErrors)(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price) for i in range(100))
        average_error = np.mean([item[0] for item in results])
        average_terminal_error = np.mean([item[1] for item in results])
        return max(average_error, average_terminal_error)

    sol = minimize_scalar(maxErrorFromFee, bounds=(0, 0.15), method='Brent')
    optimal_fee = sol.x
    return optimal_fee
