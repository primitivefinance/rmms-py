'''
Contains a set of functions used to find the optimal fee for a given set of parameters.
'''

import cfmm
import numpy as np
from scipy.optimize import minimize_scalar
from simulate import simulate
from time_series import generateGBM

def returnErrors(fee, initial_tau, timestep_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters and a gbm, return the errors under 
    optimal arbitrage for that gbm
    '''
    t, gbm = generateGBM(time_horizon, drift, volatility, initial_price, timestep_size)
    Pool = cfmm.CoveredCallAMM(0.5, strike, volatility, initial_tau, fee)
    _, _, _, mse, terminal_deviation = simulate(Pool, t, gbm)
    return mse, terminal_deviation

def findOptimalFee(initial_tau, timestep_size, time_horizon, volatility, drift, strike, initial_price):
    '''
    Given some parameters, return the fee that minimizes the maximum between the mean square
    error and the terminal square error (square of the error at the last step of the 
    simulation)
    '''
    sigma = volatility
    mu = drift
    T = time_horizon
    K = strike
    dtau = timestep_size
    tau0 = initial_tau
    p0 = initial_price

    def maxErrorFromFee(fee): 
        '''
        Return the max of the average mse and average terminal square error from 100 
        simulations with different price actions given these parameters
        '''
        mse_array = []
        square_terminal_error_array = []
        for i in range(100):
            mse, square_terminal_error = returnErrors(fee, tau0, dtau, T, sigma, mu, K, p0)
            mse_array.append(mse)
            square_terminal_error_array.append(square_terminal_error)
        average_mse = np.mean(mse_array)
        average_square_terminal_error = np.mean(square_terminal_error_array)
        return max(average_mse, average_square_terminal_error)

    sol = minimize_scalar(maxErrorFromFee, bounds=(0, 0.15), method='Brent')
    optimal_fee = sol.x
    return optimal_fee