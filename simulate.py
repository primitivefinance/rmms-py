'''
Functions used to run an actual simulation
'''

import numpy as np 

from arb import arbitrageExactly
from utils import getRiskyReservesGivenSpotPrice, getRisklessGivenRisky


def simulate(Pool, t, gbm):
    '''
    A function which, given a pool, a time array, and a geometric brownian 
    motion, for that time array, returns the results of a simulation of that pool 
    under optimal arbitrage.
    '''
    # Array to store the theoretical value of LP shares in the case of a pool with zero fees
    theoretical_lp_value_array = []
    # Effective value of LP shares with fees
    effective_lp_value_array = []
    initial_tau = Pool.initial_tau
    for i in range(len(gbm)): 
        theoretical_tau = initial_tau - t[i]/365
        Pool.tau = initial_tau - t[i]/365
        Pool.invariant = Pool.reserves_riskless - Pool.getRisklessGivenRiskyNoInvariant(Pool.reserves_risky)
        arbitrageExactly(gbm[i], Pool)
        theoretical_reserves_risky = getRiskyReservesGivenSpotPrice(gbm[i], Pool.K, Pool.sigma, theoretical_tau)
        theoretical_reserves_riskless = getRisklessGivenRisky(theoretical_reserves_risky, Pool.K, Pool.sigma, theoretical_tau)
        theoretical_lp_value = theoretical_reserves_risky*gbm[i] + theoretical_reserves_riskless
        theoretical_lp_value_array.append(theoretical_lp_value)
        effective_lp_value_array.append(Pool.reserves_risky*gbm[i] + Pool.reserves_riskless)
    theoretical_lp_value_array = np.array(theoretical_lp_value_array)
    effective_lp_value_array = np.array(effective_lp_value_array)
    mse = np.square(np.subtract(theoretical_lp_value_array, effective_lp_value_array)/theoretical_lp_value_array).mean()
    terminal_square_error = ((theoretical_lp_value_array[-1] - effective_lp_value_array[-1])/(theoretical_lp_value_array[-1]))**2
    return theoretical_lp_value_array, effective_lp_value_array, mse, terminal_square_error
