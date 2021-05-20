'''
Arbitrage logic
'''

import scipy
import numpy as np
from scipy.stats import norm
from scipy import optimize

def quantilePrime(x):
    return norm.pdf(norm.ppf(x))**-1

class Arbitrager():
    '''
    A class to represent an arbitrager who will look at a reference price of the risky asset, denominated in the riskless asset, the price in an AMM pool, and arbitrage the difference. Ideal arbitrager with infinite portfolio of either assets.
    '''

    def arbitrageExactly(self, market_price, Pool):
        '''
        Arbitrage the difference *exactly* at the time of the call to the function. Uses results from the following paper: https://arxiv.org/abs/2012.08040

        Params:

        reference_price (float):
            the reference price of the risky asset, denominated in the riskless asset
        Pool (AMM object):
            an AMM object, for example a CoveredCallAMM class, with some current state and reserves
        '''
        gamma = 1 - Pool.fee
        R1 = Pool.reserves_risky
        R2 = Pool.reserves_riskless
        K = Pool.K
        k = Pool.invariant
        sigma = Pool.sigma
        tau = Pool.tau

        #Marginal price of selling epsilon risky
        price_sell_risky = gamma*K*norm.pdf(norm.ppf(1 - R1) - sigma*np.sqrt(tau))*quantilePrime(1 - R1)
        #Marginal price of buying epsilon risky
        price_buy_risky = 1/(gamma * norm.pdf(norm.ppf((R2 -k)/K) + sigma*np.sqrt(tau))*quantilePrime((R2 - k)/K)*(1/K))

        #Market price
        m = market_price

        #If the price of selling epsilon of the risky asset is above the market price, we buy the optimal amount of the risky asset on the market and immediately sell it on the CFMM = **swap amount in risky**.
        if price_sell_risky > m:
            #Solve for the optimal amount in
            def func(amount_in):
                return gamma*K*norm.pdf(norm.ppf(1 - R1 - gamma*amount_in) - sigma*np.sqrt(tau))*quantilePrime(1 - R1 - gamma*amount_in) - m
            sol = scipy.optimize.root(func, Pool.reserves_risky)
            optimal_trade = sol.x[0]
            _,_ = Pool.swapAmountInRisky(optimal_trade)
        
        #If the price of buying epsilon of the risky asset is below the market price, we buy the optimal amount of the risky asset in the CFMM and immediately sell it on the market = **swap amount in riskless** in the CFMM.
        elif price_buy_risky < m:
            def func(amount_in):
                return 1/(gamma * norm.pdf(norm.ppf((R2 + gamma*amount_in - k)/K) + sigma*np.sqrt(tau))*quantilePrime((R2 + gamma*amount_in - k)/K)*(1/K)) - m
            sol = scipy.optimize.root(func, Pool.reserves_riskless)
            optimal_trade = sol.x[0]
            _,_ = Pool.swapAmountInRiskless(optimal_trade)