'''
Arbitrage logic
'''

import scipy
import numpy as np
from scipy.stats import norm
from scipy import optimize

EPSILON = 1e-3

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
        # price_sell_risky = gamma*K*norm.pdf(norm.ppf(1 - R1) - sigma*np.sqrt(tau))*quantilePrime(1 - R1)
        price_sell_risky = Pool.getMarginalPriceSwapRiskyIn(0)
        #Marginal price of buying epsilon risky
        price_buy_risky = Pool.getMarginalPriceSwapRisklessIn(0)
        
        print(f"sell price {price_sell_risky}")
        print(f"buy price {price_buy_risky}")
        print(f"market price {market_price}")

        #Market price
        m = market_price

        #If the price of selling epsilon of the risky asset is above the market price, we buy the optimal amount of the risky asset on the market and immediately sell it on the CFMM = **swap amount in risky**.
        if price_sell_risky > m + 1e-8:
            #Solve for the optimal amount in
            def func(amount_in):
                return Pool.getMarginalPriceSwapRiskyIn(amount_in) - m
            # print("gamma = ", gamma, "risky = ", R1, "riskless = ", R2, "K = ", K, "invariant = ", k, "sigma = ", sigma, "tau = ", tau, "m = ", m, "price risky > m")
            optimal_trade = scipy.optimize.bisect(func, 0, (1 - R1 - EPSILON))
            # print("result = ", func(optimal_trade))
            print("Optimal trade: ", optimal_trade, " ETH in")
            amount_out, _ = Pool.virtualSwapAmountInRisky(optimal_trade)
            #The amount of the riskless asset we get after making the swap must be higher than the value in the riskless asset at which we obtained the amount in on the market
            profit = amount_out - optimal_trade*m
            print(f"sell profit {profit} \n")
            if profit > 0:
                _, _ = Pool.swapAmountInRisky(optimal_trade)
            # print("profit = ", profit)
        
        #If the price of buying epsilon of the risky asset is below the market price, we buy the optimal amount of the risky asset in the CFMM and immediately sell it on the market = **swap amount in riskless** in the CFMM.
        elif price_buy_risky < m - 1e-8:
            def func(amount_in):
                return m - Pool.getMarginalPriceSwapRisklessIn(amount_in)
            # print("gamma = ", gamma, "risky = ", R1, "riskless = ", R2, "K = ", K, "invariant = ", k, "sigma = ", sigma, "tau = ", tau, "m = ", m, "price risky < m")
            optimal_trade = scipy.optimize.bisect(func, 0, (K - R2 - EPSILON))
            # print("result = ", func(optimal_trade))
            print("Optimal trade: ", optimal_trade, " USD in")
            amount_out, _ = Pool.virtualSwapAmountInRiskless(optimal_trade)
            #The amount of risky asset we get out times the market price must result in an amount of riskless asset higher than what we initially put in the CFMM 
            profit = amount_out*m - optimal_trade
            print(f"buy profit {profit} \n")
            if profit > 0:
                _, _ = Pool.swapAmountInRiskless(optimal_trade)
            # print("profit = ", profit)