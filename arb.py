'''
Arbitrage logic
'''

import scipy
import numpy as np

class Arbitrager():
    '''
    A class to represent an arbitrager who will look at a reference price of the risky asset, denominated in the riskless asset, the price in an AMM pool, and arbitrage the difference. Ideal arbitrager with infinite portfolio of either assets.
    '''

    def arbitrageExactly(self, reference_price, Pool):
        '''
        Arbitrage the difference *exactly* at the time of the call to the function. Only valid for the no-fee case.

        Params:

        reference_price (float):
            the reference price of the risky asset, denominated in the riskless asset
        Pool (AMM object):
            an AMM object, for example a CoveredCallAMM class, with some current state and reserves
        '''
        assert Pool.fee == 0
        #Check which asset we'll have to swap in to arbitrage
        amm_spot_price = Pool.getSpotPrice()
        if amm_spot_price > reference_price + 100:
            #Find riksy reserves corresponding to that reference price after arbitrage
            final_risky_reserves = Pool.getRiskyReservesGivenSpotPrice(reference_price)
            #Calculate risky asset to swap in
            amount_risky_to_swap = final_risky_reserves - Pool.reserves_risky
            #Perform swap
            _ = Pool.swapAmountInRisky(amount_risky_to_swap)
        elif amm_spot_price < reference_price - 100:
            #Find riskless reserves corresponding to that reference price after arbitrage
            final_risky_reserves = Pool.getRiskyReservesGivenSpotPrice(reference_price)
            final_riskless_reserves = Pool.getRisklessGivenRisky(final_risky_reserves)
            #Calculate riskless asset to swap in 
            amount_riskless_to_swap = final_riskless_reserves - Pool.reserves_riskless
            #Perform swap
            _ = Pool.swapAmountInRiskless(amount_riskless_to_swap)

    def arbitrageExactlyNonZeroFee(self, reference_price, Pool):
        '''
        Arbitrage the difference between marginal price in the pool and the reference price at the time of the call of the function in the non-zero fee case.
        '''
        amm_marginal_price = Pool.getMarginalPrice()
        if amm_marginal_price - 200 > reference_price: 
            #We want to minimize marginal_price_after_trade(amount_in) - reference_price
            def func(x):
                return Pool.getMarginalPriceAfterVirtualSwapAmountInRisky(x) - reference_price
            sol = scipy.optimize.bisect(func, 1e-15, 1 - Pool.reserves_risky - 1e-15)
            amount_to_trade = sol
            #Perform_swap
            amount_out, _ = Pool.swapAmountInRisky(amount_to_trade)
            Pool.arb_risky_balance -= amount_to_trade
            Pool.arb_riskless_balance += amount_out
            print('profitable arb')
            print("risky balance delta: ", Pool.arb_risky_balance * reference_price)
            print("riskless balance delta: ", Pool.arb_riskless_balance)
            print("cfmm risky balance: ", Pool.reserves_risky)
            print("cfmm riskless balance: ", Pool.reserves_riskless)
        elif amm_marginal_price + 200 < reference_price:
            def func(y):
                return Pool.getMarginalPriceAfterVirtualSwapAmountInRiskless(y) - reference_price
            sol = scipy.optimize.bisect(func, 1e-15, Pool.K - Pool.reserves_riskless - 1e-15)
            amount_to_trade = sol
            amount_out, _ = Pool.swapAmountInRiskless(amount_to_trade)
            Pool.arb_riskless_balance -= amount_to_trade
            Pool.arb_risky_balance += amount_out
            print('profitable arb')
            print("risky balance delta: ", Pool.arb_risky_balance * reference_price)
            print("riskless balance delta: ", Pool.arb_riskless_balance)
            print("cfmm risky balance: ", Pool.reserves_risky)
            print("cfmm riskless balance: ", Pool.reserves_riskless)


            
