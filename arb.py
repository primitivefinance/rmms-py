'''
Arbitrage logic
'''

class Arbitrager():
    '''
    A class to represent an arbitrager who will look at a reference price of the risky asset, denominated in the riskless asset, the price in an AMM pool, and arbitrage the difference. Ideal arbitrager with infinite portfolio of either assets.
    '''

    def arbitrageExactly(self, reference_price, Pool):
        '''
        Arbitrage the difference *exactly* at the time of the call to the function. Naive implement with increments in virtual amount traded before then doing a trade. 

        TODO: Improve by using a solver to find the amount to trade while limiting computational steps.

        Params:

        reference_price (float):
            the reference price of the risky asset, denominated in the riskless asset
        Pool (AMM object):
            an AMM object, for example a CoveredCallAMM class, with some current state and reserves
        '''
        #Check which asset we'll have to swap in to arbitrage
        amm_spot_price = Pool.getSpotPrice()
        if amm_spot_price > reference_price:
            #Find riksy reserves corresponding to that reference price after arbitrage
            final_risky_reserves = Pool.getRiskyReservesGivenSpotPrice(reference_price)
            #Calculate risky asset to swap in
            amount_risky_to_swap = final_risky_reserves - Pool.reserves_risky
            #Perform swap
            _ = Pool.swapAmountInRisky(amount_risky_to_swap)
        elif amm_spot_price < reference_price:
            #Find riskless reserves corresponding to that reference price after arbitrage
            final_risky_reserves = Pool.getRiskyReservesGivenSpotPrice(reference_price)
            final_riskless_reserves = Pool.getRisklessGivenRisky(final_risky_reserves)
            #Calculate riskless asset to swap in 
            amount_riskless_to_swap = final_riskless_reserves - Pool.reserves_riskless
            #Perform swap
            _ = Pool.swapAmountInRiskless(amount_riskless_to_swap)
            
