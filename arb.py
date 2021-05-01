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
        if amm_spot_price > reference_price and Pool.reserves_risky < 0.999:
            #Swap the risky asset for the riskless asset to move the price down
            #Find the smallest swap that would satisfy amm_spot < reference_price using virtual swaps
            spot_price_after_trade = amm_spot_price
            amount_in = 0.00001
            while spot_price_after_trade > reference_price and amount_in + Pool.reserves_risky < 0.999:
                spot_price_after_trade = Pool.getSpotPriceAfterVirtualSwapAmountInRisky(amount_in)
                amount_in += 0.00001
            #Actually perform the swap
            _ = Pool.swapAmountInRisky(amount_in)
        elif amm_spot_price < reference_price and Pool.reserves_riskless < 999.9:
            #Swap the riskless asset for the risky asset to move the price up
            spot_price_after_trade = amm_spot_price
            amount_in = 0.1
            while spot_price_after_trade < reference_price and amount_in + Pool.reserves_riskless < 999:
                spot_price_after_trade = Pool.getSpotPriceAfterVirtualSwapAmountInRiskless(amount_in)
                amount_in += 0.1
            #Actually perform the swap
            _ = Pool.swapAmountInRiskless(amount_in)
