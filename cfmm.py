'''
Contains the necessary AMM logic.
'''

import scipy
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-10

def blackScholesCoveredCall(x, K, sigma, tau):
    '''
    Return value of the BS covered call trading function for given reserves and parameters.
    '''
    result = x[1] - K*norm.cdf(norm.ppf(1 - x[0]) - sigma*np.sqrt(tau))
    return result

# #For analytic spot price formula

def quantilePrime(x):
    return norm.pdf(norm.ppf(x))**-1

def blackScholesCoveredCallSpotPrice(x, K, sigma, tau):
    return K*norm.pdf(norm.ppf(1 - x) - sigma*np.sqrt(tau))*quantilePrime(1-x)

class CoveredCallAMM():
    '''
    A class to represent a two-tokens AMM with the covered call trading function.

    Attributes
    ___________

    reserves_risky: float
        the reserves of the AMM pool in the risky asset 
    reserves_riskless: float
        the reserves of the AMM pool in the riskless asset
    accrued_fees: list[float]
        fees accrued by the pool over time, separate from the reserves to make sure we only execute feasible trades, ordered in risky and then riskless asset
    '''

    def __init__(self, initial_x, K, sigma, tau, fee):
        '''
        Initialize the AMM pool with a starting risky asset reserve as an input, calculate the corresponding riskless asset reserve needed to satisfy the trading function equation.
        '''
        self.reserves_risky = initial_x
        self.K = K
        self.sigma = sigma 
        self.tau = tau
        self.invariant = 0
        # function = lambda y : blackScholesCoveredCall([initial_x, y], self.K, self.sigma, self.tau)
        # #Find solution that satisfies the invariant equation Phi(x,y) = 0
        # y = scipy.optimize.root(function, initial_x, method='lm')
        # self.reserves_riskless = y.x[0]
        self.reserves_riskless = self.K*norm.cdf(norm.ppf(1-initial_x) - self.sigma*self.tau)
        self.fee = fee
        self.accured_fees = [0,0]
        self.arb_risky_balance = 0 
        self.arb_riskless_balance = 0 

    def getRisklessGivenRisky(self, risky): 
        return self.K*norm.cdf(norm.ppf(1 - risky) - self.sigma*np.sqrt(self.tau))

    def getRiskyGivenRiskless(self, riskless):
        return 1 - norm.cdf(norm.ppf(riskless/self.K) + self.sigma*np.sqrt(self.tau))

    def swapAmountInRisky(self, amount_in):
        '''
        Swap in some amount of the risky asset and get some amount of the riskless asset in return.

        Returns: 

        amount_out: the amount to be given out to the trader
        effective_price_in_risky: the effective price of the executed trade
        '''
        #Save previous reserves 
        old_reserves_riskless = self.reserves_riskless
        effective_amount_in = amount_in*(1 - self.fee)
        self.accured_fees[0] += amount_in*self.fee
        #The new reserves used for the amount out calculation using only the effective amount in
        new_reserves_risky_for_swap = self.reserves_risky + effective_amount_in
        #The actual new reserves: the full amount in is added
        new_reserves_risky = self.reserves_risky + amount_in
        # function = lambda y : blackScholesCoveredCall([new_reserves_risky, y], self.K, self.sigma, self.tau)
        # #Find solution that satisfies the invariant equation Phi(x,y) = 0
        # y = scipy.optimize.root(function, old_reserves_riskless, method='lm')
        # new_reserves_riskless = y.x[0]
        new_reserves_riskless = self.K*norm.cdf(norm.ppf(1-new_reserves_risky_for_swap) - self.sigma*np.sqrt(self.tau)) + self.invariant
        self.reserves_risky = new_reserves_risky
        self.reserves_riskless = new_reserves_riskless
        #Update invariant
        self.invariant = self.reserves_riskless - self.K*norm.cdf(norm.ppf(1 - self.reserves_risky) - self.sigma*np.sqrt(self.tau))
        #Return amount to give to trader
        amount_out = old_reserves_riskless - new_reserves_riskless
        effective_price_in_riskless = amount_out/amount_in
        return amount_out, effective_price_in_riskless

    def swapAmountInRiskless(self, amount_in):
        '''
        Swap in some amount of the riskless asset and get some amount of the risky asset in return.
        '''
        old_reserves_risky = self.reserves_risky
        effective_amount_in = amount_in*(1-self.fee)
        self.accured_fees[1] += amount_in*self.fee 
        new_reserves_riskless_for_swap = self.reserves_riskless + effective_amount_in
        new_reserves_riskless = self.reserves_riskless + amount_in
        # function = lambda x : blackScholesCoveredCall([x, new_reserves_riskless], self.K, self.sigma, self.tau)
        # x = scipy.optimize.root(function, old_reserves_risky, method='lm')
        # new_reserves_risky = x.x[0]
        new_reserves_risky = 1 - norm.cdf(norm.ppf((new_reserves_riskless_for_swap-self.invariant)/self.K) + self.sigma*np.sqrt(self.tau))
        self.reserves_risky = new_reserves_risky
        self.reserves_riskless = new_reserves_riskless
        #update invariant
        self.invariant = self.reserves_riskless - self.K*norm.cdf(norm.ppf(1 - self.reserves_risky) - self.sigma*np.sqrt(self.tau))
        amount_out = old_reserves_risky - new_reserves_risky
        effective_price_in_riskless = amount_in/amount_out
        return amount_out, effective_price_in_riskless

    def getSpotPrice(self):
        '''
        Get the current spot price of the risky asset, denominated in the riskless asset, only exact in the no-fee case.
        '''
        #TODO: Calculate analytic spot price formula, including at the limits, in order to avoid solver precision issues
        # def invariant(x):
        #     return blackScholesCoveredCall(x, self.K, self.sigma, self.tau)
        #Get gradient vector at the given reserves
        # gradient = scipy.optimize.approx_fprime([self.reserves_risky, self.reserves_riskless], invariant, 1e-10)
        #Calculate spot price denominated in the riskless asset
        # spot = gradient[0]/gradient[1]
        # return spot
        return blackScholesCoveredCallSpotPrice(self.reserves_risky, self.K, self.sigma, self.tau)

    def getMarginalPrice(self): 
        '''
        Get the marginal price given by the pool by swapping an espilon amount of either asset. Should return the same thing as getSpotPrice in the no-fee case. The marginal price is denominated in the riskless asset.
        '''
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        current_invariant = self.invariant
        _, marginal_price = self.swapAmountInRisky(EPSILON)
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        self.invariant = current_invariant
        return marginal_price


    def getSpotPriceAfterVirtualSwapAmountInRiskless(self, amount_in):
        '''
        Get the spot price that *would* result from swapping in some amount of the riskless asset. Does not change state.
        '''
        #Save current reserves to revert after operation (we want the trade to be virtual)
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        current_invariant = self.invariant
        _ = self.swapAmountInRiskless(amount_in)
        spot_price_after_trade = self.getSpotPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        self.invariant = current_invariant
        return spot_price_after_trade

    def getSpotPriceAfterVirtualSwapAmountInRisky(self, amount_in):
        '''
        Get the spot price that *would* result from swapping in some amount of the risky asset. Does not change state.
        '''
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        current_invariant = self.invariant
        _ = self.swapAmountInRisky(amount_in)
        spot_price_after_trade = self.getSpotPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        self.invariant = current_invariant
        return spot_price_after_trade

    def getMarginalPriceAfterVirtualSwapAmountInRiskless(self, amount_in): 
        '''
        Get the marginal price that *would* result from swapping in some amount of the riskless asset.
        '''
        amount_in = float(amount_in)
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        current_invariant = self.invariant
        _, _ = self.swapAmountInRiskless(amount_in)
        marginal_price_after_trade = self.getMarginalPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        self.invariant = current_invariant
        return marginal_price_after_trade


    def getMarginalPriceAfterVirtualSwapAmountInRisky(self, amount_in):
        '''
        Get the marginal price that *would* result from swapping in some amount of the risky asset.
        '''
        amount_in = float(amount_in)
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        current_invariant = self.invariant
        _, _ = self.swapAmountInRisky(amount_in)
        marginal_price_after_trade = self.getMarginalPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        self.invariant = current_invariant
        return marginal_price_after_trade

    def getRiskyReservesGivenSpotPrice(self, S):
        '''
        Given some spot price S, get the risky reserves corresponding to that spot price by solving the S = -y' = -f'(x) for x. Only useful in the no-fee case.
        '''
        def func(x):
            return S - blackScholesCoveredCallSpotPrice(x, self.K, self.sigma, self.tau)
        sol = scipy.optimize.root(func, self.reserves_risky, method='lm')
        reserves_risky = sol.x[0]
        return reserves_risky

