'''
Contains the necessary AMM logic.
'''

import scipy
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

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

    def getRisklessGivenRisky(self, risky): 
        return self.K*norm.cdf(norm.ppf(1 - risky) - self.sigma*np.sqrt(self.tau))

    def getRiskyGivenRiskless(self, riskless):
        return 1 - norm.cdf(norm.ppf(riskless/self.K) + self.sigma*np.sqrt(self.tau))

    def swapAmountInRisky(self, amount_in):
        '''
        Swap in some amount of the risky asset and get some amount of the riskless asset in return.
        '''
        #Save previous reserves 
        old_reserves_riskless = self.reserves_riskless
        effective_amount_in = amount_in*(1 - self.fee)
        self.accured_fees[0] += amount_in*self.fee
        new_reserves_risky = self.reserves_risky + effective_amount_in
        # function = lambda y : blackScholesCoveredCall([new_reserves_risky, y], self.K, self.sigma, self.tau)
        # #Find solution that satisfies the invariant equation Phi(x,y) = 0
        # y = scipy.optimize.root(function, old_reserves_riskless, method='lm')
        # new_reserves_riskless = y.x[0]
        new_reserves_riskless = self.K*norm.cdf(norm.ppf(1-new_reserves_risky) - self.sigma*np.sqrt(self.tau))
        self.reserves_risky = new_reserves_risky
        self.reserves_riskless = new_reserves_riskless
        #Return amount to give to trader
        return old_reserves_riskless - new_reserves_riskless

    def swapAmountInRiskless(self, amount_in):
        '''
        Swap in some amount of the riskless asset and get some amount of the risky asset in return.
        '''
        old_reserves_risky = self.reserves_risky
        effective_amount_in = amount_in*(1-self.fee)
        self.accured_fees[1] += amount_in*self.fee 
        new_reserves_riskless = self.reserves_riskless + effective_amount_in
        # function = lambda x : blackScholesCoveredCall([x, new_reserves_riskless], self.K, self.sigma, self.tau)
        # x = scipy.optimize.root(function, old_reserves_risky, method='lm')
        # new_reserves_risky = x.x[0]
        new_reserves_risky = 1 - norm.cdf(norm.ppf(new_reserves_riskless/self.K) + self.sigma*np.sqrt(self.tau))
        self.reserves_risky = new_reserves_risky
        self.reserves_riskless = new_reserves_riskless
        return old_reserves_risky - new_reserves_risky

    def getSpotPrice(self):
        '''
        Get the current spot price of the risky asset, denominated in the riskless asset.
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

    def getSpotPriceAfterVirtualSwapAmountInRiskless(self, amount_in):
        '''
        Get the spot price that *would* result from swapping in some amount of the riskless asset. Does not change state.
        '''
        #Save current reserves to revert after operation (we want the trade to be virtual)
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        _ = self.swapAmountInRiskless(amount_in)
        spot_price_after_trade = self.getSpotPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        return spot_price_after_trade

    def getSpotPriceAfterVirtualSwapAmountInRisky(self, amount_in):
        '''
        Get the spot price that *would* result from swapping in some amount of the risky asset. Does not change state.
        '''
        current_reserves_riskless = self.reserves_riskless
        current_reserves_risky = self.reserves_risky
        _ = self.swapAmountInRisky(amount_in)
        spot_price_after_trade = self.getSpotPrice()
        self.reserves_riskless = current_reserves_riskless
        self.reserves_risky = current_reserves_risky
        return spot_price_after_trade

    def getRiskyReservesGivenSpotPrice(self, S):
        '''
        Given some spot price S, get the risky reserves corresponding to that spot price by solving the S = -y' = -f'(x) for x.
        '''
        def func(x):
            return S - blackScholesCoveredCallSpotPrice(x, self.K, self.sigma, self.tau)
        sol = scipy.optimize.root(func, self.reserves_risky, method='lm')
        reserves_risky = sol.x[0]
        return reserves_risky

