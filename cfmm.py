'''
Contains the necessary AMM logic.
'''

import math
from math import nan
import scipy
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

from utils import nonnegative, quantilePrime, blackScholesCoveredCallSpotPrice

EPSILON = 1e-10

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
        fees accrued by the pool over time, separate from the reserves to make
        sure we only execute feasible trades, ordered in risky and then
        riskless asset
    '''

    def __init__(self, initial_x, K, sigma, tau, fee):
        '''
        Initialize the AMM pool with a starting risky asset reserve as an
        input, calculate the corresponding riskless asset reserve needed to
        satisfy the trading function equation.
        '''
        self.reserves_risky = initial_x
        self.K = K
        self.sigma = sigma 
        self.tau = tau
        self.initial_tau = tau
        self.invariant = 0
        # function = lambda y : blackScholesCoveredCall([initial_x, y], self.K, self.sigma, self.tau)
        # #Find solution that satisfies the invariant equation Phi(x,y) = 0
        # y = scipy.optimize.root(function, initial_x, method='lm')
        # self.reserves_riskless = y.x[0]
        self.reserves_riskless = self.K*norm.cdf(norm.ppf(1-initial_x) - self.sigma*np.sqrt(self.tau))
        self.fee = fee
        self.accured_fees = [0,0]

    def getRisklessGivenRisky(self, risky): 
        return self.invariant + self.K*norm.cdf(norm.ppf(1 - risky) - self.sigma*np.sqrt(self.tau))

    def getRisklessGivenRiskyNoInvariant(self, risky):
        return self.K*norm.cdf(norm.ppf(1 - risky) - self.sigma*np.sqrt(self.tau))

    def getRiskyGivenRiskless(self, riskless):
        return 1 - norm.cdf(norm.ppf((riskless - self.invariant)/self.K) + self.sigma*np.sqrt(self.tau))

    def swapAmountInRisky(self, amount_in):
        '''
        Swap in some amount of the risky asset and get some amount of the riskless asset in return.

        Returns: 

        amount_out: the amount to be given out to the trader
        effective_price_in_risky: the effective price of the executed trade
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_riskless = self.getRisklessGivenRisky(self.reserves_risky + gamma*amount_in)
        amount_out = self.reserves_riskless - new_reserves_riskless
        self.reserves_risky += amount_in
        self.reserves_riskless -= amount_out
        assert nonnegative(new_reserves_riskless)
        #Update invariant
        self.invariant = self.reserves_riskless - self.getRisklessGivenRiskyNoInvariant(self.reserves_risky) 
        # print('post invariant: ', self.invariant)
        effective_price_in_riskless = amount_out/amount_in
        return amount_out, effective_price_in_riskless

    def virtualSwapAmountInRisky(self, amount_in): 
        '''
        Perform a swap and then revert the state of the pool. Useful to
        estimate the effective price that one would get in a non analytical way
        (what actually happens at the end of the day in the pool)
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        # print(f"Old reserves riskless = {self.reserves_riskless}")
        # print(f"Old reserves risky = {self.reserves_risky}")
        new_reserves_riskless = self.getRisklessGivenRisky(self.reserves_risky + gamma*amount_in)
        # print(f"New reserves riskless = {new_reserves_riskless}")
        if new_reserves_riskless <= 0 or math.isnan(new_reserves_riskless):
            return 0, 0
        assert nonnegative(new_reserves_riskless)
        # print(f"New reserves risky = {self.reserves_risky + gamma*amount_in}")
        amount_out = self.reserves_riskless - new_reserves_riskless
        # assert nonnegative(amount_out)
        effective_price_in_riskless = amount_out/amount_in
        return amount_out, effective_price_in_riskless

    def swapAmountInRiskless(self, amount_in):
        '''
        Swap in some amount of the riskless asset and get some amount of the
        risky asset in return.
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_risky = self.getRiskyGivenRiskless(self.reserves_riskless + gamma*amount_in)
        assert nonnegative(new_reserves_risky)
        amount_out = self.reserves_risky - new_reserves_risky
        self.reserves_riskless += amount_in
        self.reserves_risky -= amount_out
        #Update invariant
        self.invariant = self.reserves_riskless - self.getRisklessGivenRiskyNoInvariant(self.reserves_risky)
        # print("---------------------invariant here: ", self.invariant)
        effective_price_in_riskless = amount_in/amount_out
        return amount_out, effective_price_in_riskless

    def virtualSwapAmountInRiskless(self, amount_in): 
        '''
        Perform a swap and then revert the state of the pool. Useful to
        estimate the effective price that one would get in a non analytical way
        (what actually happens at the end of the day in the pool)
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        # print(f"Old reserves riskless = {self.reserves_riskless}")
        # print(f"Old reserves risky = {self.reserves_risky}")
        new_reserves_risky = self.getRiskyGivenRiskless(self.reserves_riskless + gamma*amount_in)
        # print(f"New reserves risky = {new_reserves_risky}")
        if new_reserves_risky <= 0 or math.isnan(new_reserves_risky):
            return 0, 0
        assert nonnegative(new_reserves_risky)
        # print(f"New reserves riskless = {self.reserves_riskless + gamma*amount_in}")
        amount_out = self.reserves_risky - new_reserves_risky
        # assert nonnegative(amount_out)
        effective_price_in_riskless = amount_in/amount_out
        return amount_out, effective_price_in_riskless


    def getSpotPrice(self):
        '''
        Get the current spot price (ie "reported price" using CFMM jargon) of
        the risky asset, denominated in the riskless asset, only exact in the
        no-fee case.
        '''
        return blackScholesCoveredCallSpotPrice(self.reserves_risky, self.K, self.sigma, self.tau)

    def getMarginalPriceSwapRiskyIn(self, amount_in):
        '''
        Returns the marginal price after a trade of size amount_in (in the
        risky asset) with the current reserves (in RISKLESS.RISKY-1).
        See https://arxiv.org/pdf/2012.08040.pdf 
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        R = self.reserves_risky 
        k = self.invariant
        K = self.K  
        sigma = self.sigma
        tau = self.tau
        return gamma*K*norm.pdf(norm.ppf(1 - R - gamma*amount_in) - sigma*np.sqrt(tau))*quantilePrime(1 - R - gamma*amount_in)

    def getMarginalPriceSwapRisklessIn(self, amount_in):
        '''
        Returns the marginal price of a trade of size amount_in (in the
        riskless asset) with the current reserves (in RISKLESS.RISKY-1)
        See https://arxiv.org/pdf/2012.08040.pdf  
        '''
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        R = self.reserves_riskless 
        k = self.invariant
        K = self.K  
        sigma = self.sigma
        tau = self.tau
        return  1/(gamma * norm.pdf(norm.ppf((R + gamma*amount_in - k)/K) + sigma*np.sqrt(tau))*quantilePrime((R + gamma*amount_in - k)/K)*(1/K))

    def getRiskyReservesGivenSpotPrice(self, S):
        '''
        Given some spot price S, get the risky reserves corresponding to that
        spot price by solving the S = -y' = -f'(x) for x. Only useful in the
        no-fee case.
        '''
        def func(x):
            return S - blackScholesCoveredCallSpotPrice(x, self.K, self.sigma, self.tau)
        sol = scipy.optimize.root(func, self.reserves_risky)
        reserves_risky = sol.x[0]
        return reserves_risky
