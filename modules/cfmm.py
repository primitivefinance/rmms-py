"""
Contains the necessary AMM logic.
"""

import math
from math import inf
import scipy
from scipy.stats import norm
from scipy import optimize
import numpy as np

from modules.utils import nonnegative, quantilePrime, blackScholesCoveredCallSpotPrice

EPSILON = 1e-8


class CoveredCallAMM(object):
    """
    A class to represent a two-tokens AMM with the covered call trading function.

    Attributes
    ___________

    reserves_risky: float
        the reserves of the AMM pool in the risky asset
    reserves_riskless: float
        the reserves of the AMM pool in the riskless asset
    tau: float
        the time to maturity for this pool in the desired units
    K: float
        the strike price for this pool
    sigma: float
        the volatility for this pool, scaled to be consistent with the unit of tau (annualized if tau is in years etc)
    invariant: float
        the invariant of the CFMM
    """

    def __init__(self, initial_x, k, sigma, tau, fee):
        """
        Initialize the AMM pool with a starting risky asset reserve as an
        input, calculate the corresponding riskless asset reserve needed to
        satisfy the trading function equation.
        """
        self.reserves_risky = initial_x
        self.K = k
        self.sigma = sigma
        self.tau = tau
        self.initial_tau = tau
        self.invariant = 0
        self.reserves_riskless = self.K * norm.cdf(norm.ppf(1 - initial_x) - self.sigma * np.sqrt(self.tau))
        self.fee = fee
        self.accured_fees = [0, 0]

    def getRisklessGivenRisky(self, risky):
        return self.invariant + self.K * norm.cdf(norm.ppf(1 - risky) - self.sigma * np.sqrt(self.tau))

    def getRisklessGivenRiskyNoInvariant(self, risky):
        return self.K * norm.cdf(norm.ppf(1 - risky) - self.sigma * np.sqrt(self.tau))

    def getRiskyGivenRiskless(self, riskless):
        return 1 - norm.cdf(norm.ppf((riskless - self.invariant) / self.K) + self.sigma * np.sqrt(self.tau))

    def swapAmountInRisky(self, amount_in):
        """
        Swap in some amount of the risky asset and get some amount of the riskless asset in return.

        Returns: 

        amount_out: the amount to be given out to the trader
        effective_price_in_risky: the effective price of the executed trade
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_riskless = self.getRisklessGivenRisky(self.reserves_risky + gamma * amount_in)
        amount_out = self.reserves_riskless - new_reserves_riskless
        self.reserves_risky += amount_in
        self.reserves_riskless -= amount_out
        assert nonnegative(new_reserves_riskless)
        # Update invariant
        self.invariant = self.reserves_riskless - self.getRisklessGivenRiskyNoInvariant(self.reserves_risky)
        effective_price_in_riskless = amount_out / amount_in
        return amount_out, effective_price_in_riskless

    def virtualSwapAmountInRisky(self, amount_in):
        """
        Perform a swap and then revert the state of the pool.

        Returns:

        amount_out: the amount that the trader would get out given the amount in
        effective_price_in_riskless: the effective price the trader would pay for that
        trade denominated in the riskless asset
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_riskless = self.getRisklessGivenRisky(self.reserves_risky + gamma * amount_in)
        if new_reserves_riskless <= 0 or math.isnan(new_reserves_riskless):
            return 0, 0
        assert nonnegative(new_reserves_riskless)
        amount_out = self.reserves_riskless - new_reserves_riskless
        if amount_in == 0:
            effective_price_in_riskless = inf
        else:
            effective_price_in_riskless = amount_out / amount_in
        return amount_out, effective_price_in_riskless

    def swapAmountInRiskless(self, amount_in):
        """
        Swap in some amount of the riskless asset and get some amount of the risky asset in return.

        Returns:

        amount_out: the amount to be given to the trader
        effective_price_in_riskless: the effective price the trader actually paid for that trade
        denominated in the riskless asset
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_risky = self.getRiskyGivenRiskless(self.reserves_riskless + gamma * amount_in)
        assert nonnegative(new_reserves_risky)
        amount_out = self.reserves_risky - new_reserves_risky
        assert nonnegative(amount_out)
        self.reserves_riskless += amount_in
        self.reserves_risky -= amount_out
        # Update invariant
        self.invariant = self.reserves_riskless - self.getRisklessGivenRiskyNoInvariant(self.reserves_risky)
        if amount_in == 0:
            effective_price_in_riskless = inf
        else:
            effective_price_in_riskless = amount_in / amount_out
        return amount_out, effective_price_in_riskless

    def virtualSwapAmountInRiskless(self, amount_in):
        """
        Perform a swap and then revert the state of the pool.

        Returns:

        amount_out: the amount that the trader would get out given the amount in
        effective_price_in_riskless: the effective price the trader would pay for that
        trade denominated in the riskless asset
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_risky = self.getRiskyGivenRiskless(self.reserves_riskless + gamma * amount_in)
        if new_reserves_risky <= 0 or math.isnan(new_reserves_risky):
            return 0, 0
        assert nonnegative(new_reserves_risky)
        amount_out = self.reserves_risky - new_reserves_risky
        if amount_out == 0:
            effective_price_in_riskless = inf
        else:
            effective_price_in_riskless = amount_in / amount_out
        return amount_out, effective_price_in_riskless

    def getSpotPrice(self):
        """
        Get the current spot price (ie "reported price" using CFMM jargon) of
        the risky asset, denominated in the riskless asset, only exact in the
        no-fee case.
        """
        return blackScholesCoveredCallSpotPrice(self.reserves_risky, self.K, self.sigma, self.tau)

    def getMarginalPriceSwapRiskyIn(self, amount_in):
        """
        Returns the marginal price after a trade of size amount_in (in the
        risky asset) with the current reserves (in RISKLESS.RISKY-1).
        See https://arxiv.org/pdf/2012.08040.pdf
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        r = self.reserves_risky
        K = self.K
        sigma = self.sigma
        tau = self.tau
        return gamma * K * norm.pdf(norm.ppf(float(1 - r - gamma * amount_in)) - sigma * np.sqrt(tau)) * quantilePrime(
            1 - r - gamma * amount_in)

    def getMarginalPriceSwapRisklessIn(self, amount_in):
        """
        Returns the marginal price after a trade of size amount_in (in the
        riskless asset) with the current reserves (in RISKLESS.RISKY-1)
        See https://arxiv.org/pdf/2012.08040.pdf
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        R = self.reserves_riskless
        invariant = self.invariant
        K = self.K
        sigma = self.sigma
        tau = self.tau
        if ((gamma * norm.pdf(norm.ppf(float((R + gamma * amount_in - invariant) / K)) + sigma * np.sqrt(tau)) *
             quantilePrime((R + gamma * amount_in - invariant) / K) * (1 / K)) < EPSILON):
            # Infinity
            return 1e8
        else:
            return 1 / (gamma * norm.pdf(
                norm.ppf(float((R + gamma * amount_in - invariant) / K)) + sigma * np.sqrt(tau)) * quantilePrime(
                (R + gamma * amount_in - invariant) / K) * (1 / K))

    def getRiskyReservesGivenSpotPrice(self, s):
        """
        Given some spot price S in the no-fee case, get the risky reserves corresponding to that
        spot price by solving the S = -y' = -f'(x) for x.
        """

        def func(x):
            return s - blackScholesCoveredCallSpotPrice(x, self.K, self.sigma, self.tau)

        sol = scipy.optimize.root(func, self.reserves_risky)
        reserves_risky = sol.x[0]
        return reserves_risky
