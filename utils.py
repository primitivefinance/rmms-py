'''
A collection of utility functions including some 
covered call AMM relevant math.
'''

import math
from math import nan
import numpy as np
from scipy.optimize import newton
from scipy.stats import norm

def nonnegative(x):
    if isinstance(x, np.ndarray):
        return (x >= 0).all()
    return x >= 0

def blackScholesCoveredCall(x, K, sigma, tau):
    '''
    Return value of the BS covered call trading function for given reserves and parameters.
    '''
    result = x[1] - K*norm.cdf(norm.ppf(1 - x[0]) - sigma*np.sqrt(tau))
    return result

def quantilePrime(x):
    '''
    Analytical formula for the derivative of the quantile function (inverse of
    the CDF).
    '''
    EPSILON = 1e-16
    if (x > 1 - EPSILON) or (x < 0 + EPSILON):
        return 0
    else:
        return norm.pdf(norm.ppf(x))**-1

def blackScholesCoveredCallSpotPrice(x, K, sigma, tau):
    '''
    Analytical formula for the spot price (reported price) of the BS covered
    call CFMM in the zero fees case.
    '''
    return K*norm.pdf(norm.ppf(1 - x) - sigma*np.sqrt(tau))*quantilePrime(1-x)

# Functions for analytic zero fees spot price and reserves calculations
def getRiskyReservesGivenSpotPrice(S, K, sigma, tau):
    '''
    Given some spot price S, get the risky reserves corresponding to that spot price by solving 
    S = -y' = -f'(x) for x. Only useful in the no-fee case.
    '''
    def func(x):
        return S - blackScholesCoveredCallSpotPrice(x, K, sigma, tau)
    if S > K:
        sol, r = newton(func, 0.01, maxiter=100, disp=False, full_output=True)
    else: 
        sol, r = newton(func, 0.5, maxiter=100, disp=False, full_output=True)
    reserves_risky = r.root
    #The reserves almost don't change anymore at the boundaries, so if we haven't 
    # converged, we return what we logically know to be very close to the actual 
    # reserves.
    if math.isnan(reserves_risky) and S > K:
        return 0
    elif math.isnan(reserves_risky) and S < K:
        return 1
    return reserves_risky

def getRisklessGivenRisky(risky, K, sigma, tau): 
    if risky == 0:
        return K
    elif risky == 1:
        return 0
    return K*norm.cdf(norm.ppf(1 - risky) - sigma*np.sqrt(tau))