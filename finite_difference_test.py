import numpy as np
import cfmm

#Annualized volatility
sigma = 0.50
sigma_str = str(sigma)
#Simulation spanning a year
initial_tau = 1

K = 1100

fee = 0

Pool = cfmm.CoveredCallAMM(0.5, K, sigma, initial_tau, fee)

EPSILON = 1e-8

#TEST SWAP AMOUNT IN RISKY
x = np.linspace(0.0001, 0.3, 10)
right, _ = Pool.virtualSwapAmountInRisky(x+EPSILON)
left, _ = Pool.virtualSwapAmountInRisky(x)
#RESULT IN USD PER ETH
finite_difference = (right - left)/EPSILON
analytical = Pool.getMarginalPriceSwapRiskyIn(x)
print(finite_difference)
print(analytical)

#TEST SWAP AMOUNT IN RISKLESS
x = np.linspace(0.0001, 100, 10)
# x = 0.000001
right, _ = Pool.virtualSwapAmountInRiskless(x+EPSILON)
left, effective_price = Pool.virtualSwapAmountInRiskless(x)
#RESULT IN USD PER ETH
finite_difference = EPSILON/(right-left)
analytical = Pool.getMarginalPriceSwapRisklessIn(x)
print(finite_difference)
print(analytical)