import numpy as np
import cfmm
import matplotlib.pyplot as plt

#Annualized volatility
sigma = 0.50
sigma_str = str(sigma)
#Simulation spanning a year
initial_tau = 1

K = 1100

fee = 0

initial_amount_risky = 0.5 
Pool = cfmm.CoveredCallAMM(initial_amount_risky, K, sigma, initial_tau, fee)
riskless_reserves = Pool.reserves_riskless

EPSILON = 1e-8

# #TEST SWAP AMOUNT IN RISKY
# x = np.linspace(0.0001, 0.3, 10)
# right, _ = Pool.virtualSwapAmountInRisky(x+EPSILON)
# left, _ = Pool.virtualSwapAmountInRisky(x)
# #RESULT IN USD PER ETH
# finite_difference = (right - left)/EPSILON
# analytical = Pool.getMarginalPriceSwapRiskyIn(x)
# print(finite_difference)
# print(analytical)

# #TEST SWAP AMOUNT IN RISKLESS
# x = np.linspace(0.0001, 100, 10)
# # x = 0.000001
# right, _ = Pool.virtualSwapAmountInRiskless(x+EPSILON)
# left, effective_price = Pool.virtualSwapAmountInRiskless(x)
# #RESULT IN USD PER ETH
# finite_difference = EPSILON/(right-left)
# analytical = Pool.getMarginalPriceSwapRisklessIn(x)
# print(finite_difference)
# print(analytical)

# TEST THAT THE AMOUNT OUT IS A CONCAVE FUNCTION
# OF THE AMOUNT IN IN BOTH CASES
# x = np.linspace(0.0001, 0.99999999*(1 - initial_amount_risky), 1000)
# y = np.linspace(0.0001, 0.99999999*(Pool.K - riskless_reserves), 1000)
# amounts_out_swap_risky_in, _ = Pool.virtualSwapAmountInRisky(x)
# amounts_out_swap_riskless_in, _ = Pool.virtualSwapAmountInRiskless(y)
# plt.plot(x, amounts_out_swap_risky_in)
# plt.title('Amount out riskless = f(amount in risky)')
# plt.show(block = False)
# plt.figure()
# plt.plot(y, amounts_out_swap_riskless_in)
# plt.title('Amount out risky = f(amount in riskless)')
# plt.show()

#VIRTUAL SWAPS EFFECTIVE PRICE TESTS
Pool.tau -= 1/365
_, effective_price_sell_risky = Pool.virtualSwapAmountInRisky(1e-8)
_, effective_price_buy_risky = Pool.virtualSwapAmountInRiskless(1e-8)
print(effective_price_sell_risky)
print(effective_price_buy_risky)