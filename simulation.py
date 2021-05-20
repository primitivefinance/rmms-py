import cfmm
import arb
import time_series

import matplotlib.pyplot as plt 
import numpy as np

#Annualized volatility of 58%
sigma = 0.58
#Simulation spanning a year
initial_tau = 1

Pool = cfmm.CoveredCallAMM(0.5, 1500, sigma, initial_tau, 0.01)
Arbitrager = arb.Arbitrager()

T = 365
mu = 0.00003
#Daily vol from annualized vol
sigma = 0.58/np.sqrt(T)
S0 = 1000
dt = 1
t, S = time_series.generateGBM(T, mu, sigma, S0, dt)

# plt.plot(t, S)
# plt.show()

#Store spot prices after each step
spot_price_array = []
#Marginal price affter each step
min_marginal_price_array = []
max_marginal_price_array = []

#Array to store the theoretical value of LP shares in the case of a pool with zero fees
theoretical_lp_value_array = []
#Effective value of LP shares with fees
effective_lp_value_array = []


for i in range(len(S)):
    # if i % 36 == 0: 
    #     print("In progress... ", round(i/365), "%")
    #Update pool's time to maturity
    Pool.tau = initial_tau - t[i]/365
    #Perform arbitrage step
    Arbitrager.arbitrageExactly(S[i], Pool)
    #Get reserves given the reference price in the zero fees case
    theoretical_reserves_risky = Pool.getRiskyReservesGivenSpotPrice(S[i])
    theoretical_reserves_riskless = Pool.getRisklessGivenRisky(theoretical_reserves_risky)
    theoretical_lp_value = theoretical_reserves_risky*S[i] + theoretical_reserves_riskless
    theoretical_lp_value_array.append(theoretical_lp_value)
    effective_lp_value_array.append(Pool.reserves_risky*S[i] + Pool.reserves_riskless)
    spot_price_array.append(Pool.getSpotPrice())
    max_marginal_price_array.append(Pool.getMarginalPriceSwapRisklessIn(0))
    min_marginal_price_array.append(Pool.getMarginalPriceSwapRiskyIn(0))

plt.plot(t, S, label = "Reference price")
plt.plot(t, spot_price_array, label = "Pool spot price")
plt.plot(t, min_marginal_price_array, label = "Min pool price")
plt.plot(t, max_marginal_price_array, label = "Max pool price")
plt.title("Arbitrage between CFMM and reference price")
plt.xlabel("Time steps (days)")
plt.ylabel("Price (USD)")
plt.legend(loc='best')
plt.show()
