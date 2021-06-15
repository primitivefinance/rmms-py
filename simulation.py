import cfmm
import arb
import time_series

import matplotlib.pyplot as plt 
import numpy as np

#Annualized volatility of 58%
sigma = 0.58
#Simulation spanning a year
initial_tau = 1

Pool = cfmm.CoveredCallAMM(0.5, 1500, sigma, initial_tau, 0.000003)
Arbitrager = arb.Arbitrager()

T = 365
mu = 0.00003
#Daily vol from annualized vol
sigma = 0.58/np.sqrt(T)
S0 = 1000
dt = 1
t, S = time_series.generateGBM(T, mu, sigma, S0, dt)

arb_start_balance = 100000

# plt.plot(t, S)
# plt.show()

#Store spot prices after each step
spot_price_array = []
#Marginal price affter each step
marginal_price_array = []

#Array to store the theoretical value of LP shares in the case of a pool with zero fees
theoretical_lp_value_array = []
#Effective value of LP shares with fees
effective_lp_value_array = []
risky_reserve_value_array = []
riskless_reserve_value_array = []


for i in range(len(S)):
    #Update pool's time to maturity
    # Pool.tau = initial_tau - t[i]/365
    #Perform arbitrage step
    print(Pool.reserves_risky)
    Arbitrager.arbitrageExactlyNonZeroFee(S[i], Pool)
    #Get reserves given the reference price in the zero fees case
    theoretical_reserves_risky = Pool.getRiskyReservesGivenSpotPrice(S[i])
    theoretical_reserves_riskless = Pool.getRisklessGivenRisky(theoretical_reserves_risky)
    theoretical_lp_value = theoretical_reserves_risky*S[i] + theoretical_reserves_riskless
    theoretical_lp_value_array.append(theoretical_lp_value)
    effective_lp_value_array.append(Pool.reserves_risky*S[i] + Pool.reserves_riskless)
    risky_reserve_value_array.append(Pool.reserves_risky*S[i])
    riskless_reserve_value_array.append(Pool.reserves_riskless)
    marginal_price_array.append(Pool.getMarginalPrice())
    print('effective LP value: ', effective_lp_value_array[i])
    print('theoretical LP value: ', theoretical_lp_value_array[i])
    print('Day: ', i)

print('Start Risky Value: ', risky_reserve_value_array[0])
print('Start Riskless Value: ', riskless_reserve_value_array[0])
print('End Reference Pricel ', S[364])
print('End Risky Value: ', risky_reserve_value_array[364])
print('End Riskless Value: ', riskless_reserve_value_array[364])
print(Pool.accured_fees)
plt.plot(t, S, label = "Reference price")
plt.plot(t, risky_reserve_value_array, label = "Pool Risky Balance")
plt.plot(t, riskless_reserve_value_array, label = "Pool Riskless Balance")
plt.plot(t, effective_lp_value_array, label = "Effective LP Value")
plt.plot(t, theoretical_lp_value_array, label = "Theoretical LP value")
plt.title("Arbitrage between CFMM and reference price")
plt.xlabel("Time steps (days)")
plt.ylabel("Price (USD)")
plt.legend(loc='best')
plt.show()
