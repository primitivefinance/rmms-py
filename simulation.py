import cfmm
import arb
import time_series

import matplotlib.pyplot as plt 
import numpy as np

sigma = 0.58
tau = 1

Pool = cfmm.CoveredCallAMM(0.5, 1000, sigma, tau, 0)

#Quick test

print(Pool.reserves_risky)
print(Pool.reserves_riskless)
print(Pool.getSpotPrice())

Arbitrager = arb.Arbitrager()

# spot_price = Pool.getSpotPrice()
# reference_price = 1.1*spot_price
# Arbitrager.arbitrageExactly(reference_price, Pool)

# print("Reference price: ", reference_price)
# print("Spot price after trade: ", Pool.getSpotPrice())

T = 365
mu = 0.00003
sigma = 0.5/np.sqrt(T)
S0 = 1000
dt = 1
t, S = time_series.generateGBM(T, mu, sigma, S0, dt)

# plt.plot(t, S)
# plt.show()

#Store spot prices after each step
spot_price_array = []

for i in range(len(S)):
    if i % 30 == 1: 
        print("In progress... 1")
    spot_price_array.append(Pool.getSpotPrice())
    Arbitrager.arbitrageExactly(S[i], Pool)

plt.plot(t, S)
plt.plot(t, spot_price_array)
plt.show()
