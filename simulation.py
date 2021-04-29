import cfmm
import arb
import time_series

sigma = 0.2
tau = 1

Pool = cfmm.CoveredCallAMM(0.5, 1000, sigma, tau, 0)

#Quick test

print(Pool.reserves_risky)
print(Pool.reserves_riskless)
print(Pool.getSpotPrice())

Arbitrager = arb.Arbitrager()

spot_price = Pool.getSpotPrice()
reference_price = 0.90*spot_price
Arbitrager.arbitrageExactly(reference_price, Pool)

print(reference_price)
print(Pool.getSpotPrice())