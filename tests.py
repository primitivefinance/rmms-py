import numpy as np
import cfmm
import matplotlib.pyplot as plt

#Simulation parameters

#Annualized volatility
sigma = 0.50
sigma_str = str(sigma)

#Initial time to expiry
initial_tau = 1
#Strike price
K = 1100
#Fee 
fee = 0
#Initial amount of the risky asset in the pool
initial_amount_risky = 0.5 
#Generate AMM pool
Pool = cfmm.CoveredCallAMM(initial_amount_risky, K, sigma, initial_tau, fee)
#The current reserves in the pool
riskless_reserves = Pool.reserves_riskless

EPSILON = 1e-8

Pool.tau -= 1/365

# #TEST SWAP AMOUNT IN RISKY
if False: 
    x = np.linspace(0.0001, 0.3, 10)
    right, _ = Pool.virtualSwapAmountInRisky(x+EPSILON)
    left, _ = Pool.virtualSwapAmountInRisky(x)
    #RESULT IN USD PER ETH
    finite_difference = (right - left)/EPSILON
    analytical = Pool.getMarginalPriceSwapRiskyIn(x)
    print(finite_difference)
    print(analytical)

# #TEST SWAP AMOUNT IN RISKLESS
if False: 
    x = np.linspace(0.0001, 100, 10)
    # x = 0.000001
    right, _ = Pool.virtualSwapAmountInRiskless(x+EPSILON)
    left, effective_price = Pool.virtualSwapAmountInRiskless(x)
    #RESULT IN USD PER ETH
    finite_difference = EPSILON/(right-left)
    analytical = Pool.getMarginalPriceSwapRisklessIn(x)
    print(finite_difference)
    print(analytical)

# TEST THAT THE AMOUNT OUT IS A CONCAVE FUNCTION
# OF THE AMOUNT IN IN BOTH CASES
if False: 
    x = np.linspace(0.0001, 0.99999999*(1 - initial_amount_risky), 1000)
    y = np.linspace(0.0001, 0.99999999*(Pool.K - riskless_reserves), 1000)
    # x = np.linspace(0, 1e-4, 1000)
    # y = np.linspace(0, 1e-3, 1000)
    amounts_out_swap_risky_in, _ = Pool.virtualSwapAmountInRisky(x)
    amounts_out_swap_riskless_in, _ = Pool.virtualSwapAmountInRiskless(y)
    plt.plot(x, amounts_out_swap_risky_in)
    # plt.tight_layout()
    plt.title('Amount out riskless = f(amount in risky)')
    plt.show(block = False, )
    plt.figure()
    plt.plot(y, amounts_out_swap_riskless_in)
    # plt.tight_layout()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.title('Amount out risky = f(amount in riskless)')
    plt.show()

#VIRTUAL SWAPS EFFECTIVE PRICE TESTS
if False: 
    _, effective_price_sell_risky = Pool.virtualSwapAmountInRisky(1e-8)
    _, effective_price_buy_risky = Pool.virtualSwapAmountInRiskless(1e-8)
    theoretical_price_sell = Pool.getMarginalPriceSwapRiskyIn(0)
    theoretical_price_buy = Pool.getMarginalPriceSwapRisklessIn(0)
    print(effective_price_sell_risky)
    print(effective_price_buy_risky)
    print(theoretical_price_sell)
    print(theoretical_price_buy)