import cfmm
import arb
import time_series

import matplotlib.pyplot as plt 
import numpy as np
import scipy
from scipy.stats import norm

import math

from random import seed

# #For analytic spot price formula


def getRiskyReservesGivenSpotPrice(S, K, sigma, tau):
    '''
    Given some spot price S, get the risky reserves corresponding to that spot price by solving the S = -y' = -f'(x) for x. Only useful in the no-fee case.
    '''
    def func(x):
        return S - cfmm.blackScholesCoveredCallSpotPrice(x, K, sigma, tau)
    if S > K:
        sol, r = scipy.optimize.newton(func, 0.01, maxiter=250, disp=False, full_output=True)
    else: 
        sol, r = scipy.optimize.newton(func, 0.5, maxiter=250, disp=False, full_output=True)
    reserves_risky = r.root
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

#Annualized volatility
sigma = 0.50
#Simulation spanning a year
initial_tau = 1

fees = np.linspace(0,0.05, 10)

mse = []

# fees = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

s = 5

seeds = [5]

for s in seeds: 

    np.random.seed(s)

    fees = [0]

    for fee in fees: 

        gamma_str = str(1 - fee)

        #Annualized volatility
        sigma = 0.50
        sigma_str = str(sigma)
        #Simulation spanning a year
        initial_tau = 1

        K = 1100

        Pool = cfmm.CoveredCallAMM(0.5, K, sigma, initial_tau, fee)
        Arbitrager = arb.Arbitrager()

        K_str = str(K);

        T = 365
        mu = 0.00003
        #Daily vol from annualized vol
        sigma = sigma/np.sqrt(T)
        S0 = 1000
        dt = 1
        t, S = time_series.generateGBM(T, mu, sigma, S0, dt)

        length = len(S)
        constant_price = []
        for i in range(length):
            constant_price.append(S0)

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
            theoretical_tau = initial_tau - t[i]/365
            dtau = 1
            if i % dtau == 0:
            #     # print("hey")
                Pool.tau = initial_tau - t[i]/365
            #Make sure tau > 0.005 to avoid numerical errors
            if Pool.tau > 0.05:
                #Perform arbitrage step
                Arbitrager.arbitrageExactly(S[i], Pool)
                #Get reserves given the reference price in the zero fees case
                theoretical_reserves_risky = getRiskyReservesGivenSpotPrice(S[i], Pool.K, Pool.sigma, theoretical_tau)
                theoretical_reserves_riskless = getRisklessGivenRisky(theoretical_reserves_risky, Pool.K, Pool.sigma, theoretical_tau)
                if S[i] > 2300 and S[i] < 2350:
                    print(theoretical_reserves_risky)
                    print(theoretical_reserves_riskless)
                theoretical_lp_value = theoretical_reserves_risky*S[i] + theoretical_reserves_riskless
                theoretical_lp_value_array.append(theoretical_lp_value)
                effective_lp_value_array.append(Pool.reserves_risky*S[i] + Pool.reserves_riskless)
                spot_price_array.append(Pool.getSpotPrice())
                max_marginal_price_array.append(Pool.getMarginalPriceSwapRisklessIn(0))
                min_marginal_price_array.append(Pool.getMarginalPriceSwapRiskyIn(0))
            if Pool.tau < 0.05: 
                max_index = i
                break
            max_index = i

        mse.append(np.square(np.subtract(theoretical_lp_value_array, effective_lp_value_array)).mean())

    # plt.plot(fees, mse, 'o')
    # plt.xlabel("Fee")
    # plt.ylabel("MSE")
    # plt.title("Mean square error with theoretical payoff as a function of the fee parameter\n" + r"$\sigma = 0.5$, $K = 1100$, $\gamma = 1$, $\mathrm{d}\tau = 30 \ \mathrm{days}$")
    # plt.show()

    theoretical_lp_value_array = np.array(theoretical_lp_value_array)
    effective_lp_value_array = np.array(effective_lp_value_array)

    plt.plot(t[0:max_index], S[0:max_index], label = "Reference price")
    # plt.plot(t[0:max_index], spot_price_array, label = "Pool spot price")
    plt.plot(t[0:max_index], min_marginal_price_array, label = "Min pool price")
    plt.plot(t[0:max_index], max_marginal_price_array, label = "Max pool price")
    plt.title("Arbitrage between CFMM and reference price\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau)) + ", np.seed("+str(s)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)+"_seed"+str(s)
    filename = 'price_evol_'+params_string+'.svg'
    plt.savefig('sim_results/'+filename)
    plt.show(block = True)

    plt.figure()
    plt.plot(t[0:max_index], theoretical_lp_value_array, label = "Theoretical LP value")
    plt.plot(t[0:max_index], effective_lp_value_array, label = "Effective LP value")
    plt.title("Value of LP shares\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau))+" days"+ ", np.seed("+str(s)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Value (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)+"_seed"+str(s)
    filename = 'lp_value_'+params_string+'.svg'
    plt.savefig('sim_results/'+filename)
    plt.show(block = True)

#     plt.figure()
#     plt.plot(t[0:max_index], 100*abs(theoretical_lp_value_array-effective_lp_value_array)/theoretical_lp_value_array, label="Seed = "+str(s))


# plt.title("Drift of LP shares value vs. theoretical \n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau))+" days")
# plt.xlabel("Time steps (days)")
# plt.ylabel("Drift (%)")
# plt.legend(loc='best')
# params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)
# filename = 'drift_seed_comparison'+params_string+'.svg'
# plt.savefig('sim_results/'+filename)
# plt.show()