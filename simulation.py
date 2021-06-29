from configparser import ConfigParser

import cfmm
import arb
import time_series

import matplotlib.pyplot as plt 
import numpy as np
import scipy
from scipy.stats import norm

import math

EPSILON = 1e-8

#Import config 
config_object = ConfigParser()
config_object.read("config.ini")

STRIKE_PRICE = float(config_object.get("Pool parameters", "STRIKE_PRICE"))
TIME_TO_MATURITY = float(config_object.get("Pool parameters", "TIME_TO_MATURITY"))
FEE = float(config_object.get("Pool parameters", "FEE"))/100

INITIAL_REFERENCE_PRICE = float(config_object.get("Price action parameters", "INITIAL_REFERENCE_PRICE"))
ANNUALIZED_VOL = float(config_object.get("Price action parameters", "ANNUALIZED_VOL"))/100
DRIFT = float(config_object.get("Price action parameters", "DRIFT"))
TIME_HORIZON = float(config_object.get("Price action parameters", "TIME_HORIZON"))
TIME_STEPS_SIZE = float(config_object.get("Price action parameters", "TIME_STEPS_SIZE"))

TAU_UPDATE_FREQUENCY = float(config_object.get("Simulation parameters", "TAU_UPDATE_FREQUENCY"))
SIMULATION_CUTOFF = float(config_object.get("Simulation parameters", "SIMULATION_CUTOFF"))
SEED = int(config_object.get("Simulation parameters", "SEED"))

IS_CONSTANT_PRICE = config_object.getboolean("Simulation parameters", "IS_CONSTANT_PRICE")
PLOT_PRICE_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PRICE_EVOL")
PLOT_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_EVOL")
PLOT_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_DRIFT")
SAVE_PRICE_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PRICE_EVOL")
SAVE_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_EVOL")
SAVE_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_DRIFT")

# Functions for analytic zero fees spot price and reserves calculations
def getRiskyReservesGivenSpotPrice(S, K, sigma, tau):
    '''
    Given some spot price S, get the risky reserves corresponding to that spot price by solving 
    S = -y' = -f'(x) for x. Only useful in the no-fee case.
    '''
    def func(x):
        return S - cfmm.blackScholesCoveredCallSpotPrice(x, K, sigma, tau)
    if S > K:
        sol, r = scipy.optimize.newton(func, 0.01, maxiter=250, disp=False, full_output=True)
    else: 
        sol, r = scipy.optimize.newton(func, 0.5, maxiter=250, disp=False, full_output=True)
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

#Initialize pool parameters
sigma = ANNUALIZED_VOL
initial_tau = TIME_TO_MATURITY
K = STRIKE_PRICE
fee = FEE
np.random.seed(SEED)

#Stringify for plotting
gamma_str = str(1 - fee)
sigma_str = str(sigma)
K_str = str(K)

#Initialize pool and arbitrager objects
Pool = cfmm.CoveredCallAMM(0.5, K, sigma, initial_tau, fee)
Arbitrager = arb.Arbitrager()

#Initialize GBM parameters
T = TIME_HORIZON
mu = DRIFT
dt = TIME_STEPS_SIZE
S0 = INITIAL_REFERENCE_PRICE
# Number of timesteps of the sized used in the simulation
# that there would be in a year.
N_year = 365/dt
# Timestep vol from annualized vol. Example: if each timestep
# represents a day, we need to convert annualized vol to daily 
# vol, which is done by dividing the annualized vol by 
# sqrt(number of days in a year). If every time step is an hour,
# we do the same but dividing by sqrt(number of hours in a year), 
# etc.

sigma_timestep = sigma/np.sqrt(N_year)

t, S = time_series.generateGBM(T, mu, sigma_timestep, S0, dt)

if IS_CONSTANT_PRICE:
    length = len(S)
    constant_price = []
    for i in range(length):
        constant_price.append(S0)
    S = constant_price

plt.plot(t, S)
plt.show()

# Prepare storage variables

# Store spot prices after each step
spot_price_array = []
# Marginal price affter each step
min_marginal_price_array = []
max_marginal_price_array = []

# Array to store the theoretical value of LP shares in the case of a pool with zero fees
theoretical_lp_value_array = []
# Effective value of LP shares with fees
effective_lp_value_array = []

# Mean square error
mse = []

dtau = TAU_UPDATE_FREQUENCY

for i in range(len(S)):
    # if i % 36 == 0: 
    #     print("In progress... ", round(i/365), "%")
    #Update pool's time to maturity
    theoretical_tau = initial_tau - t[i]/365
    
    if i % dtau == 0:
        # print("hey")
        Pool.tau = initial_tau - t[i]/365
        #Changing tau changes the value of the invariant even if no trade happens
        Pool.invariant = Pool.reserves_riskless - getRisklessGivenRisky(Pool.reserves_risky, Pool.K, Pool.sigma, Pool.tau)
    # This is to avoid numerical errors that have been observed when getting 
    # closer to maturity. TODO: Figure out what causes these numerical errors.

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
        # _, max_marginal_price = Pool.virtualSwapAmountInRiskless(EPSILON)
        # _, min_marginal_price = Pool.virtualSwapAmountInRisky(EPSILON)
        max_marginal_price_array.append(Pool.getMarginalPriceSwapRisklessIn(0))
        min_marginal_price_array.append(Pool.getMarginalPriceSwapRiskyIn(0))
        # max_marginal_price_array.append(max_marginal_price)
        # min_marginal_price_array.append(min_marginal_price)
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

if PLOT_PRICE_EVOL: 
    plt.plot(t[0:max_index], S[0:max_index], label = "Reference price")
    # plt.plot(t[0:max_index], spot_price_array, label = "Pool spot price")
    plt.plot(t[0:max_index], min_marginal_price_array[0:max_index], label = "Min pool price")
    plt.plot(t[0:max_index], max_marginal_price_array[0:max_index], label = "Max pool price")
    plt.title("Arbitrage between CFMM and reference price\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau)) + ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)+"_seed"+str(SEED)
    filename = 'price_evol_'+params_string+'.svg'
    plt.plot()
    if SAVE_PRICE_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = False)

if PLOT_PAYOFF_EVOL:
    plt.figure()
    plt.plot(t[0:max_index], theoretical_lp_value_array[0:max_index], label = "Theoretical LP value")
    plt.plot(t[0:max_index], effective_lp_value_array[0:max_index], label = "Effective LP value")
    plt.title("Value of LP shares\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau))+" days"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Value (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)+"_seed"+str(SEED)
    filename = 'lp_value_'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = True)

#     plt.figure()
#     plt.plot(t[0:max_index], 100*abs(theoretical_lp_value_array-effective_lp_value_array)/theoretical_lp_value_array, label="Seed = "+str(s))

if PLOT_PAYOFF_DRIFT:
    plt.title("Drift of LP shares value vs. theoretical \n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=sigma_str, strike=K_str, gam=gamma_str, dt=str(dtau))+" days")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Drift (%)")
    plt.legend(loc='best')
    params_string = "sigma"+sigma_str+"_K"+K_str+"_gamma"+gamma_str+"_dtau"+str(dt)
    filename = 'drift_seed_comparison'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_DRIFT:
        plt.savefig('sim_results/'+filename)
    plt.show()