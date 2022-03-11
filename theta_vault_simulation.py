'''
Run an individual simulation from the config.ini parameters 
and display and/or record the results.
'''

from configparser import ConfigParser

import matplotlib.pyplot as plt 
import numpy as np

from modules import cfmm
from scipy import optimize
from scipy.stats import norm
from modules.arb import arbitrageExactly
from modules.utils import getRiskyGivenSpotPriceWithDelta, getRisklessGivenRisky, generateGBM

EPSILON = 1e-8

#Import config 
config_object = ConfigParser()
config_object.read("config.ini")

STRIKE_PRICE = float(config_object.get("Pool parameters", "STRIKE_PRICE"))
TIME_TO_MATURITY = float(config_object.get("Pool parameters", "TIME_TO_MATURITY"))
FEE = float(config_object.get("Pool parameters", "FEE"))

INITIAL_REFERENCE_PRICE = float(config_object.get("Price action parameters", "INITIAL_REFERENCE_PRICE"))
ANNUALIZED_VOL = float(config_object.get("Price action parameters", "ANNUALIZED_VOL"))
DRIFT = float(config_object.get("Price action parameters", "DRIFT"))
TIME_HORIZON = float(config_object.get("Price action parameters", "TIME_HORIZON"))
TIME_STEPS_SIZE = float(config_object.get("Price action parameters", "TIME_STEPS_SIZE"))

TAU_UPDATE_FREQUENCY = float(config_object.get("Simulation parameters", "TAU_UPDATE_FREQUENCY"))
SIMULATION_CUTOFF = float(config_object.get("Simulation parameters", "SIMULATION_CUTOFF"))
SEED = int(config_object.get("Simulation parameters", "SEED"))
MAX_DIVERGENCE = float(config_object.get("Simulation parameters", "MAX_DIVERGENCE"))

IS_CONSTANT_PRICE = config_object.getboolean("Simulation parameters", "IS_CONSTANT_PRICE")
PLOT_PRICE_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PRICE_EVOL")
PLOT_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_EVOL")
PLOT_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_DRIFT")
SAVE_PRICE_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PRICE_EVOL")
SAVE_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_EVOL")
SAVE_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_DRIFT")

#Initialize pool parameters
sigma = ANNUALIZED_VOL
initial_tau = TIME_TO_MATURITY
K = STRIKE_PRICE
fee = FEE
gamma = 1 - FEE
np.random.seed(SEED)

#Stringify for plotting
gamma_str = str(1 - fee)
sigma_str = str(sigma)
K_str = str(K)

#Initialize pool and arbitrager objects
Pool = cfmm.CoveredCallAMM(0.5, K, sigma, .3, fee)

#Initialize GBM parameters
T = TIME_HORIZON
dt = TIME_STEPS_SIZE
S0 = INITIAL_REFERENCE_PRICE


t, S = generateGBM(T, DRIFT, ANNUALIZED_VOL, S0, dt)

if IS_CONSTANT_PRICE:
    length = len(S)
    constant_price = []
    for i in range(length):
        constant_price.append(S0)
    S = constant_price

plt.plot(t, S)
plt.show()

# next_initial_S = next initial spot price
# K1 = original strike price 
# v1 = original sigma 
# t1 = original maturity 
# inv1 = original invariant 
# K2 = next strike price 
# v2 = next sigma 
# t2 = next maaturity 
# p0 = original implied price
def findNextPool(initial_S, K1, v1, t1, inv1, K2, v2, t2, p0):
    H1 = (K2*norm.cdf(np.log(initial_S/K2)/(v2*np.sqrt(t2))-0.5*v2*np.sqrt(t2)))/norm.cdf(-np.log(initial_S/K2)/(v2*np.sqrt(t2))-0.5*v2*np.sqrt(t2))
    H2 = (K1*norm.cdf(np.log(p0/K1)/(v1*np.sqrt(t1))-0.5*v1*np.sqrt(t1))+inv1)/norm.cdf(-np.log(p0/K1)/(v1*np.sqrt(t1))-0.5*v1*np.sqrt(t1))
    return H1 - H2

def findRootNextPool(initial_S, K1, v1, t1, inv1, K2, v2, t2, p0):
    root = optimize.root(findNextPool, initial_S, (K1, v1, t1, inv1, K2, v2, t2, p0))
    return root.x[0]


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

dtau = TAU_UPDATE_FREQUENCY

# 1. Check spot price
# 2. Set strike price that is within x% of spot price (25%)
# 3. Initialize CFMM 
# 4. Run arbitraguer, check divergence from strike price at each timestep, if 1 + x% > 1 - spot_price/strike_price > 1 - x%
# 5. Move to new CFMM with strike x% +/- i with expiry + 1 month
max_index = 0
def runStrategyWithArbitrage(pool, current_index):
    for i in range(len(S) - current_index):
        #Update pool's time to maturity
        theoretical_tau = initial_tau - t[i + current_index]
        next_start_index = i + current_index
        print(vars(pool))
        
        if i % dtau == 0:
            pool.tau = initial_tau - t[i + current_index]
            #Changing tau changes the value of the invariant even if no trade happens
            pool.invariant = pool.reserves_riskless - pool.getRisklessGivenRiskyNoInvariant(pool.reserves_risky)
            spot_price_array.append(pool.getSpotPrice())
            # _, max_marginal_price = Pool.virtualSwapAmountInRiskless(EPSILON)
            # _, min_marginal_price = Pool.virtualSwapAmountInRisky(EPSILON)

        if pool.tau >= 0:
            #Perform arbitrage step
            arbitrageExactly(S[i + current_index], pool)
            max_marginal_price_array.append(pool.getMarginalPriceSwapRisklessIn(0))
            min_marginal_price_array.append(pool.getMarginalPriceSwapRiskyIn(0))
            #Get reserves given the reference price in the zero fees case
            theoretical_reserves_risky = getRiskyGivenSpotPriceWithDelta(S[i + current_index], pool.K, pool.sigma, theoretical_tau)
            theoretical_reserves_riskless = getRisklessGivenRisky(theoretical_reserves_risky, pool.K, pool.sigma, theoretical_tau)
            theoretical_lp_value = theoretical_reserves_risky*S[i + current_index] + theoretical_reserves_riskless
            theoretical_lp_value_array.append(theoretical_lp_value)
            effective_lp_value_array.append(pool.reserves_risky*S[i + current_index] + pool.reserves_riskless)
            if i + current_index >= len(S) - 1:
                print("breaking") 
                max_index = i
                break
            if 1 + MAX_DIVERGENCE < (S[i + current_index] / K) or 1 - MAX_DIVERGENCE < 1 - (S[i + current_index] / K):
                new_K = 0
                if S[i + current_index] > K:
                    new_K = S[i + current_index] + .20*(S[i + current_index])
                else:
                    new_K = S[i + current_index] - .20*(S[i + current_index])
                next_pool_initial_price = findRootNextPool(pool.getSpotPrice(), pool.K, pool.sigma, pool.tau, pool.invariant, new_K, pool.sigma, pool.tau, pool.getSpotPrice())
#                print("implied_price", pool.getSpotPrice())
#                print("next_price", next_pool_initial_price)
                next_risky = getRiskyGivenSpotPriceWithDelta(next_pool_initial_price, new_K, pool.sigma, pool.tau)
                next_pool = cfmm.CoveredCallAMM(next_risky, new_K, sigma, pool.tau, fee)
#                print("next pool", vars(next_pool))
                runStrategyWithArbitrage(next_pool, next_start_index + 1)
                break
        if pool.tau < 0: 
            max_index = i
            break
        max_index = i

runStrategyWithArbitrage(Pool, 0)

# plt.plot(fees, mse, 'o')
# plt.xlabel("Fee")
# plt.ylabel("MSE")
# plt.title("Mean square error with theoretical payoff as a function of the fee parameter\n" + r"$\sigma = 0.5$, $K = 1100$, $\gamma = 1$, $\mathrm{d}\tau = 30 \ \mathrm{days}$")
# plt.show()

theoretical_lp_value_array = np.array(theoretical_lp_value_array)
effective_lp_value_array = np.array(effective_lp_value_array)

#Mean square error
mse = np.square(np.subtract(theoretical_lp_value_array, effective_lp_value_array)/theoretical_lp_value_array).mean()

if PLOT_PRICE_EVOL: 
    plt.plot(t[0:max_index], S[0:max_index], label = "Reference price")
    # plt.plot(t[0:max_index], spot_price_array, label = "Pool spot price")
    plt.plot(t[0:max_index], min_marginal_price_array[0:max_index], label = "Price sell risky")
    plt.plot(t[0:max_index], max_marginal_price_array[0:max_index], label = "Price buy risky")
    plt.title("Arbitrage between CFMM and reference price\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $\tau_0 = {tau}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=round(1-FEE, 3), dt=round(24*TIME_STEPS_SIZE*365), tau = TIME_TO_MATURITY)+" hours"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TIME_STEPS_SIZE)+"_seed"+str(SEED)
    filename = 'price_evol_'+params_string+'.svg'
    plt.plot()
    if SAVE_PRICE_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = False)

if PLOT_PAYOFF_EVOL:
    plt.figure()
    plt.plot(t[0:max_index], theoretical_lp_value_array[0:max_index], label = "Theoretical LP value")
    plt.plot(t[0:max_index], effective_lp_value_array[0:max_index], label = "Effective LP value")
    plt.title("Value of LP shares\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $\tau_0 = {tau}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=round(1-FEE, 3), dt=round(24*TIME_STEPS_SIZE*365), tau = TIME_TO_MATURITY)+" hours"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Value (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TAU_UPDATE_FREQUENCY)+"_seed"+str(SEED)
    filename = 'lp_value_'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = True)


if PLOT_PAYOFF_DRIFT:
    plt.figure()
    plt.plot(t[0:max_index], 100*abs(theoretical_lp_value_array[max_index]-effective_lp_value_array[max_index])/theoretical_lp_value_array, label=f"Seed = {SEED}")
    plt.title("Drift of LP shares value vs. theoretical \n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $\tau_0 = {tau}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=1-FEE, dt=TIME_STEPS_SIZE, tau = TIME_TO_MATURITY)+" days"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Drift (%)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TAU_UPDATE_FREQUENCY)+"_seed"+str(SEED)
    filename = 'drift_seed_comparison'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_DRIFT:
        plt.savefig('sim_results/'+filename)
    plt.show()

# print("MSE = ", mse)
# print("final divergence = ", 100*abs(theoretical_lp_value_array[-1] - effective_lp_value_array[-1])/theoretical_lp_value_array[-1], "%")
