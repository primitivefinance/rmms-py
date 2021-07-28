from time_series import generateGBM
from simulate import simulate
from optimize_fee import findOptimalFee
from scipy.stats import norm
import matplotlib.pyplot as plt

import numpy as np
import cfmm
from utils import blackScholesCoveredCallSpotPrice
from cfmm import CoveredCallAMM
import matplotlib.pyplot as plt
import arb

def main():
    def getRisklessGivenRisky(risky, K, sigma, tau): 
        if risky == 0:
            return K
        elif risky == 1:
            return 0
        return K*norm.cdf(norm.ppf(1 - risky) - sigma*np.sqrt(tau))

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

    # print(blackScholesCoveredCallSpotPrice(0.5, 1100, 0.5, 1))
    # def BSPriceSimplified(x, K, sigma, tau):
    #     return (K*norm.pdf(norm.ppf(1 - x) - sigma*np.sqrt(tau)))/(norm.pdf(norm.ppf(1 - x)))

    # def furtherSimplified(x, K, sigma, tau):
    #     return (K/(2*np.pi))*np.exp(sigma*np.sqrt(tau)*norm.ppf())

    # print(BSPriceSimplified(0.5, 1100, 0.5, 1))

    # print(BSPriceSimplified(0.000000001, 1100, 0.5, 1))

    # TEST OF SPOT PRICE AT BOUNDARIES
    if False:
        # Annualized volatility
        sigma = 0.50
        # Initial time to expiry
        initial_tau = 1
        # Strike price
        K = 1100
        fee = 0
        initial_amount_risky = 0.5 
        # Study the effect of kinks for small values of tau
        taus = [120/365, 60/365, 30/365]
        x = np.linspace(1e-15, 1-1e-15, 1000000) 
        for tau in taus:
            y = blackScholesCoveredCallSpotPrice(x, Pool.K, Pool.sigma, tau)
            plt.plot(x, y, label=f"tau = {round(tau,2)}")
        plt.title("Reported price behavior for different values of tau \n" + r"$\sigma = {vol}$".format(vol=Pool.sigma) +" ; " r"$K = {strike}$".format(strike=K) + " ; " +r"$\gamma = {gam}$".format(gam=1 - fee))
        plt.xlabel("Risky reserves (ETH)")
        plt.ylabel("Reported price (USD)")
        plt.legend(loc='best')
        plt.show(block = True)

            
        # Zoom in on kinks of the spot price curve
        taus = [0.3, 0.1, 0.05]
        for tau in taus: 
            Pool.tau = tau
            x_near_zero = np.linspace(1e-10, 1e-16, 1000000)
            x_near_one = np.linspace(1-1e-10, 1-1e-16, 1000000)
            s_near_zero = blackScholesCoveredCallSpotPrice(x_near_zero, Pool.K, Pool.sigma, Pool.tau)
            s_near_one = blackScholesCoveredCallSpotPrice(x_near_one, Pool.K, Pool.sigma, Pool.tau)
            plt.plot(x_near_zero, s_near_zero)
            # plt.gca().invert_xaxis()
            plt.show(block = False)
            plt.figure()
            plt.plot(x_near_one, s_near_one)
            plt.show(block = True)  

    # COMPARE ANALYTICAL TO FINITE DIFFERENCES MARGINAL PRICE 
    # CALCULATIONS
    if False: 
        x = np.linspace(0.0001, 0.3, 10)
        right, _ = Pool.virtualSwapAmountInRisky(x+EPSILON)
        left, _ = Pool.virtualSwapAmountInRisky(x)
        #RESULT IN USD PER ETH
        finite_difference = (right - left)/EPSILON
        analytical = Pool.getMarginalPriceSwapRiskyIn(x)
        print(finite_difference)
        print(analytical)

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

    # CHECK THE EFFECT OF UPDATING K ON THE BUY AND SELL PRICES
    if False: 
        #Annualized volatility
        sigma = 0.50
        #Initial time to expiry
        initial_tau = 1
        #Strike price
        K = 1100
        #Fee 
        fee = 0.05
        #Initial amount of the risky asset in the pool
        initial_amount_risky = 0.5 
        #Generate AMM pool
        Pool = cfmm.CoveredCallAMM(initial_amount_risky, K, sigma, initial_tau, fee)
        #The current reserves in the pool
        riskless_reserves = Pool.reserves_riskless
        EPSILON = 1e-8
        print("Before doing anything")
        print("Invariant = ", Pool.invariant)
        print("Max price: ", Pool.getMarginalPriceSwapRisklessIn(0))
        print("Min price: ", Pool.getMarginalPriceSwapRiskyIn(0), "\n")
        
        Pool.tau -= 15/365
        print("Before updating k after the update in tau")
        print("Invariant = ", Pool.invariant)
        print("Max price: ", Pool.getMarginalPriceSwapRisklessIn(0))
        print("Min price: ", Pool.getMarginalPriceSwapRiskyIn(0), "\n")
        Pool.invariant = Pool.reserves_riskless - getRisklessGivenRisky(Pool.reserves_risky, Pool.K, Pool.sigma, Pool.tau)
        print("After updating k after the update in tau")
        print("Invariant = ", Pool.invariant)
        print("Max price: ", Pool.getMarginalPriceSwapRisklessIn(0))
        print("Min price: ", Pool.getMarginalPriceSwapRiskyIn(0), "\n")
        max_price = Pool.getMarginalPriceSwapRisklessIn(0)
        m = 1.1*max_price
        #Initialize arbitrager
        Arbitrager = arb.Arbitrager()
        Arbitrager.arbitrageExactly(m, Pool)
        print("After an arbitrage with m > max_price")
        print("Invariant = ", Pool.invariant)
        print("Max price: ", Pool.getMarginalPriceSwapRisklessIn(0))
        print("Min price: ", Pool.getMarginalPriceSwapRiskyIn(0), "\n")

    # NEGATIVE RESERVES OCCURRENCES TEST
    if False: 
        # Annualized volatility
        sigma = 0.50
        # Initial time to expiry
        initial_tau = 1
        # Strike price
        K = 1100
        fee = 0
        initial_amount_risky = 0.5 
        # Initialize some arbitrary pool
        Pool = cfmm.CoveredCallAMM(initial_amount_risky, K, sigma, initial_tau, fee)
        # The parameters that cause an issue in the main routine
        Pool.tau = 0.5192307692307692
        Pool.invariant = -19.093097109440244
        Pool.reserves_risky = 0.9516935976350682
        Pool.reserves_riskless = 4.665850286101332

        reserves_risky = np.linspace(0.8, 1, 1000)

        # With zero invariant
        Pool.invariant = 0
        reserves_riskless = Pool.getRisklessGivenRisky(reserves_risky)
        plt.plot(reserves_risky, reserves_riskless, label = "With invariant = 0")

        # With "correct" invariant with respect to the original state of the pool
        Pool.invariant = -19.093097109440244
        reserves_riskless = Pool.getRisklessGivenRisky(reserves_risky)
        plt.plot(reserves_risky, reserves_riskless, label = "With 'valid' invariant for the current tau")

        plt.title("Negative invariant causing negative reserves \n" + r"$\sigma = {vol}$".format(vol=Pool.sigma) +" ; " + r"$K = {strike}$".format(strike=Pool.K) + " ; " +r"$\gamma = {gam}$".format(gam=1 - Pool.fee) +"\n" + r"$\tau = {tau}$".format(tau=Pool.tau) + "\n" + r"Initial $\tau = 1$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # optimal_trade = 0.011221844928059747
        # Pool.swapAmountInRisky(optimal_trade)

    #INVARIANT CHANGE TEST
    if False: 
        K = 2100
        initial_tau = 0.165
        sigma = 1.5
        fee = 0
        Pool = cfmm.CoveredCallAMM(0.5, K, sigma, initial_tau, fee)
        print("Invariant before = ", Pool.invariant)
        new_invariant = Pool.reserves_riskless - Pool.getRisklessGivenRiskyNoInvariant(Pool.reserves_risky)
        print("Invariant after = ", Pool.invariant)

    #GBM generation tests
    if False: 

        # The goal is to produce a sensible GBM over a 30 days time windows with an annualized volatility of 150%
        # and some arbitrary drift. Try to change the drift, the unit of volatility etc in such a way that it 
        # produces something that makes sense.

        import time_series

        # Example 1: converted annualized vol to timestep vol, high drift  
        # NOTE: the price is shooting up, completely unrealistic
        initial_price = 1100
        annualized_vol = 1.5
        drift = 0.00434077479319
        # Total duration of the GBM in days
        time_horizon = 30
        # Time steps size in days
        time_steps_size = 0.0138889 #20 minutes
        # Number of time steps in a year
        N_timesteps = 365/time_steps_size
        # Scaled down volatility from annualized volatility
        sigma_timesteps = annualized_vol/np.sqrt(N_timesteps)
        t, S = time_series.generateGBM(time_horizon, drift, sigma_timesteps, initial_price, time_steps_size)
        plt.plot(t, S)
        plt.title("Example 1")
        plt.show()

        # # Example 2: converted annualized vol to vol, low drift
        # # NOTE: the price might as well stay constant, completely unrealistic
        drift = 0.0004
        t, S = time_series.generateGBM(time_horizon, drift, sigma_timesteps, initial_price, time_steps_size)
        plt.plot(t, S)
        plt.title("Example 2")
        plt.show()

        # # Example 3: converted annualized vol to timestep vol, drift somewhere in the middle
        # #NOTE: completely unrealistic, just going up regularly
        drift = 0.004
        t, S = time_series.generateGBM(time_horizon, drift, sigma_timesteps, initial_price, time_steps_size)
        plt.plot(t, S)
        plt.title("Example 3")
        plt.show()

        #Example 4: everything expressed in years, WORKING
        time_horizon = 1
        drift = 1
        time_steps_size = 0.0027397260274
        t, S = time_series.generateGBM(time_horizon, drift, annualized_vol, initial_price, time_steps_size)
        plt.plot(t, S)
        plt.title("Example 4")
        plt.show()

    #Simulate and return errors tests 
    if False: 
        fee = 0.01
        strike = 2000
        initial_price = 0.8*2000
        volatility = 0.5
        drift = 0.5
        time_steps_size = 0.0027397260274
        time_horizon = 1
        initial_tau = 1
        # Pool = CoveredCallAMM(0.5, strike, volatility, initial_tau, fee)
        # t, gbm = generateGBM(time_horizon, drift, volatility, initial_price, time_steps_size)
        # from simulate import simulate
        # mse, terminal_square_error = simulate(Pool, t, gbm)
        from optimize_fee import returnErrors, findOptimalFee
        mse, terminal_deviation = returnErrors(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price)

    #Test max error from fee runtime
    if False: 
        fee = 0.01
        strike = 2000
        initial_price = 0.8*2000
        volatility = 0.5
        drift = 0.5
        time_steps_size = 0.0027397260274
        time_horizon = 1
        initial_tau = 1
        # fee = findOptimalFee(initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price)
        # print(fee)
        from optimize_fee import returnErrors
        def maxErrorFromFee(fee): 
            '''
            Return the max of the average mse and average terminal square error from 100 
            simulations with different price actions given these parameters
            '''
            mse_array = []
            square_terminal_error_array = []
            for i in range(100):
                mse, square_terminal_error = returnErrors(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, strike*0.8)
                mse_array.append(mse)
                square_terminal_error_array.append(square_terminal_error)
            average_mse = np.mean(mse_array)
            average_square_terminal_error = np.mean(square_terminal_error_array)
            return max(average_mse, average_square_terminal_error)
        m = maxErrorFromFee(0.01)

    #Test parallel processing
    if True:
        import time 
        from optimize_fee import returnErrors
        from joblib import Parallel, delayed

        fee = 0.01
        strike = 2000
        initial_price = 0.8*2000
        volatility = 0.5
        drift = 0.5
        time_steps_size = 0.0027397260274
        time_horizon = 1
        initial_tau = 1
        # fee = findOptimalFee(initial_tau, time_steps_size, time_horizon, volatility, drift, strike, initial_price)
        # print(fee)

        def maxErrorFromFeeJoblib(fee): 
            '''
            Return the max of the average mse and average terminal square error from 100 
            simulations with different price actions given these parameters
            '''
            results = Parallel(n_jobs=-2, verbose=1, backend='multiprocessing')(delayed(returnErrors)(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, strike*0.8) for i in range(2))
            print('results = ', results)
            average_mse = np.mean([item[0] for item in results])
            average_square_terminal_error = np.mean([item[1] for item in results])
            return max(average_mse, average_square_terminal_error)

        def maxErrorFromFee(fee): 
            '''
            Return the max of the average mse and average terminal square error from 100 
            simulations with different price actions given these parameters
            '''
            mse_array = []
            square_terminal_error_array = []
            for i in range(2):
                mse, square_terminal_error = returnErrors(fee, initial_tau, time_steps_size, time_horizon, volatility, drift, strike, strike*0.8)
                mse_array.append(mse)
                square_terminal_error_array.append(square_terminal_error)
            average_mse = np.mean(mse_array)
            average_square_terminal_error = np.mean(square_terminal_error_array)
            return max(average_mse, average_square_terminal_error)

        # fees = fee*np.ones(2)
        start = time.time()
        m = maxErrorFromFee(0.01)
        end = time.time()
        print("Without joblib = ", end - start)
        start = time.time()
        m = maxErrorFromFeeJoblib(0.01)
        end = time.time()
        print("With joblib = ", end - start)

        # def returnErrors(fee, initial_tau, timestep_size, time_horizon, volatility, drift, strike, initial_price):
        #     '''
        #     Given some parameters and a gbm, return the errors under 
        #     optimal arbitrage for that gbm
        #     '''
        #     t, gbm = generateGBM(time_horizon, drift, volatility, initial_price, timestep_size)
        #     Pool = cfmm.CoveredCallAMM(0.5, strike, volatility, initial_tau, fee)
        #     _, _, mse, terminal_square_error = simulate(Pool, t, gbm)
        #     return mse, terminal_square_error   

        # start = time.time()
        # m = maxErrorFromFee(0.01)
        # end = time.time()
        # print("Runtime without numba: ", end - start)
        # start = time.time()
        # m = maxErrorFromFee(0.01)
        # end = time.time()
        # print("Runtime with numba (compilation included): ", end - start)
        # start = time.time()
        # m = maxErrorFromFee(0.01)
        # end = time.time()
        # print("Runtime with numba (from cache): ", end - start)

if __name__ == '__main__':
    main()


