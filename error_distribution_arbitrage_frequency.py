'''
Get some information on error distribution given a fixed fee and some parameters.
'''

from math import inf
from joblib.parallel import Parallel, delayed
from time import time
import numpy as np 
from scipy import stats
from scipy.integrate import quad
import matplotlib.pyplot as plt

from modules.simulate import simulate
from modules import cfmm
from modules import utils

K = 2000
volatility = 0.8
tau = 0.3285421
fee = 0.01
time_horizon = 0.3285421
drift = 1
initial_price = K*0.8
# dt = 0.000913242
gamma = 1 - fee

N_paths = 100

time_steps_size = np.linspace(5.70776e-5, 0.000570776, 6)

means = []
variances = []

plt.figure()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.hot(np.flip(np.linspace(0, 0.66, 7)))))

for dt in time_steps_size:

    def returnError():
        np.random.seed()
        Pool = cfmm.CoveredCallAMM(0.5, K, volatility, tau, fee)
        t, gbm = utils.generateGBM(time_horizon, drift, volatility, initial_price, dt)
        _, _, _, terminal_error = simulate(Pool, t, gbm)
        return terminal_error

    # start = time()
    errors = Parallel(n_jobs = -2, verbose = 0, backend = 'loky')(delayed(returnError)() for i in range(N_paths))
    # end = time()
    # print("runtime = ", end - start)

    errors = np.array(errors)
    shape, loc, scale = stats.lognorm.fit(errors, floc=0)
    m = np.log(scale)
    s = shape
    means.append(np.exp(m + s**2/2))
    variances.append((np.exp(s**2) - 1)*np.exp(2*m + s**2))

    # if fee == fees[-1]:
    #     binwidth = abs((max(errors) - min(errors))/(N_paths/3))
    #     plt.hist(errors, np.arange(min(errors), max(errors) + binwidth, binwidth), density=True)

    def pdf_fit(x):
        return stats.lognorm.pdf(x, shape, loc, scale)

    # print("Integral = ", quad(pdf_fit, 0, inf)[0])

    x = np.linspace(0, 0.15, 1000)
    plt.plot(x, pdf_fit(x), label=r"$dt = {dt}$".format(dt=round(24*dt*365, 1)) + " hours")

    # plt.plot(x, pdf_fit(x), label=r"$\gamma = {gamma}$, $\mu = {mu}$, $\sigma = {sigma}$".format(gamma=round(gamma, 3)))

plt.title("Distribution of errors with fixed parameters for different arbitrage frequencies \n" + 
r"$\sigma = {vol}$, $\mu = {drift}$, $K = {strike}$, $\gamma = {gam}$".format(vol=volatility, drift = drift, strike=K, gam=gamma, dt=round(24*dt*365)) + ", Time horizon = 120 days, Initial price = 0.8*K" +
" \n" + "Lognormal fits over 100 paths")

plt.legend(loc="best")
plt.xlabel("Terminal error")

plt.show(block = False)

plt.figure()
plt.title("Mean error as a function of arbitrage frequency (error bars = fitted variance of the distribution) \n" + 
r"$\sigma = {vol}$, $\mu = {drift}$, $K = {strike}$, $\gamma = {gam}$".format(vol=volatility, drift = drift, strike=K, gam=gamma, dt=round(24*dt*365)) + ", Time horizon = 120 days, Initial price = 0.8*K" +
" \n" + "Lognormal fits over 100 paths")
plt.errorbar(24*np.array(time_steps_size)*365, means, yerr = variances, fmt='o', capsize=5)
plt.xlabel("Arbitrage frequency (hours)")
plt.ylabel("Mean terminal error and variance")

plt.show()