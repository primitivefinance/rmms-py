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
# fee = 0.005
time_horizon = 0.3285421
drift = 1
initial_price = K*0.8
dt = 0.000913242
# gamma = 1 - fee

N_paths = 150

fees = np.linspace(0.02, 0.09, 5)

plt.figure()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.hot(np.linspace(0, 0.6, 5))))

for fee in fees:

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
    shape, loc, scale = stats.lognorm.fit(errors)
    m = np.log(scale)
    s = shape

    if fee == fees[-1]:
        binwidth = abs((max(errors) - min(errors))/(N_paths/3))
        plt.hist(errors, np.arange(min(errors), max(errors) + binwidth, binwidth), density=True)

    def pdf_fit(x):
        return stats.lognorm.pdf(x, shape, loc, scale)

    # print("Integral = ", quad(pdf_fit, 0, inf)[0])

    gamma = 1 - fee
    x = np.linspace(0, 0.15, 1000)
    plt.plot(x, pdf_fit(x), label=r"$\gamma = {gamma}$".format(gamma=round(gamma, 5)))

    # plt.plot(x, pdf_fit(x), label=r"$\gamma = {gamma}$, $\mu = {mu}$, $\sigma = {sigma}$".format(gamma=round(gamma, 3)))

plt.title("Distribution of errors with fixed parameters for different fees \n" + 
r"$\sigma = {vol}$, $\mu = {drift}$, $K = {strike}$, $d\tau = {dt}$".format(vol=volatility, drift = drift, strike=K, gam=1-fee, dt=8)+" hours" + ", Time horizon = 120 days, Initial price = 0.8*K" +" \n" + "Lognormal fits over 150 paths")

plt.legend(loc="best")
plt.xlabel("Terminal error")

plt.show()