from joblib.parallel import Parallel, delayed
import numpy as np 
import json
import time 
from datetime import datetime
from pathlib import Path

import optimize_fee

# Script used to map a tuple of (volatility, drift, strike price) to the optial
# fee to choose for the pool, i.e. the fee that minimizes the max of the mean 
# square error and the terminal square error. 

# Currently for the case of 30 days to maturity with 1 hour tau update / arbitrage 
# intervals.

# Initial time to maturity in years
INITIAL_TAU = 0.08213552 #30 days
# Time horizon of the GBM in years
TIME_HORIZON = 0.08213552
# Time step size of the GBM in years =
# tau update frequency = arbitrage frequency
# 4 hour here
TIME_STEPS_SIZE = 0.000456621 #4 hours
# Arbitrary strike price, what matters is the difference between 
# initial price and strike price
STRIKE = 2000


# Array storing set of parameters to explore: volatility, drift, 
# initial price distance from strike price*
# *for example if the strike priced is K, a parameter of 0.8 will
# start the simulation with an initial price of 0.8*K

parameters = [np.linspace(0.5, 1.5, 3), np.linspace(-2, 2, 3), [0.8]]
# optimal_fee_array = [[0 for i in range(len(parameters[0]))], [0 for i in range(len(parameters[1]))], [0 for i in range(len(parameters[2]))]]
optimal_fee_array = [
    [
        [0 for i in range(len(parameters[2]))] for i in range(len(parameters[1]))
    ] for i in range(len(parameters[0]))
]

def findOptimalFeeParallel(volatility, drift, strike_proportion):
    return optimize_fee.findOptimalFee(INITIAL_TAU, TIME_STEPS_SIZE, TIME_HORIZON, volatility, drift, STRIKE, STRIKE*strike_proportion)

start_nested = time.time()

# With parallelization of the main loop

# Needs to be debugged: takes longer than non nested parallelization for some reason

# optimal_fee_array = Parallel(n_jobs=-1, verbose=0, backend='loky')(delayed(findOptimalFeeParallel)(volatility, drift, strike_proportion) for strike_proportion in parameters[2] for drift in parameters[1] for volatility in parameters[0])

# end_nested = time.time()

start = time.time()

# Without parallelization of the main loop

for i in range(len(parameters[0])): 
    for j in range(len(parameters[1])):
        for m in range(len(parameters[2])):
            volatility = parameters[0][i]
            drift = parameters[1][j]
            strike_proportion = parameters[2][m]
            initial_price = STRIKE*strike_proportion
            optimal_fee = findOptimalFeeParallel(volatility, drift, initial_price)
            # optimal_fee_array[i][j][m] = optimal_fee

end = time.time()

print("Without nested Joblib: ", end - start)
# print("With nested Joblib: ", end_nested - start_nested)

# data = {}
# data['parameters'] = np.array(parameters)
# data['optimal_fees'] = optimal_fee_array
# now = datetime.now()
# dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
# filename = 'optimization_results_'+ dt_string + '.dat'
# Path('optimization').mkdir(parents=True, exist_ok=True)
# with open('optimization/'+filename, 'w+') as f:
#     json.dump(data, f)


