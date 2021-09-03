# RMMS simulations

This project is intended to investigate the replication of payoffs using custom Constant Function Market Makers (CFMMs) in the spirit of the 2021 paper from [Angeris, Evans and Chitra.](https://stanford.edu/~guillean/papers/rmms.pdf) For now it only focuses on the Covered Call replication. The project is organized as follows:

``modules`` contains all the simulation toolkit. In particular:

- ``modules/arb.py`` implements the optimal arbitrage logic.
- ``modules/cfmm.py`` implements the actual CFMM pool logic.
- ``modules/utils.py`` contains a number of utility functions (math, geometric brownian motion generation).
- ``modules/simulate.py`` is simply the function used to run an individual simulation.
- ``modules/optimize_fee.py`` contains the logic required to find the optimal fee given some market and pool parameters.

``simulation.py`` is a script used to run individual simulations whose parameters are specified in the ``config.ini`` file.

``optimal_fees_parallel.py`` is a script to run an actual fee optimization routine for a prescribed parameter space (to be specified within the script itself).

``optimal_fees_visualization.py`` is a script that generates a visual representation of the output of a fee optimization routine.

``error_distribution.py`` is a script to plot the distribution of errors given some market and pool parameters for different fee regimes.

All the different functions and design choices are documented in a separate document.