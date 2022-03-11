### TODO

Goal: A simulation that generates a gbm and a cfmm, arbitrageur arbs the pool according to ref market (gbm), determine at what point to do rebalance and whether or not rebalance is optimal vs swap, run many such sims in parallel to see how frequently rebalance is optimal

entities: cfmm, arbitrageur, 
 
1. ~~Newtons method~~

Test cases
1. The pool hasn't been arbitraged in n timesteps, so do the rebalance 
2. The pool is expired, so do the rebalance
3. The pool reaches x% the way to maturity
4. The reference price is x% away from the strike price 

## TODO Tomorrow
1. Move the root finding fn to theta vault simulation file
2. Using result from root call getRiskyGivenSpotPriceWithDelta to determine new pool initial_x
3. Move liquidity from old pool to new pool
4. Continue arbitrage routine until divergence_max is reached, repeat
