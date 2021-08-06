from datetime import datetime
import numpy as np

def evolution_strategy(
    func,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters,
    pool):

    # Assume initial params is a 1-D array
    num_params = len(initial_params)
    params = initial_params
    t0 = datetime.now()
    N = np.random.randn(population_size, num_params)
    if pool == -1:
        ### slow way
        offspring_rewards = np.zeros(population_size) # stores the reward
        # # loop through each "offspring"
        for i in range(population_size):
            params_ = params + sigma*N[i]
            R[i] = func(params_)
    else:
        ### fast way
        params_ = [params + sigma*N[j] for j in range(population_size)]
        offspring_rewards = pool.map(func, params_)
        offspring_rewards = np.array(offspring_rewards)
    mu = offspring_rewards.mean()
    s = offspring_rewards.std()
    if s == 0:
        # we can't apply the following equation
        print("Skipping")
        return -1  
    advantage = (offspring_rewards - mu) / s
    params += lr/(population_size*sigma) * np.dot(N.T, advantage)
    return params, offspring_rewards, (datetime.now() - t0)