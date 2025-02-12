# helper function to generate random subset sum problem instance
def gen_rand_ssp(N=8):
    # sample N random integers 
    S = np.random.randint(low=1, high=1e4, size=N, dtype=int)
    
    # sample K indices from S to get T
    K = np.random.randint(low=1, high=N, dtype=int)
    subset_indices = np.random.choice(N, size=K, replace=False)
    
    B_subset = S[subset_indices]
    T = B_subset.sum()
    
    return S, T, B_subset
    
def sample_random_subset_state(N=8):
    K = np.random.randint(low=1, high=N, dtype=int)
    subset_indices = np.random.choice(N, size=K, replace=False)
    
    rnd_subset_state = np.zeros(N)
    rnd_subset_state[subset_indices] = 1
    
    return rnd_subset_state
    
if __name__ == "__main__":
    import numpy as np
    from local_search import *
    
    # set random seed for reproducibility
    rand_seed=735122311
    np.random.seed(rand_seed)
    
    # tests 1
    N = 20
    S, T, B_subset = gen_rand_ssp(N=N)
    state1 = sample_random_subset_state(N)
    f_cost = objective_f(state1, S, T)
    
    print('SSP state: {}'.format(state1))
    print('f(s) = {}'.format(f_cost))
    print('_______________________________________________________________________')
    
    state2 = sample_random_subset_state(N)
    f_cost = objective_f(state2, S, T)
    print('TSP state: {}'.format(state2))
    print('f(s) = {}'.format(f_cost))
    print('_______________________________________________________________________')
    
    # tests run simulated annealing
    np.random.seed(rand_seed)
    N = 20
    S, T, B_subset = gen_rand_ssp(N=N)
    initial_state = sample_random_subset_state(N)
    initial_fcost = objective_f(initial_state, S, T)
    initial_temp = 30000
    final_state, iters = simulated_annealing(initial_state, S, T, initial_temp=initial_temp)
    final_fcost = objective_f(final_state, S, T)
    final_Bsubset = S[[idx for idx in range(len(S)) if final_state[idx] == 1]]
    
    print('S = {},\nB_subset = {},\nT = {}\n'.format(S, B_subset, T))
    print('Initial state objective value: {}'.format(initial_fcost))
    print('Final state objective value: {}.\nFinal subset: {}.'.format(final_fcost, final_Bsubset))
    print('# iterations: {}'.format(iters)) 
    print('_______________________________________________________________________')
    
    np.random.seed(rand_seed)
    N = 40
    S, T, B_subset = gen_rand_ssp(N=N)
    initial_state = sample_random_subset_state(N)
    initial_fcost = objective_f(initial_state, S, T)
    initial_temp = 1000000
    final_state, iters = simulated_annealing(initial_state, S, T, initial_temp=initial_temp)
    final_fcost = objective_f(final_state, S, T)
    final_Bsubset = S[[idx for idx in range(len(S)) if final_state[idx] == 1]]
    
    print('S = {},\nB_subset = {},\nT = {}\n'.format(S, B_subset, T))
    print('Initial state objective value: {}'.format(initial_fcost))
    print('Final state objective value: {}.\nFinal subset: {}.'.format(final_fcost, final_Bsubset))
    print('# iterations: {}'.format(iters)) 
    print('_______________________________________________________________________')