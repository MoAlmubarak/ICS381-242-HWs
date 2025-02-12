import numpy as np
from copy import deepcopy

def objective_f(state, S, T):
    """Returns the objective value f(B) of state.
    
    Args:
        state: List or numpy array with 0/1 values
        S: Numpy 1D array of integers
        T: Target sum integer
    """
    # Construct subset B represented by state using exact provided method
    B = S[[idx for idx in range(len(S)) if state[idx] == 1]]
    return abs(T - np.sum(B))

def get_neighbor(state):
    """Generates one new random neighbor for state.
    
    Args:
        state: List/array of 0/1 values
    """
    # Make a deep copy of state
    n_state = deepcopy(state)
    
    # Case 1: Both on and off bits exist
    if 0 < sum(state) < len(state):
        # Sample u uniformly
        u = np.random.uniform()
        
        if u < 0.5:  # Remove element
            # Get indices that have value 1
            indices = [i for i, x in enumerate(state) if x == 1]
            idx = np.random.choice(indices)
            n_state[idx] = 0
        else:  # Add element
            # Get indices that have value 0
            indices = [i for i, x in enumerate(state) if x == 0]
            idx = np.random.choice(indices)
            n_state[idx] = 1
            
    # Case 2: All bits are on
    elif sum(state) == len(state):
        # Remove element
        indices = [i for i, x in enumerate(state) if x == 1]
        idx = np.random.choice(indices)
        n_state[idx] = 0
        
    # Case 3: All bits are off
    else:
        # Add element
        indices = [i for i, x in enumerate(state) if x == 0]
        idx = np.random.choice(indices)
        n_state[idx] = 1
        
    return n_state

def simulated_annealing(initial_state, S, T, initial_temp=1000):
    """Implementation of simulated annealing for subset sum problem.
    
    Args:
        initial_state: Initial subset selection (list/array of 0/1)
        S: Numpy 1D array of integers
        T: Target sum integer
        initial_temp: Initial temperature (default 1000)
    
    Returns:
        Tuple of (final_state, iterations)
    """
    # 1. Set temperature to initial_temp
    temp = initial_temp
    
    # 2. Set current state to initial_state
    current = initial_state
    
    # 3. Initialize iteration counter
    iters = 0
    
    # 4. Main loop with exact condition
    while temp >= 0:
        # a. The scheduler update
        temp = temp * 0.9999
        
        # b. Check if temperature too low
        if temp < 1e-14:
            return current, iters
            
        # c. Check if goal reached
        if objective_f(current, S, T) == 0:
            return current, iters
            
        # d. Generate random successor
        next_state = get_neighbor(current)
        
        # e. Compute deltaE as specified
        deltaE = objective_f(current, S, T) - objective_f(next_state, S, T)
        
        # f. If improvement, accept it
        if deltaE > 0:
            current = next_state
        # g. If not improvement, maybe accept it with specific probability
        elif deltaE <= 0:
            u = np.random.uniform()
            if u <= np.exp(deltaE/temp):
                current = next_state
            
        # h. Increment iteration counter
        iters += 1
    
    # 5. Return final values
    return current, iters