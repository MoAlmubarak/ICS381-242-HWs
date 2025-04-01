from copy import deepcopy
 
def backtracking(csp):
    # domain dictionary of all variables 
    # we getting this info from the csp class
    # we create a deep copy to avoid modifying csp.domains when we pass by ref
    current_domains = {}
    for var in csp.variables:
        current_domains[var] = deepcopy(csp.domains[var])
    
    # call ac3 before searching
    is_consistent, updated_domains = ac3(csp, arcs_queue=None, 
                                         assignment=None,
                                         current_domains=current_domains, )
    if is_consistent:
        return backtracking_helper(csp, assignment={}, current_domains=updated_domains)
    else:
        return None

def backtracking_helper(csp, assignment={}, current_domains=None):
    # base-case
    if csp.is_goal(assignment):
        return assignment
        
    var = var_selection_function(csp, assignment, current_domains)             # example use mrv
    val_order = val_order_function(csp, var, assignment, current_domains)      # example use lcv
    
    # loop on each child search node
    for val in val_order:
        assignment[var] = val
        current_domains[var] = [val] # restrict domain of what we currently assigned
        
        if csp.check_partial_assignment(assignment):
            # create deep copy of current_domains for each child path 
            # since ac3 potentially modifies it and sometimes we may need to backtrack and revert to older current_domains. 
            # not efficient memory-wise but easier to code than undo-stack
            child_domains = deepcopy(current_domains)
            # setup queue with just the arcs of unassigned neighbors of var
            arcs_queue = [(n, var) for n in csp.adjacency[var] if n not in assignment]
            arcs_queue = set(arcs_queue)
            is_consistent, child_domains = ac3(csp, arcs_queue=arcs_queue, 
                                               assignment=assignment,
                                               current_domains=child_domains)
            
            # if ac3 finds no issues, then continue down this search path
            if is_consistent:
                result = backtracking_helper(csp, assignment=assignment, current_domains=child_domains)
                
                # if this downstream path is returning a solution, 
                # return pass it to parent
                if result is not None:
                    return result
                    
        assignment.pop(var, None) # technically this line is not needed with this code
    
    # if no value for current variable is viable, 
    # then we return None indicating no-solution down this path
    return None

# default ordering; returns variables in same order as given
# def var_selection_function(csp, assignment, current_domains):
#     return [var for var in csp.variables if var not in assignment][0]

# default ordering; returns values in same order as given
# def val_order_function(csp, var, assignment, current_domains):
#     return [val for val in current_domains[var]]
    
def ac3(csp, arcs_queue=None, assignment=None, current_domains=None):
    # if no arcs_queue is passed,
    # setup a full arcs queue based on csp.adjacency
    if arcs_queue is None:
        arcs_queue = []
        for var in csp.variables:
            neighbors = csp.adjacency[var]
            for n in neighbors:
                arcs_queue.extend([(var, n), (n, var)])
    arcs_queue = set(arcs_queue)
    
    # no domains passed, 
    # set it up from csp.domains
    if current_domains is None:
        current_domains = {}
        for var in csp.variables:
            current_domains[var] = deepcopy(csp.domains[var])
    if assignment is None:
        assignment = {}
    
    # ac3 starts looping
    while len(arcs_queue) > 0:
        xi, xj = arcs_queue.pop()
        if revise(csp, xi, xj, current_domains):
            if len(current_domains[xi]) == 0:
                return False, current_domains

            neighbors = [n for n in csp.adjacency[xi] if (n != xj and n not in assignment)]
            for xk in neighbors:
                arcs_queue.add((xk, xi))
                
    return True, current_domains
    
def revise(csp, xi, xj, current_domains):
    revised = False
    for valxi in deepcopy(current_domains[xi]): # why a deepcopy is needed?
        satisfiable = False
        for valxj in current_domains[xj]:
            if csp.constraint_consistent(xi, valxi, xj, valxj):
                satisfiable = True
                break
        if not satisfiable:
            current_domains[xi].remove(valxi)
            revised = True
            
    return revised

# Minimum Remaining Values (MRV) heuristic implementation
def mrv(csp, assignment, current_domains):
    """
    Minimum Remaining Values (MRV) heuristic selects the variable with the fewest 
    remaining values in its domain.
    
    Args:
        csp: The constraint satisfaction problem
        assignment: Current partial assignment
        current_domains: Current domains of variables
    
    Returns:
        Variable with the minimum remaining values
    """
    unassigned_vars = [var for var in csp.variables if var not in assignment]
    
    # Return the variable with the minimum remaining values in its domain
    return min(unassigned_vars, key=lambda var: len(current_domains[var]))

# Least Constraining Value (LCV) heuristic implementation
def lcv(csp, var, assignment, current_domains):
    """
    Least Constraining Value (LCV) heuristic orders values by the number of values 
    they eliminate in neighboring variables. Values that eliminate the fewest options 
    for neighboring variables are tried first.
    
    Args:
        csp: The constraint satisfaction problem
        var: The variable to assign
        assignment: Current partial assignment
        current_domains: Current domains of variables
    
    Returns:
        List of values ordered by the LCV heuristic
    """
    # Get neighboring variables that are not yet assigned
    neighbors = [n for n in csp.adjacency[var] if n not in assignment]
    
    # For each value in the domain of var, count how many values it would eliminate
    # from neighboring variables
    def count_conflicts(val):
        conflicts = 0
        for neighbor in neighbors:
            for neighbor_val in current_domains[neighbor]:
                if not csp.constraint_consistent(var, val, neighbor, neighbor_val):
                    conflicts += 1
        return conflicts
    
    # Return values ordered by the number of conflicts they create (least first)
    return sorted(current_domains[var], key=count_conflicts)

# Update the variable selection and value ordering functions in backtracking
def var_selection_function(csp, assignment, current_domains):
    """
    Uses the MRV (Minimum Remaining Values) heuristic to select the next variable.
    """
    return mrv(csp, assignment, current_domains)

def val_order_function(csp, var, assignment, current_domains):
    """
    Uses the LCV (Least Constraining Value) heuristic to order the values.
    """
    return lcv(csp, var, assignment, current_domains)