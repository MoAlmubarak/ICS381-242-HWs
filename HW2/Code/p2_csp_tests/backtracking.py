from copy import deepcopy

def backtracking(csp):
    """
    Backtracking search algorithm for CSP.
    """
    # domain dictionary of all variables 
    # we getting this info from the csp class
    # we create a deep copy to avoid modifying csp.domains when we pass by ref
    current_domains = {}
    for var in csp.variables:
        current_domains[var] = deepcopy(csp.domains[var])
    
    # call ac3 before searching
    is_consistent, updated_domains = ac3(csp, arcs_queue=None, 
                                         assignment=None,
                                         current_domains=current_domains)
    if is_consistent:
        return backtracking_helper(csp, assignment={}, current_domains=updated_domains)
    else:
        return None

def backtracking_helper(csp, assignment={}, current_domains=None):
    """
    Helper function for backtracking search.
    """
    # base-case: if assignment is complete, return it
    if len(assignment) == len(csp.variables):
        return assignment
        
    # Select unassigned variable using MRV heuristic
    var = var_selection_function(csp, assignment, current_domains)
    
    # Order domain values using LCV heuristic
    val_order = val_order_function(csp, var, assignment, current_domains)
    
    # loop on each child search node
    for val in val_order:
        # Assign the value to the variable
        assignment[var] = val
        # Restrict domain of the assigned variable
        old_domain = current_domains[var]
        current_domains[var] = [val]
        
        # Check if assignment is consistent with constraints
        if csp.check_partial_assignment(assignment):
            # Create deep copy of current_domains for each child path 
            child_domains = deepcopy(current_domains)
            
            # Setup queue with arcs of unassigned neighbors of var
            arcs_queue = [(n, var) for n in csp.adjacency[var] if n not in assignment]
            arcs_queue = set(arcs_queue)
            
            # Apply AC3 to enforce arc consistency
            is_consistent, child_domains = ac3(csp, arcs_queue=arcs_queue, 
                                               assignment=assignment,
                                               current_domains=child_domains)
            
            # If AC3 finds no issues, continue down this search path
            if is_consistent:
                result = backtracking_helper(csp, assignment=assignment, current_domains=child_domains)
                
                # If solution found, return it
                if result is not None:
                    return result
        
        # If we reach here, this value didn't work, remove it from assignment
        assignment.pop(var)
        current_domains[var] = old_domain
    
    # If no value works, return None (no solution in this branch)
    return None

def var_selection_function(csp, assignment, current_domains):
    """
    Uses the MRV (Minimum Remaining Values) heuristic to select the next variable.
    """
    unassigned_vars = [var for var in csp.variables if var not in assignment]
    return min(unassigned_vars, key=lambda var: len(current_domains[var]))

def val_order_function(csp, var, assignment, current_domains):
    """
    Uses the LCV (Least Constraining Value) heuristic to order the values.
    """
    # Get unassigned neighbors
    neighbors = [n for n in csp.adjacency[var] if n not in assignment]
    
    # Count conflicts for each value
    def count_conflicts(val):
        conflicts = 0
        for neighbor in neighbors:
            for neighbor_val in current_domains[neighbor]:
                if not csp.constraint_consistent(var, val, neighbor, neighbor_val):
                    conflicts += 1
        return conflicts
    
    # Return values ordered by the number of conflicts (least first)
    return sorted(current_domains[var], key=count_conflicts)
    
def ac3(csp, arcs_queue=None, assignment=None, current_domains=None):
    """
    AC3 algorithm for enforcing arc consistency.
    """
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
        
        # Skip if xi is already assigned
        if xi in assignment:
            continue
            
        if revise(csp, xi, xj, current_domains):
            if len(current_domains[xi]) == 0:
                return False, current_domains

            neighbors = [n for n in csp.adjacency[xi] if (n != xj and n not in assignment)]
            for xk in neighbors:
                arcs_queue.add((xk, xi))
                
    return True, current_domains
    
def revise(csp, xi, xj, current_domains):
    """
    Revises domain of xi with respect to xj.
    """
    revised = False
    for valxi in deepcopy(current_domains[xi]):  # deepcopy needed to avoid modifying during iteration
        satisfiable = False
        for valxj in current_domains[xj]:
            if csp.constraint_consistent(xi, valxi, xj, valxj):
                satisfiable = True
                break
        if not satisfiable:
            current_domains[xi].remove(valxi)
            revised = True
            
    return revised