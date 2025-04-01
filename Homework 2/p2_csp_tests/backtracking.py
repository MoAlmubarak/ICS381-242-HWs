from copy import deepcopy

def backtracking(csp):
    """
    Main backtracking function that sets up the initial domains and 
    calls the recursive helper function.
    
    Args:
        csp: The CSP object with variables, domains, and constraints
        
    Returns:
        A complete assignment that satisfies all constraints, or None if no solution exists
    """
    # Initialize the current domains from the CSP's domains
    current_domains = {}
    for var in csp.variables:
        current_domains[var] = deepcopy(csp.domains[var])
    
    # Run AC3 before search to enforce arc consistency
    is_consistent, updated_domains = ac3(csp, arcs_queue=None, 
                                         assignment=None,
                                         current_domains=current_domains)
    
    # If AC3 finds inconsistency, then no solution exists
    if is_consistent:
        return backtracking_helper(csp, assignment={}, current_domains=updated_domains)
    else:
        return None

def backtracking_helper(csp, assignment={}, current_domains=None):
    """
    Recursive helper function for backtracking search.
    
    Args:
        csp: The CSP object
        assignment: Current partial assignment
        current_domains: Current domains of unassigned variables
        
    Returns:
        A complete assignment that satisfies all constraints, or None if no solution exists
    """
    # Base case: check if assignment is complete
    if csp.is_goal(assignment):
        return assignment
    
    # Select variable using Minimum Remaining Values (MRV) heuristic
    var = select_unassigned_variable(csp, assignment, current_domains)
    
    # Order values using Least Constraining Value (LCV) heuristic
    ordered_values = order_domain_values(csp, var, assignment, current_domains)
    
    # Try each value in the ordered domain
    for val in ordered_values:
        # Add {var = val} to assignment
        assignment[var] = val
        
        # Restrict domain of currently assigned variable
        current_domains[var] = [val]
        
        # Check if the assignment is consistent with constraints
        if csp.check_partial_assignment(assignment):
            # Create a deep copy of domains for this branch
            child_domains = deepcopy(current_domains)
            
            # Set up arc consistency queue for neighbors of newly assigned variable
            arcs_queue = [(n, var) for n in csp.adjacency[var] if n not in assignment]
            arcs_queue = set(arcs_queue)
            
            # Run AC3 to enforce arc consistency
            is_consistent, child_domains = ac3(csp, arcs_queue=arcs_queue, 
                                              assignment=assignment,
                                              current_domains=child_domains)
            
            # Continue search if arc consistency is maintained
            if is_consistent:
                result = backtracking_helper(csp, assignment, child_domains)
                
                # Return result if a solution is found
                if result is not None:
                    return result
        
        # If no solution found with this value, remove it from assignment
        assignment.pop(var)
    
    # If no value works, return None (backtrack)
    return None

def select_unassigned_variable(csp, assignment, current_domains):
    """
    Select an unassigned variable using the Minimum Remaining Values (MRV) heuristic.
    Breaks ties using the Degree heuristic (most constraints on other variables).
    
    Args:
        csp: The CSP object
        assignment: Current partial assignment
        current_domains: Current domains of unassigned variables
        
    Returns:
        The variable with the fewest remaining legal values
    """
    # Get all unassigned variables
    unassigned = [var for var in csp.variables if var not in assignment]
    
    if not unassigned:
        return None
    
    # Find the variable with the minimum remaining values (MRV)
    # Break ties with the degree heuristic (most constraints on other variables)
    return min(unassigned, 
              key=lambda var: (len(current_domains[var]), 
                              -sum(1 for n in csp.adjacency[var] if n not in assignment)))

def order_domain_values(csp, var, assignment, current_domains):
    """
    Order the values in the domain using the Least Constraining Value (LCV) heuristic.
    
    Args:
        csp: The CSP object
        var: The variable whose values to order
        assignment: Current partial assignment
        current_domains: Current domains of unassigned variables
        
    Returns:
        List of values ordered by LCV
    """
    if var is None:
        return []
        
    # For each value, count how many values would be eliminated from neighbors' domains
    def count_conflicts(val):
        count = 0
        # Check each neighbor
        for neighbor in csp.adjacency[var]:
            if neighbor not in assignment:  # Only consider unassigned neighbors
                # Check each value in the neighbor's domain
                for n_val in current_domains[neighbor]:
                    # Count if constraint is not satisfied
                    if not csp.constraint_consistent(var, val, neighbor, n_val):
                        count += 1
        return count
    
    # Return values sorted by the number of conflicts they create (least to most)
    return sorted(current_domains[var], key=count_conflicts)

def ac3(csp, arcs_queue=None, assignment=None, current_domains=None):
    """
    AC3 algorithm for enforcing arc consistency.
    
    Args:
        csp: The CSP object
        arcs_queue: Queue of arcs to process (optional)
        assignment: Current partial assignment (optional)
        current_domains: Current domains of variables (optional)
        
    Returns:
        (is_consistent, updated_domains) tuple
    """
    # If no arcs_queue is provided, initialize it with all arcs
    if arcs_queue is None:
        arcs_queue = []
        for var in csp.variables:
            neighbors = csp.adjacency[var]
            for n in neighbors:
                arcs_queue.extend([(var, n), (n, var)])
    arcs_queue = set(arcs_queue)
    
    # If no domains provided, initialize from CSP
    if current_domains is None:
        current_domains = {}
        for var in csp.variables:
            current_domains[var] = deepcopy(csp.domains[var])
            
    # Initialize assignment if not provided
    if assignment is None:
        assignment = {}
    
    # Process each arc in the queue
    while arcs_queue:
        xi, xj = arcs_queue.pop()
        
        # Revise the domain of xi with respect to xj
        if revise(csp, xi, xj, current_domains):
            # If domain of xi is now empty, return False (inconsistent)
            if not current_domains[xi]:
                return False, current_domains
            
            # Add neighbors of xi (except xj) to the queue
            neighbors = [n for n in csp.adjacency[xi] if (n != xj and n not in assignment)]
            for xk in neighbors:
                arcs_queue.add((xk, xi))
    
    # All arcs are consistent
    return True, current_domains

def revise(csp, xi, xj, current_domains):
    """
    Revise the domain of xi with respect to xj.
    Remove values from xi's domain that don't satisfy any constraint with xj.
    
    Args:
        csp: The CSP object
        xi: First variable
        xj: Second variable
        current_domains: Current domains of variables
        
    Returns:
        True if xi's domain was changed, False otherwise
    """
    revised = False
    
    # Check each value in xi's domain
    for valxi in list(current_domains[xi]):  # Use list to avoid modifying during iteration
        # Check if there's a consistent value in xj's domain
        if not any(csp.constraint_consistent(xi, valxi, xj, valxj) 
                  for valxj in current_domains[xj]):
            # No consistent value found, remove valxi from domain
            current_domains[xi].remove(valxi)
            revised = True
    
    return revised