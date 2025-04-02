from copy import deepcopy
from typing import Dict, List, Tuple, Optional


class SchedulerCSP:
    """A constraint satisfaction problem solver for course scheduling."""
    
    def __init__(self, courses: List[str], professors: List[str], 
                 loc_info_dict: Dict[str, int], course_info_dict: Dict[str, Tuple], 
                 time_slots: List[int]):
        """
        Initialize the Scheduling CSP.
        
        Args:
            courses: List of course names
            professors: List of professor names
            loc_info_dict: Dictionary mapping location names to capacity
            course_info_dict: Dictionary mapping course names to tuples of 
                             (preferred_professors, student_count, duration, prereq_courses)
            time_slots: List of available time slots
        """
        self.variables = courses
        self.professors = professors
        self.loc_info_dict = loc_info_dict
        self.course_info_dict = course_info_dict
        self.time_slots = time_slots
        
        self.domains: Dict[str, List[Tuple]] = {}
        self.adjacency: Dict[str, List[str]] = {}
        
        self._initialize_domains()
        self._initialize_adjacency()
    
    def _initialize_domains(self) -> None:
        """Initialize the domains for each course variable with valid assignments."""
        for course in self.variables:
            self.domains[course] = []
            
            preferred_profs, num_students, duration, _ = self.course_info_dict[course]
            
            # Find suitable locations based on capacity
            suitable_locations = [loc for loc, capacity in self.loc_info_dict.items() 
                                if capacity >= num_students]
            
            max_start_time = len(self.time_slots) - duration
            
            # Add all possible combinations of (professor, location, start_time)
            for prof in preferred_profs:
                for loc_name in suitable_locations:
                    for start_time in range(max_start_time + 1):
                        self.domains[course].append((prof, loc_name, start_time))
    
    def _initialize_adjacency(self) -> None:
        """Initialize the adjacency graph of variables based on constraints."""
        for var in self.variables:
            self.adjacency[var] = [other_var for other_var in self.variables if var != other_var]
    
    def constraint_consistent(self, var1: str, val1: Tuple, var2: str, val2: Tuple) -> bool:
        """
        Check if the assignment of val1 to var1 and val2 to var2 satisfies all constraints.
        
        Args:
            var1: First variable (course)
            val1: Value for first variable (prof, location, start_time)
            var2: Second variable (course)
            val2: Value for second variable (prof, location, start_time)
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        prof1, loc1, start_time1 = val1
        prof2, loc2, start_time2 = val2
        
        _, _, duration1, prereqs1 = self.course_info_dict[var1]
        _, _, duration2, prereqs2 = self.course_info_dict[var2]
        
        end_time1 = start_time1 + duration1
        end_time2 = start_time2 + duration2
        
        # Check prerequisite constraints
        if var2 in prereqs1 and end_time2 > start_time1:
            return False
        if var1 in prereqs2 and end_time1 > start_time2:
            return False
        
        # Check time conflicts for professors
        if prof1 == prof2:
            if not (end_time1 <= start_time2 or end_time2 <= start_time1):
                return False
        
        # Check time conflicts for locations
        if loc1 == loc2:
            if not (end_time1 <= start_time2 or end_time2 <= start_time1):
                return False
        
        return True
    
    def check_partial_assignment(self, assignment: Dict[str, Tuple]) -> bool:
        """
        Check if the current partial assignment is consistent.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if the assignment is consistent, False otherwise
        """
        # Convert to items once for efficiency
        items = list(assignment.items())
        n = len(items)
        
        for i in range(n):
            var1, val1 = items[i]
            for j in range(i + 1, n):
                var2, val2 = items[j]
                if not self.constraint_consistent(var1, val1, var2, val2):
                    return False
        return True
    
    def is_goal(self, assignment: Optional[Dict[str, Tuple]]) -> bool:
        """
        Check if the assignment is complete and consistent.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if assignment is complete and consistent, False otherwise
        """
        return (
            assignment is not None and
            set(assignment.keys()) == set(self.variables) and
            self.check_partial_assignment(assignment)
        )


def backtracking(csp: SchedulerCSP) -> Optional[Dict[str, Tuple]]:
    """
    Main backtracking function with AC3 preprocessing.
    
    Args:
        csp: The SchedulerCSP instance
        
    Returns:
        Complete assignment dictionary if solution exists, None otherwise
    """
    # Pre-process with AC3
    current_domains = deepcopy(csp.domains)
    is_consistent, updated_domains = ac3(csp, current_domains=current_domains)
    
    if is_consistent:
        return backtracking_helper(csp, {}, updated_domains)
    return None


def backtracking_helper(csp: SchedulerCSP, assignment: Dict[str, Tuple], 
                        current_domains: Dict[str, List[Tuple]]) -> Optional[Dict[str, Tuple]]:
    """
    Recursive helper function for backtracking search with forward checking and constraint propagation.
    
    Args:
        csp: The SchedulerCSP instance
        assignment: Current partial assignment
        current_domains: Current domains for each variable
        
    Returns:
        Complete assignment dictionary if solution exists, None otherwise
    """
    if csp.is_goal(assignment):
        return assignment
    
    var = select_unassigned_variable(csp, assignment, current_domains)
    if var is None:
        return None
    
    for val in order_domain_values(csp, var, assignment, current_domains):
        assignment[var] = val
        if csp.check_partial_assignment(assignment):
            # Forward checking and constraint propagation
            child_domains = deepcopy(current_domains)
            child_domains[var] = [val]  # Assign the chosen value
            
            # Set up arcs for AC3
            arcs_queue = [(neighbor, var) for neighbor in csp.adjacency[var] 
                          if neighbor not in assignment]
            
            # Run constraint propagation
            is_consistent, updated_domains = ac3(csp, arcs_queue, child_domains)
            
            if is_consistent:
                result = backtracking_helper(csp, assignment, updated_domains)
                if result is not None:
                    return result
        
        del assignment[var]  # More efficient than pop()
    
    return None

def select_unassigned_variable(csp: SchedulerCSP, assignment: Dict[str, Tuple], 
                              current_domains: Dict[str, List[Tuple]]) -> Optional[str]:
    """
    Select an unassigned variable using the Minimum Remaining Values (MRV) 
    and degree heuristic as a tie-breaker.
    
    Args:
        csp: The SchedulerCSP instance
        assignment: Current partial assignment
        current_domains: Current domains for each variable
        
    Returns:
        Selected variable or None if all are assigned
    """
    unassigned = [var for var in csp.variables if var not in assignment]
    if not unassigned:
        return None
    
    # Sort by domain size (MRV) and break ties with degree heuristic
    return min(unassigned, key=lambda var: (len(current_domains[var]), 
                                           -sum(1 for n in csp.adjacency[var] 
                                               if n not in assignment)))


def order_domain_values(csp: SchedulerCSP, var: str, assignment: Dict[str, Tuple], 
                       current_domains: Dict[str, List[Tuple]]) -> List[Tuple]:
    """
    Order the values in the domain using the Least Constraining Value (LCV) heuristic.
    
    Args:
        csp: The SchedulerCSP instance
        var: Variable to assign
        assignment: Current partial assignment
        current_domains: Current domains for each variable
        
    Returns:
        Ordered list of domain values
    """
    def count_conflicts(val: Tuple) -> int:
        conflicts = 0
        for neighbor in csp.adjacency[var]:
            if neighbor in assignment:
                continue
            for n_val in current_domains[neighbor]:
                if not csp.constraint_consistent(var, val, neighbor, n_val):
                    conflicts += 1
        return conflicts
    
    return sorted(current_domains[var], key=count_conflicts)

def ac3(csp: SchedulerCSP, arcs_queue=None, 
        current_domains: Optional[Dict[str, List[Tuple]]] = None) -> Tuple[bool, Dict[str, List[Tuple]]]:
    """
    AC3 algorithm for enforcing arc consistency.
    
    Args:
        csp: The SchedulerCSP instance
        arcs_queue: Queue of arcs to process, or None to process all
        current_domains: Current domains for each variable
        
    Returns:
        Tuple of (is_consistent, updated_domains)
    """
    if current_domains is None:
        current_domains = deepcopy(csp.domains)
    
    if arcs_queue is None:
        # Initialize queue with all arcs
        arcs_queue = list((var, neighbor) for var in csp.variables 
                          for neighbor in csp.adjacency[var])
    else:
        # Ensure we have a list for operations
        arcs_queue = list(arcs_queue)
        
    while arcs_queue:
        xi, xj = arcs_queue.pop(0)
        if revise(csp, xi, xj, current_domains):
            if not current_domains[xi]:
                return False, current_domains
            
            # Add affected arcs back to the queue
            for xk in csp.adjacency[xi]:
                if xk != xj:
                    arcs_queue.append((xk, xi))
    
    return True, current_domains
    return True, current_domains


def revise(csp: SchedulerCSP, xi: str, xj: str, 
           current_domains: Dict[str, List[Tuple]]) -> bool:
    """
    Revise the domain of xi with respect to xj.
    
    Args:
        csp: The SchedulerCSP instance
        xi: First variable
        xj: Second variable
        current_domains: Current domains for each variable
        
    Returns:
        True if domain of xi was revised, False otherwise
    """
    revised = False
    xi_domain = current_domains[xi]
    xj_domain = current_domains[xj]
    
    # Optimize by creating a list for removal rather than modifying during iteration
    to_remove = []
    for valxi in xi_domain:
        # Check if there's at least one compatible value in xj's domain
        if not any(csp.constraint_consistent(xi, valxi, xj, valxj) for valxj in xj_domain):
            to_remove.append(valxi)
            revised = True
    
    # Remove incompatible values
    for val in to_remove:
        xi_domain.remove(val)
        
    return revised
