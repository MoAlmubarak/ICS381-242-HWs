import numpy as np
from copy import deepcopy

class SchedulerCSP:
    def __init__(self, courses, professors, loc_info_dict, course_info_dict, time_slots):
        """
        Initialize the Scheduling CSP.
        
        Args:
            courses: list of course names
            professors: list of professor names
            loc_info_dict: dictionary mapping location names to capacity
            course_info_dict: dictionary mapping course names to tuples of 
                             (preferred_professors, student_count, duration, prereq_courses)
            time_slots: list of available time slots
        """
        self.variables = courses
        self.professors = professors
        self.loc_info_dict = loc_info_dict
        self.course_info_dict = course_info_dict
        self.time_slots = time_slots
        
        self.domains = {}
        self.adjacency = {}
        
        self._initialize_domains()
        self._initialize_adjacency()
    
    def _initialize_domains(self):
        """Initialize the domains for each course variable."""
        for course in self.variables:
            self.domains[course] = []
            
            preferred_profs, num_students, duration, prereqs = self.course_info_dict[course]
            
            # Add all possible combinations of (professor, location, start_time)
            for prof in preferred_profs:
                for loc_name, capacity in self.loc_info_dict.items():
                    # Check if location capacity can handle the student count
                    if capacity >= num_students:
                        # Calculate possible start times (ensuring course doesn't go beyond available time slots)
                        max_start_time = len(self.time_slots) - duration
                        for start_time in range(max_start_time + 1):
                            self.domains[course].append((prof, loc_name, start_time))
    
    def _initialize_adjacency(self):
        """Initialize the adjacency graph of variables based on constraints."""
        for var in self.variables:
            self.adjacency[var] = []
            
            for other_var in self.variables:
                if var != other_var:
                    # Add all other variables as neighbors (all variables constrain each other)
                    if other_var not in self.adjacency[var]:
                        self.adjacency[var].append(other_var)
    
    def constraint_consistent(self, var1, val1, var2, val2):
        """
        Check if the assignment of val1 to var1 and val2 to var2 satisfies all constraints.
        
        Args:
            var1: First variable (course name)
            val1: Value for var1 (professor, location, start_time)
            var2: Second variable (course name)
            val2: Value for var2 (professor, location, start_time)
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        # Unpack values
        prof1, loc1, start_time1 = val1
        prof2, loc2, start_time2 = val2
        
        # Get course durations
        _, _, duration1, prereqs1 = self.course_info_dict[var1]
        _, _, duration2, prereqs2 = self.course_info_dict[var2]
        
        # Calculate end times
        end_time1 = start_time1 + duration1
        end_time2 = start_time2 + duration2
        
        # Check prerequisite constraints
        if var2 in prereqs1:
            # Var2 is a prerequisite of Var1, so Var2 must finish before Var1 starts
            if end_time2 > start_time1:
                return False
        
        if var1 in prereqs2:
            # Var1 is a prerequisite of Var2, so Var1 must finish before Var2 starts
            if end_time1 > start_time2:
                return False
        
        # Check time conflicts - a professor can't teach two courses at the same time
        if prof1 == prof2:
            # Check if the time intervals overlap
            if start_time1 < end_time2 and start_time2 < end_time1:
                return False
        
        # Check location conflicts - a location can't be used for two courses at the same time
        if loc1 == loc2:
            # Check if the time intervals overlap
            if start_time1 < end_time2 and start_time2 < end_time1:
                return False
        
        # If all constraints are satisfied, return True
        return True
    
    def check_partial_assignment(self, assignment):
        """
        Check if the current partial assignment is consistent.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if the assignment is consistent, False otherwise
        """
        # Check each pair of assigned variables
        for var1 in assignment:
            val1 = assignment[var1]
            
            for var2 in assignment:
                if var1 != var2:
                    val2 = assignment[var2]
                    
                    # Check if this pair satisfies all constraints
                    if not self.constraint_consistent(var1, val1, var2, val2):
                        return False
        
        return True
    
    def is_goal(self, assignment):
        """
        Check if the assignment is complete and consistent.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if the assignment is complete and consistent, False otherwise
        """
        # Check if assignment is None (no solution)
        if assignment is None:
            return False
            
        # Check if all variables are assigned
        if set(assignment.keys()) != set(self.variables):
            return False
        
        # Check if the assignment is consistent
        return self.check_partial_assignment(assignment)