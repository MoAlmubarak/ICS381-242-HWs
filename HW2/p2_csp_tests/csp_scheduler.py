import itertools
import copy
import numpy as np

class SchedulerCSP:
    def __init__(self, courses, professors, loc_info_dict, course_info_dict, time_slots):
        # Store the input arguments
        self.loc_info_dict = loc_info_dict
        self.course_info_dict = course_info_dict
        self.time_slots = time_slots
        
        # Set variables to courses
        self.variables = courses
        
        # Initialize domains dictionary
        self.domains = {}
        
        # Create domains for each course
        for course in courses:
            # Get course information
            course_info = course_info_dict[course]
            barred_professors = course_info[0]
            student_count = course_info[1]
            
            # Filter professors (exclude barred ones)
            valid_professors = [prof for prof in professors if prof not in barred_professors]
            
            # Filter locations (ensure capacity >= student count)
            valid_locations = [loc for loc, capacity in loc_info_dict.items() if capacity >= student_count]
            
            # Use time_slots directly for start_time
            valid_time_slots = time_slots
            
            # Create domain as cartesian product of valid professors, locations, and time slots
            self.domains[course] = list(itertools.product(valid_professors, valid_locations, valid_time_slots))
        
        # Create adjacency dictionary (constraint graph)
        self.adjacency = {}
        for course in courses:
            # Every course is connected to all other courses
            self.adjacency[course] = [other_course for other_course in courses if other_course != course]
    
    def constraint_consistent(self, var1, val1, var2, val2):
        """
        Check if the assignment of val1 to var1 and val2 to var2 is consistent with all constraints.
        
        Args:
            var1 (str): First course
            val1 (tuple): Assignment tuple (prof, loc, start_time) for var1
            var2 (str): Second course
            val2 (tuple): Assignment tuple (prof, loc, start_time) for var2
            
        Returns:
            bool: True if the assignment is consistent, False otherwise
        """
        # Extract values from tuples
        prof1, loc1, start_time1 = val1
        prof2, loc2, start_time2 = val2
        
        # Get course durations
        duration1 = self.course_info_dict[var1][2]
        duration2 = self.course_info_dict[var2][2]
        
        # Calculate end times
        end_time1 = start_time1 + duration1
        end_time2 = start_time2 + duration2
        
        # Check1: Determine if times overlap
        # Two time intervals don't overlap if one ends before or at the same time the other starts
        no_overlap = ((start_time1 < start_time2) and (end_time1 <= start_time2)) or ((start_time2 < start_time1) and (end_time2 <= start_time1))
        overlap = not no_overlap
        
        # Resource conflicts check
        if overlap:
            # Check same professor with overlapping time
            if prof1 == prof2:
                return False
            
            # Check same location with overlapping time
            if loc1 == loc2:
                return False
        
        # Check after-list constraints
        after_courses1 = self.course_info_dict[var1][3]
        after_courses2 = self.course_info_dict[var2][3]
        
        # If var2 is in var1's after-list, var2's start_time should be strictly after var1's end_time
        if var2 in after_courses1 and not (start_time2 > end_time1):
            return False
        
        # If var1 is in var2's after-list, var1's start_time should be strictly after var2's end_time
        if var1 in after_courses2 and not (start_time1 > end_time2):
            return False
        
        # All constraints are satisfied
        return True
    
    def check_partial_assignment(self, assignment):
        """
        Check if the partial assignment is consistent.
        
        Args:
            assignment (dict): A dictionary where key is a variable and value is a value
            
        Returns:
            bool: True if the partial assignment is consistent, False otherwise
        """
        # Handle None assignment
        if assignment is None:
            return False
        
        # Check each assigned variable against its assigned neighbors
        for var1 in assignment:
            val1 = assignment[var1]
            
            # Get assigned neighbors
            assigned_neighbors = [var2 for var2 in self.adjacency[var1] if var2 in assignment]
            
            # Check constraints with each assigned neighbor
            for var2 in assigned_neighbors:
                val2 = assignment[var2]
                if not self.constraint_consistent(var1, val1, var2, val2):
                    return False
        
        # All constraints satisfied
        return True
    
    def is_goal(self, assignment):
        """
        Check if the assignment is complete and consistent.
        
        Args:
            assignment (dict): A dictionary where key is a variable and value is a value
            
        Returns:
            bool: True if assignment is complete and consistent, False otherwise
        """
        # Handle None assignment
        if assignment is None:
            return False
        
        # Check if assignment is consistent
        if not self.check_partial_assignment(assignment):
            return False
        
        # Check if assignment is complete (all variables are assigned)
        return len(assignment) == len(self.variables)
