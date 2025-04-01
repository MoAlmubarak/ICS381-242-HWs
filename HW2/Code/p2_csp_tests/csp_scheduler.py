from collections import defaultdict

class SchedulerCSP:
    """
    CSP formulation of the course scheduling problem.
    
    Variables: Courses
    Domains: (professor, location, start_time) tuples
    Constraints:
    - Course prerequisites must be respected
    - No professor can teach multiple courses at the same time
    - No location can host multiple courses at the same time
    - Location capacity must be sufficient for the course
    """
    
    def __init__(self, courses, professors, loc_info_dict, course_info_dict, time_slots):
        """
        Initialize the SchedulerCSP.
        
        Args:
            courses: List of course names
            professors: List of professor names
            loc_info_dict: Dictionary mapping location names to capacities
            course_info_dict: Dictionary mapping course names to [preferred_profs, student_count, duration, prerequisites]
            time_slots: List of available time slots
        """
        self.variables = courses
        self.professors = professors
        self.loc_info_dict = loc_info_dict
        self.course_info_dict = course_info_dict
        self.time_slots = time_slots
        
        # Create mapping of courses that must come before others
        self.before_courses = defaultdict(list)
        for course in course_info_dict:
            for prereq in course_info_dict[course][3]:  # prerequisites are at index 3
                self.before_courses[prereq].append(course)
        
        # Create domains for each variable (course)
        self.domains = self._create_domains()
        
        # Create adjacency list for constraint graph
        self.adjacency = self._create_adjacency()
    
    def _create_domains(self):
        """
        Create domains for each course.
        
        A domain value is a tuple (professor, location, start_time) that satisfies:
        - Professor is in the preferred list for the course
        - Location has enough capacity for the course
        - Course fits within available time slots
        """
        domains = {}
        
        for course in self.variables:
            domain = []
            preferred_profs, student_count, duration, _ = self.course_info_dict[course]
            
            # Consider all allowed combinations
            for prof in preferred_profs:
                for loc_name, capacity in self.loc_info_dict.items():
                    # Check if location has enough capacity
                    if capacity >= student_count:
                        # Check all possible start times
                        for t_idx in range(len(self.time_slots) - duration + 1):
                            # Course must fit within available time slots
                            domain.append((prof, loc_name, t_idx))
            
            domains[course] = domain
        
        return domains
    
    def _create_adjacency(self):
        """
        Create adjacency list for the constraint graph.
        All variables are connected to all other variables.
        """
        adjacency = {var: [] for var in self.variables}
        
        # Connect each variable to all other variables
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 != var2:
                    adjacency[var1].append(var2)
        
        return adjacency
    
    def is_goal(self, assignment):
        """
        Check if assignment is complete and consistent.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if assignment is complete and consistent, False otherwise
        """
        # Handle None assignment
        if assignment is None:
            return False
        
        # Check if assignment is complete
        if len(assignment) != len(self.variables):
            return False
        
        # Check if assignment is consistent
        return self.check_partial_assignment(assignment)
    
    def check_partial_assignment(self, assignment):
        """
        Check if a partial assignment is consistent with all constraints.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            True if consistent, False otherwise
        """
        # Handle None assignment
        if assignment is None:
            return False
            
        # Check all pairs of assigned variables
        for course1 in assignment:
            prof1, loc1, start_time1 = assignment[course1]
            duration1 = self.course_info_dict[course1][2]
            end_time1 = start_time1 + duration1
            
            # Check prerequisite constraints
            prereq_list = self.course_info_dict[course1][3]
            for prereq in prereq_list:
                if prereq in assignment:
                    # Prerequisite course must end before this course starts
                    _, _, start_time_prereq = assignment[prereq]
                    duration_prereq = self.course_info_dict[prereq][2]
                    end_time_prereq = start_time_prereq + duration_prereq
                    
                    if end_time_prereq > start_time1:
                        return False
            
            # Check for conflicts with other assigned courses
            for course2 in assignment:
                if course1 != course2:
                    prof2, loc2, start_time2 = assignment[course2]
                    duration2 = self.course_info_dict[course2][2]
                    end_time2 = start_time2 + duration2
                    
                    # Check for time overlaps
                    time_overlap = (start_time1 < end_time2) and (start_time2 < end_time1)
                    
                    # No professor can teach two courses at the same time
                    if time_overlap and prof1 == prof2:
                        return False
                    
                    # No location can host two courses at the same time
                    if time_overlap and loc1 == loc2:
                        return False
        
        return True
    
    def constraint_consistent(self, var1, val1, var2, val2):
        """
        Check if a binary constraint between two variables is satisfied.
        
        Args:
            var1: First variable (course)
            val1: Value for var1 (prof, loc, time)
            var2: Second variable (course)
            val2: Value for var2 (prof, loc, time)
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        prof1, loc1, start_time1 = val1
        prof2, loc2, start_time2 = val2
        
        # Get durations
        duration1 = self.course_info_dict[var1][2]
        duration2 = self.course_info_dict[var2][2]
        
        # Calculate end times
        end_time1 = start_time1 + duration1
        end_time2 = start_time2 + duration2
        
        # Check for time overlaps
        time_overlap = (start_time1 < end_time2) and (start_time2 < end_time1)
        
        # No professor can teach two courses at the same time
        if time_overlap and prof1 == prof2:
            return False
        
        # No location can host two courses at the same time
        if time_overlap and loc1 == loc2:
            return False
            
        # Check prerequisite constraints in both directions
        if var2 in self.course_info_dict[var1][3]:  # var2 is a prerequisite for var1
            if end_time2 > start_time1:
                return False
                
        if var1 in self.course_info_dict[var2][3]:  # var1 is a prerequisite for var2
            if end_time1 > start_time2:
                return False
        
        return True