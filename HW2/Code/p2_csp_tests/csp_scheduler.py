from collections import defaultdict

class SchedulerCSP:
    """
    CSP formulation of the course scheduling problem.
    
    Variables:
    - Courses
    
    Domains:
    - Each course can be assigned (professor, location, start_time)
    
    Constraints:
    - Course prerequisites must be respected
    - No professor can teach multiple courses at the same time
    - No location can host multiple courses at the same time
    - Location capacity must be sufficient for the course
    - Courses must be taught by available professors
    - Courses must fit within the given time slots based on duration
    """
    
    def __init__(self, courses, professors, loc_info_dict, course_info_dict, time_slots):
        """
        Initialize the SchedulerCSP with course, professor, location, and time information.
        
        Args:
            courses: List of course names
            professors: List of professor names
            loc_info_dict: Dictionary of location name -> capacity
            course_info_dict: Dictionary of course name -> [preferred_profs, student_count, duration, prerequisites]
            time_slots: List of available time slots
        """
        self.variables = courses
        self.professors = professors
        self.loc_info_dict = loc_info_dict
        self.course_info_dict = course_info_dict
        self.time_slots = time_slots
        
        # Create a mapping of courses that must come before other courses
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
        
        Domain for a course is a list of (professor, location, start_time) tuples
        that satisfy the basic constraints:
        - Professor must be preferred for the course
        - Location capacity must be sufficient for the course
        - Course must fit within time slots
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
        Create an adjacency list for the constraint graph.
        
        Two courses are adjacent if:
        - One is a prerequisite for the other
        - They could potentially share the same professor or location at the same time
        """
        adjacency = {var: [] for var in self.variables}
        
        # Add edges for all pairs of different courses
        for i, course1 in enumerate(self.variables):
            for j, course2 in enumerate(self.variables[i+1:], i+1):
                # Courses are adjacent if they have constraints between them
                adjacency[course1].append(course2)
                adjacency[course2].append(course1)
        
        return adjacency
    
    def is_goal(self, assignment):
        """
        Check if the assignment is complete and consistent.
        
        Args:
            assignment: Dictionary mapping variables (courses) to values (prof, loc, time)
            
        Returns:
            True if assignment is complete and consistent, False otherwise
        """
        # Check if assignment is complete (all variables assigned)
        if len(assignment) != len(self.variables):
            return False
        
        # Check if assignment is consistent with all constraints
        return self.check_partial_assignment(assignment)
    
    def check_partial_assignment(self, assignment):
        """
        Check if a partial assignment is consistent with all constraints.
        
        Args:
            assignment: Dictionary mapping variables (courses) to values (prof, loc, time)
            
        Returns:
            True if consistent, False otherwise
        """
        # Check all binary constraints between assigned variables
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
            
        # Check prerequisite constraints
        if var2 in self.course_info_dict[var1][3]:  # var2 is a prerequisite for var1
            if end_time2 > start_time1:
                return False
                
        if var1 in self.course_info_dict[var2][3]:  # var1 is a prerequisite for var2
            if end_time1 > start_time2:
                return False
        
        return True
