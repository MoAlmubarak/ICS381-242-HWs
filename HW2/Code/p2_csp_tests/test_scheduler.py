import numpy as np  # Add this import at the top of the file
def display_schedule(sol_assignment, schedulercsp):
    print('==================')
    print('Course: Pref-Prof, # Students, Duration, Before-Courses')
    for course_name in schedulercsp.course_info_dict:
        barred_prof_list, student_count, duration, after_courses_list = schedulercsp.course_info_dict[course_name]
        print(f'{course_name}: {barred_prof_list}, {student_count}, {duration}, {after_courses_list}')
    
    print('==================')
    print('Location: Capacity')
    for loc_name in schedulercsp.loc_info_dict:
        capacity = schedulercsp.loc_info_dict[loc_name]
        print(f'{loc_name} {capacity}', end=", ")
    print()
    
    # display timetable schedule
    print('================== TimeTable ==================')
    timetable = np.empty((len(schedulercsp.time_slots), len(schedulercsp.variables)), dtype=object)
    timetable[:] = ''
    cols = []
    for course in sol_assignment:
        prof, loc, start_time = sol_assignment[course]
        end_time = start_time + schedulercsp.course_info_dict[course][2]
        
        course_id = int(course.split('-')[-1])
        prof_id = int(prof.split('-')[-1])
        loc_id = int(prof.split('-')[-1])
        timetable[start_time:end_time,course_id] = f'P{prof_id}L{loc_id}'
        cols.append(f'C{course_id}')
        
    import pandas as pd
    
    timetable_df = pd.DataFrame(data=timetable, index=schedulercsp.time_slots, columns=cols)
    
    print(timetable_df.to_string())
    
    
if __name__ == "__main__":    
    from backtracking import *
    from csp_scheduler import *
    import pickle, time
    
    with open('./scheduler_data_0.pickle', 'rb') as handle:
        courses, professors, loc_info_dict, course_info_dict, time_slots = pickle.load(handle) # read in courses, loc_info_dict, course_info_dict, time_slots
    
    cspscheduler = SchedulerCSP(courses, professors, loc_info_dict, course_info_dict, time_slots)
               
    print(len(cspscheduler.variables))
    print('_______________________________________________________________________')
    print(cspscheduler.variables) 
    print('_______________________________________________________________________')
    print(cspscheduler.time_slots) 
    print('_______________________________________________________________________')
    print(cspscheduler.course_info_dict) 
    print('_______________________________________________________________________')
    print(cspscheduler.loc_info_dict) 
    print('_______________________________________________________________________')
    print(cspscheduler.course_info_dict['Course-5']) 
    print(cspscheduler.domains['Course-5']) 
    print('_______________________________________________________________________')
    
    # solvable
    for i in range(3):
        fname = f'./scheduler_data_{i}.pickle'
        with open(fname, 'rb') as handle:
            courses, professors, loc_info_dict, course_info_dict, time_slots = pickle.load(handle) # read in courses, loc_info_dict, course_info_dict, time_slots
            
        cspscheduler = SchedulerCSP(courses, professors, loc_info_dict, course_info_dict, time_slots)
            
        # run backtracking search to solve
        start_time = time.time()
        sol_assignment = backtracking(cspscheduler)
        end_time = time.time()
        
        is_complete_and_consistent = cspscheduler.is_goal(sol_assignment)
        print('Time taken: {} sec'.format(end_time - start_time))
        display_schedule(sol_assignment, cspscheduler)
        print('_______________________________________________________________________')
        print('_______________________________________________________________________')
    
    # unsolvable
    for i in range(2):
        fname = f'./nosolution_{i}.pickle'
        with open(fname, 'rb') as handle:
            courses, professors, loc_info_dict, course_info_dict, time_slots = pickle.load(handle) # read in the schedulercsp object
            
        cspscheduler = SchedulerCSP(courses, professors, loc_info_dict, course_info_dict, time_slots)
        # run backtracking search to solve
        start_time = time.time()
        sol_assignment = backtracking(cspscheduler)
        end_time = time.time()
        
        is_complete_and_consistent = cspscheduler.is_goal(sol_assignment)
        print('Sol: {}'.format(sol_assignment))
        print('Is sol complete and consistent: {}'.format(is_complete_and_consistent))
        print('Time taken: {} sec'.format(end_time - start_time))
        print('_______________________________________________________________________')
        print('_______________________________________________________________________')