import numpy as np
from copy import deepcopy

class GridHunterProblem:
    def __init__(self, initial_agent_info, N, monster_coords):
        self.N = N # Grid size
        self.monster_coords = monster_coords # List of (row, col) tuples for monster positions
        # Create initial state with all monsters alive (False = alive)
        monster_states = tuple(False for _ in monster_coords)
        self.initial_state = initial_agent_info + (0,) + monster_states

    def move_monsters(self, timestep):
        new_positions = [] # New monster positions
        # Move monsters based on timestep
        for row, col in self.monster_coords:
            if timestep % 4 == 0 or timestep % 4 == 2:  # Initial position
                new_positions.append((row, col))
            elif timestep % 4 == 1:  # Move left
                new_positions.append((row, max(0, col - 1)))
            elif timestep % 4 == 3:  # Move right
                new_positions.append((row, min(self.N - 1, col + 1)))
        return new_positions

    def actions(self, state):
        row, col, forw, health = state[:4] # Agent info
        
        # Check if agent is dead
        if health <= 0:
            return []
            
        # All actions
        all_actions = ['move-forward', 'turn-left', 'turn-right', 'shoot-arrow', 'stay']
        
        # Check if move-forward would go out of bounds
        direction_deltas = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }
        # Calculate the change in row and column based on the current direction
        delta_row, delta_col = direction_deltas[forw]
        # Determine the next position if the agent moves forward
        next_row, next_col = row + delta_row, col + delta_col
            
        # Remove move-forward if it would go out of bounds
        if next_row < 0 or next_row >= self.N or next_col < 0 or next_col >= self.N:
            all_actions.remove('move-forward')
            
        return all_actions

    def result(self, state, action):
        row, col, forw, health, mstep = state[:5]
        monster_states = list(state[5:])
        
        # Update timestep and monster positions
        new_mstep = (mstep + 1) % 4
        new_monster_positions = self.move_monsters(new_mstep)
        
        # Handle different actions
        new_row, new_col, new_forw = row, col, forw
        
        if action == 'move-forward':
            if forw == 'north':
                new_row -= 1
            elif forw == 'south':
                new_row += 1
            elif forw == 'east':
                new_col += 1
            elif forw == 'west':
                new_col -= 1
        elif action == 'turn-left':
            new_forw = {
            'north': 'west',
            'west': 'south',
            'south': 'east',
            'east': 'north'
            }[forw]
        elif action == 'turn-right':
            new_forw = {
            'north': 'east',
            'east': 'south',
            'south': 'west',
            'west': 'north'
            }[forw]
        elif action == 'shoot-arrow':
            # Get arrow path based on direction
            arrow_row, arrow_col = new_row, new_col
            while True:
                if forw == 'north':
                    arrow_row -= 1
                elif forw == 'south':
                    arrow_row += 1
                elif forw == 'east':
                    arrow_col += 1
                elif forw == 'west':
                    arrow_col -= 1
                    
                # Check if arrow hit wall
                if arrow_row < 0 or arrow_row >= self.N or arrow_col < 0 or arrow_col >= self.N:
                    break
                    
                # Check if arrow hits any monsters
                for i, (m_row, m_col) in enumerate(new_monster_positions):
                    if arrow_row == m_row and arrow_col == m_col and not monster_states[i]:
                        monster_states[i] = True  # Monster dies
                        
        # Check for damage from alive monsters
        new_health = health
        for i, (m_row, m_col) in enumerate(new_monster_positions):
            if not monster_states[i]:  # If monster is alive
                if new_row == m_row and new_col == m_col:
                    new_health -= 1
                    
        return (new_row, new_col, new_forw, new_health, new_mstep) + tuple(monster_states)

    def action_cost(self, state1, action, state2):
        return 1 # All actions have cost 1

    def is_goal(self, state):
        health = state[3] # Agent health
        monster_states = state[5:] # Monster states
        return all(monster_states) and health > 0 # All monsters dead and agent alive

    def h(self, node):
        if self.is_goal(node.state): # If goal state reached
            return 0 # Heuristic is 0
            
        row = node.state[0] # Agent row
        monster_positions = self.move_monsters(node.state[4]) # Monster positions at next timestep
        monster_states = node.state[5:] # Monster states (alive or dead)
        
        # Find minimum row distance to any alive monster
        min_distance = float('inf')
        for i, (m_row, _) in enumerate(monster_positions):
            if not monster_states[i]:  # If monster is alive
                distance = abs(row - m_row) # Manhattan distance to monster row position
                min_distance = min(min_distance, distance) # Update minimum distance
                
        return min_distance