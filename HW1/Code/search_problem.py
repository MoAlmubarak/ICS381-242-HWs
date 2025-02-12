import numpy as np
from copy import deepcopy

class GridHunterProblem:
    def __init__(self, initial_agent_info, N, monster_coords):
        """Initialize the GridHunterProblem.
        
        Args:
            initial_agent_info: Tuple (row, col, forw, health) for agent's starting position
            N: Size of the grid (N x N)
            monster_coords: List of (row, col) tuples for monster starting positions
        """
        self.N = N
        self.monster_coords = monster_coords
        # Create initial state with all monsters alive (False = alive)
        monster_states = tuple(False for _ in monster_coords)
        self.initial_state = initial_agent_info + (0,) + monster_states

    def move_monsters(self, timestep):
        """Return new monster positions based on timestep.
        
        Args:
            timestep: Integer representing monster movement phase
        Returns:
            List of (row, col) tuples for new monster positions
        """
        new_positions = []
        for row, col in self.monster_coords:
            if timestep % 4 == 0:  # Initial position
                new_positions.append((row, col))
            elif timestep % 4 == 1:  # Move left
                new_positions.append((row, max(0, col - 1)))
            elif timestep % 4 == 2:  # Back to initial
                new_positions.append((row, col))
            else:  # Move right
                new_positions.append((row, min(self.N - 1, col + 1)))
        return new_positions

    def actions(self, state):
        """Return list of valid actions from state.
        
        Args:
            state: Current state tuple (row, col, forw, health, mstep, *monster_states)
        Returns:
            List of valid action strings
        """
        row, col, forw, health, mstep = state[:5]
        
        # Check if agent is dead
        if health <= 0:
            return []
            
        # All possible actions
        # Return actions in specific order as required
        all_actions = ['move-forward', 'turn-left', 'turn-right', 'shoot-arrow', 'stay']
        
        # Check if move-forward would go out of bounds
        next_row, next_col = row, col
        if forw == 'north':
            next_row -= 1
        elif forw == 'south':
            next_row += 1
        elif forw == 'east':
            next_col += 1
        elif forw == 'west':
            next_col -= 1
            
        # Remove move-forward if it would go out of bounds
        if next_row < 0 or next_row >= self.N or next_col < 0 or next_col >= self.N:
            all_actions.remove('move-forward')
            
        return all_actions

    def result(self, state, action):
        """Return the state that results from taking action in state.
        
        Args:
            state: Current state tuple
            action: Action string
        Returns:
            Resulting state tuple
        """
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
            if forw == 'north':
                new_forw = 'west'
            elif forw == 'west':
                new_forw = 'south'
            elif forw == 'south':
                new_forw = 'east'
            elif forw == 'east':
                new_forw = 'north'
        elif action == 'turn-right':
            if forw == 'north':
                new_forw = 'east'
            elif forw == 'east':
                new_forw = 'south'
            elif forw == 'south':
                new_forw = 'west'
            elif forw == 'west':
                new_forw = 'north'
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
        """Return cost of taking action from state1 to state2."""
        return 1

    def is_goal(self, state):
        """Return True if state is a goal state."""
        health = state[3]
        monster_states = state[5:]
        return all(monster_states) and health > 0

    def h(self, node):
        """Return heuristic estimate of distance to goal.
        
        Args:
            node: Node object containing state information
        Returns:
            Heuristic value (minimum row distance to any alive monster)
        """
        if self.is_goal(node.state):
            return 0
            
        row = node.state[0]
        monster_positions = self.move_monsters(node.state[4])
        monster_states = node.state[5:]
        
        # Find minimum row distance to any alive monster
        min_distance = float('inf')
        for i, (m_row, _) in enumerate(monster_positions):
            if not monster_states[i]:  # If monster is alive
                distance = abs(row - m_row)
                min_distance = min(min_distance, distance)
                
        return min_distance