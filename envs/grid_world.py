import numpy as np
class SimpleGridWorld:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.num_state = grid_size * grid_size
        self.num_action = 4
        
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.trap_pos = (1, 2)
        
        self.agent_pos = list(self.start_pos)
        
    def reset(self):
        self.agent_pos = list(self.start_pos)
        return self._get_state()
    
    def _get_state(self):
        y, x = self.agent_pos
        return y * self.grid_size + x
    
    def step(self, action):
        y, x = self.agent_pos

        if action == 0 and y > 0: # up
            y -= 1
        elif action == 1 and y < self.grid_size - 1: # down
            y += 1
        elif action == 2 and x > 0: # left
            x -= 1
        elif action == 3 and x < self.grid_size - 1: # right
            x += 1
        
        self.agent_pos = [y, x]
        next_state = self._get_state()
        curr_posn = tuple(self.agent_pos)
        
        # rewards
        if curr_posn == self.goal_pos:
            return next_state, 10, True
        elif curr_posn == self.trap_pos:
            return next_state, -10, True
        else:
            return next_state, -1, False