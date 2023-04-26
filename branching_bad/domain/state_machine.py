import numpy as np
import torch as th


class BatchedCSG2DStateMachine:
    
    def __init__(self, batch_size, n_commands, device):
        self.n_commands = n_commands
        self.max_bools = 1
        self.max_canvas = self.max_bools + 1
        self.bool_count = th.from_numpy(np.zeros(batch_size)).to(device)
        self.canvas_count = th.from_numpy(np.zeros(batch_size)).to(device)
        self.n_bool_ops = 3
    
    def get_state_mask(self):
        # can you draw:
        state_canvas = self.canvas_count < self.bool_count + 1
        state_canvas = th.where(state_canvas, 1, 0)
        # can you add boolean?
        state_bool = th.logical_and(self.bool_count < self.max_bools, state_canvas)
        state_bool = th.where(state_bool, 1, 0)
        state_end = th.logical_and(self.bool_count <= self.max_bools, self.canvas_count == self.bool_count + 1)
        state_end = th.where(state_end, 1, 0)
        # now expand: 
        state_canvas = state_canvas.unsqueeze(1).expand(-1, self.n_commands -4 )
        state_bool = state_bool.unsqueeze(1).expand(-1, self.n_bool_ops)
        state_end = state_end.unsqueeze(1)
        state = th.cat([state_canvas, state_bool, state_end], -1)
        
        return state
    
    def update_state(self, actions_type):
        self.bool_count = th.where(actions_type == 0, self.bool_count + 1, self.bool_count)
        self.canvas_count = th.where(actions_type == 1, self.canvas_count + 1, self.canvas_count)
        
    def update_state_full(self, bool_count, canvas_count):
        self.bool_count = bool_count
        self.canvas_count = canvas_count
    
    def get_bool_and_canvas_count(self):
        return self.bool_count, self.canvas_count
    
class CSG2DStateMachine:
    def __init__(self, n_commands):
        self.n_commands = n_commands
        self.max_bools = 10
        self.max_canvas = self.max_bools + 1
        self.bool_count = 0
        self.canvas_count = 0
        self.action_type_to_count = {0: self.bool_count, 1: self.canvas_count}
        
        
        self.state = (1, 0, 0)
    
    def get_state(self, n_commands):
        state_bool = self.bool_count < self.max_bools
        state_canvas = self.canvas_count < self.bool_count + 1
        state_end = (self.bool_count <= self.max_bools) and (self.canvas_count == self.bool_count + 1)
        state = (state_bool, state_canvas, state_end)
        return state
    
    def update_state(self, action_type):
        self.action_type_to_count[action_type] += 1
    
    def copy(self):
        new_state = CSG2DStateMachine(self.init_batch_size, self.n_commands)
        return new_state