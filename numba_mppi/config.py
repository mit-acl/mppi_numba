#!/usr/bin/env python3
import numpy as np

class Config:
  def __init__(self):
    self.T = 5.0
    self.dt = 0.1
    self.num_grid_samples = 1024 # Limited by the threads in a block (<=1024)
    self.num_control_rollouts = 1024 # Limited by blocks in the grid (can be >1024)
    self.num_steps = int(self.T/self.dt)

    # For sampling grids
    self.tdm_sample_thread_dim = (16, 16) # max (32, 32)

    # For visualizing state rollouts
    self.num_vis_state_rollouts = 100