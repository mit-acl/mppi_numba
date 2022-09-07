#!/usr/bin/env python3
from numba import cuda
gpu = cuda.get_current_device()
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
rec_max_control_rollouts = 3000 # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100

class Config:
  
  # Configurations that are typically fixed throughout execution
  
  def __init__(self, 
               T=10, # Horizon (s)
               dt=0.1, # Length of each step (s)
               num_grid_samples=1024, # Number of grid samples when sampling dynamics
               num_control_rollouts=1024, # Number of control sequences
               max_speed_padding=5.0, # Maximum assumed speed for padding the perimeter of grid
               tdm_sample_thread_dim=(16, 16), # Block shape used for sampling each grid map. Max (32, 32)
               num_vis_state_rollouts=20, # Number of visualization rollouts
               max_map_dim=(250, 250), # Largest dim for incoming maps with padding (in cells). Anything bigger will be cropped.
               seed=1,
               use_tdm=False,
               use_det_dynamics=False,
               use_nom_dynamics_with_speed_map=False,
               use_costmap=False, # only applicable when interfacing with costmap2d in ROS
               ):
    
    self.seed = seed
    self.use_tdm = use_tdm
    self.use_det_dynamics = use_det_dynamics
    self.use_nom_dynamics_with_speed_map = use_nom_dynamics_with_speed_map
    self.use_costmap = use_costmap
    num_true = sum([use_tdm, use_det_dynamics, use_nom_dynamics_with_speed_map, use_costmap])

    assert T > 0
    assert dt > 0
    assert T > dt
    assert not (num_true==0 or num_true>1), "MPPI Config Error: Only one of the use_tdm, use_det_dynamics, use_nom_dynamics_with_speed_map, use_costmap can be true."
    # if num_true==0 or num_true>1:
    #   print("MPPI Config: Only one of the use_tdm, use_det_dynamics, use_nom_dynamics_with_speed_map, use_costmap can be true. Use default use_tdm=true")
    #   self.use_tdm = True
    #   self.use_det_dynamics = False
    #   self.use_nom_dynamics_with_speed_map = False
    #   self.use_costmap = False

    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    assert self.num_steps > 0
    
    # Currently limited by the threads in a block (<=1024)
    self.num_grid_samples = num_grid_samples
    if self.num_grid_samples > max_threads_per_block:
      self.num_grid_samples = min([max_threads_per_block, self.num_grid_samples])
      print("MPPI Config: Currently limited by the threads in a block (<={})".format(max_threads_per_block))
    elif self.num_grid_samples < 1:
      self.num_grid_samples = 1
      print("MPPI Config: Set num_grid_samples from {} -> 1. Need at least 1 map to work with".format(num_grid_samples))
    
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > rec_max_control_rollouts:
      self.num_control_rollouts = rec_max_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        rec_max_control_rollouts, max_blocks))
    elif self.num_control_rollouts < rec_min_control_rollouts:
      self.num_control_rollouts = rec_min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        rec_min_control_rollouts, rec_max_control_rollouts))
    
    self.max_speed_padding = max_speed_padding

    self.tdm_sample_thread_dim = tdm_sample_thread_dim
    assert len(self.tdm_sample_thread_dim) == 2
    assert self.tdm_sample_thread_dim[0] > 0
    assert self.tdm_sample_thread_dim[1] > 0
    requested_num_threads_tdm = self.tdm_sample_thread_dim[0]*self.tdm_sample_thread_dim[1]
    if requested_num_threads_tdm >= max_threads_per_block:
       self.tdm_sample_thread_dim = max_square_block_dim
       print("MPPI Config: Requested {} threads per block (more than max {}) for sampling tdm. Change tdm_sample_thread_dim to {}".format(
        requested_num_threads_tdm, max_threads_per_block, max_square_block_dim))

    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, 
                                      self.num_control_rollouts,
                                      self.num_grid_samples])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])

    self.max_map_dim = max_map_dim
    