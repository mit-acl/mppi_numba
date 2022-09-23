#!/usr/bin/env python3
"""
Class definition for Model Predictive Path Integral implemented with numba (MPPI_Numba),
where the method is extended to handle terrain-dependent traction parameters for unicycle model.

As numba and GPU code don't understand class definition, we hard-coded the dynamics for unicycle.
The code can be adapted to other dynamical models by modifying the GPU kernel accordingly.
"""

import numpy as np
import math
import copy
import numba
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32


# Stage costs (device function)
@cuda.jit('float32(float32, float32, float32)', device=True, inline=True)
def stage_cost(dist2, dt, dist_weight):
  return dt + dist_weight*math.sqrt(dist2) # squared term makes the robot move faster


# Terminal costs (device function)
@cuda.jit('float32(float32, float32, boolean)', device=True, inline=True)
def term_cost(dist2, v_post_rollout, goal_reached):
  return (1-np.float32(goal_reached))*math.sqrt(dist2)/(v_post_rollout+1e-6)


# To handle unknown and obstacles
DEFAULT_UNKNOWN_COST = np.float(1e2)
DEFAULT_OBS_COST = np.float(1e5)

# Default distance weight in stage cost
DEFAULT_DIST_WEIGHT = 1.0


class MPPI_Numba(object):
  
  """ 
  Planner object that initializes GPU memory and runs MPPI on GPU via numba. 
  
  Typical workflow: 
    1. Initialize object with config that allows pre-initialization of GPU memory
    2. reset()
    3. setup(mppi_params, linear_tdm, angular_tdm) based on problem instance
    4. solve(), which returns optimized control sequence
    5. get_state_rollout() for visualization
    6. shift_and_update(next_state, optimal_u_sequence, num_shifts=1)
    7. Repeat from 2 if traction maps have to be updated.
  """

  def __init__(self, cfg):

    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_grid_samples = cfg.num_grid_samples
    self.num_control_rollouts = cfg.num_control_rollouts
    self.max_speed_padding = cfg.max_speed_padding
    self.tdm_sample_thread_dim = cfg.tdm_sample_thread_dim
    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.max_map_dim = cfg.max_map_dim
    self.seed = cfg.seed
    self.use_tdm = cfg.use_tdm
    self.use_det_dynamics = cfg.use_det_dynamics
    self.use_nom_dynamics_with_speed_map = cfg.use_nom_dynamics_with_speed_map
    self.use_costmap = cfg.use_costmap
    self.det_dyn = self.use_det_dynamics or self.use_nom_dynamics_with_speed_map or self.use_costmap

    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.noise_samples_d = None
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False
    self.reset()

    
  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.params = None
    self.params_set = False

    # Hold reference to the current linear and angular TDM
    self.lin_tdm = None
    self.ang_tdm = None
    self.tdm_set = False

    self.u_prev_d = None
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)
    self.init_device_vars_before_solving()


  def init_device_vars_before_solving(self):

    if not self.device_var_initialized:
      t0 = time.time()
      
      self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, 2), dtype=np.float32) # to be sampled collaboratively via GPU
      self.u_cur_d = cuda.to_device(self.u_seq0) 
      self.u_prev_d = cuda.to_device(self.u_seq0) 
      self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
      
      if not self.det_dyn:  
        self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)
      else:
        # self.state_rollout_batch_d = cuda.device_array((1, self.num_steps+1, 3), dtype=np.float32)
        self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)
      
      self.device_var_initialized = True
      print("MPPI planner has initialized GPU memory after {} s".format(time.time()-t0))


  def setup(self, params, lin_tdm, ang_tdm):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_tdm(lin_tdm, ang_tdm)
    self.set_params(params)


  def is_within_bound(self, v, vbounds):
    return v>=vbounds[0] and v<=vbounds[1]


  def set_params(self, params):
    if not self.is_within_bound(params['x0'][0], self.lin_tdm.xlimits):
      print("ERROR: When setting mppi params, x0[0] is not within xlimits!")
      assert False
    if not self.is_within_bound(params['x0'][1], self.lin_tdm.ylimits):
      print("ERROR: When setting mppi params, x0[1] is not within ylimits!")
      assert False

    self.params = copy.deepcopy(params)
    self.params_set = True


  def set_tdm(self, lin_tdm, ang_tdm):
    self.lin_tdm = lin_tdm
    self.ang_tdm = ang_tdm
    self.tdm_set = True


  def check_solve_conditions(self):
    if not self.params_set:
      print("MPPI parameters are not set. Cannot solve")
      return False
    if not self.tdm_set:
      print("MPPI has not received TDMs. Cannot solve")
      return False
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot solve.")
      return False
    if not self.lin_tdm.pmf_grid_initialized:
      print("Linear TDM's PMF not initialized. Cannot solve.")
      return False
    if not self.ang_tdm.pmf_grid_initialized:
      print("Angular TDM's PMF not initialized. Cannot solve.")
      return False

    # Check initial condition within bound
    if not self.is_within_bound(self.params["x0"][0], self.lin_tdm.padded_xlimits):
      print("Robot initial condition not within padded xlimits.")
      return False
    if not self.is_within_bound(self.params["x0"][1], self.lin_tdm.padded_ylimits):
      print("Robot initial condition not within padded ylimits.")
      return False

    return True


  def solve(self):
    """Entry point for different algoritims"""
    
    if not self.check_solve_conditions():
      print("MPPI solve condition not met. Cannot solve. Return")
      return

    if self.use_det_dynamics: # Use worst-case expected traction (dynamics)
      return self.solve_det_dyn()
    
    elif self.use_nom_dynamics_with_speed_map: # Use nominal traction (dynamics) but adjusts time cost 
      return self.solve_nom_dyn_w_speed_map()
    
    elif self.use_tdm: # Use sampled traction grids
      
      # If number of grid samples fit in a single block, each thread in the block handles 1 traction grid
      if self.cfg.num_grid_samples>self.cfg.max_threads_per_block:
        return self.solve_stochastic_oversized()
      
      # Otherwise, each thread in the block handles multiple traction grid, very slow!
      else:      
        return self.solve_stochastic()

    else:
      print("None of the planner options are selected.")
      assert False


  def move_mppi_task_vars_to_device(self):
    res_d = np.float32(self.lin_tdm.res) # no need to move int
    xlimits_d = cuda.to_device(self.lin_tdm.padded_xlimits.astype(np.float32))
    ylimits_d = cuda.to_device(self.lin_tdm.padded_ylimits.astype(np.float32))
    vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    v_post_rollout_d = np.float32(self.params['v_post_rollout'])
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    cvar_alpha_d = np.float32(self.params['cvar_alpha']) # for computing cvar of objectives
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                     else self.params['obs_penalty'])
    unknown_cost_d = np.float32(DEFAULT_UNKNOWN_COST if 'unknown_penalty' not in self.params 
                                     else self.params['unknown_penalty'])
    return res_d, xlimits_d, ylimits_d, vrange_d, wrange_d, xgoal_d, \
           v_post_rollout_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, cvar_alpha_d, x0_d, dt_d, obs_cost_d, unknown_cost_d


  def solve_nom_dyn_w_speed_map(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    
    Proposed in "Risk-Aware Off-Road Navigation via a Learned Speed Distribution Map" (IROS 2022). 
    This implementation is different by using worst-case speed directly instead of weighted sum of mean and worst-case speed.
    """
    
    res_d, xlimits_d, ylimits_d, vrange_d, wrange_d, xgoal_d, \
           v_post_rollout_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, cvar_alpha_d, x0_d, dt_d, obs_cost_d, unknown_cost_d = self.move_mppi_task_vars_to_device()
  
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids(alpha_dyn=1.0) # TDM is configured s.t. only the last bin corresponding to nominal traction will be sampled.
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids(alpha_dyn=1.0) # TDM is configured s.t. only the last bin corresponding to nominal traction will be sampled.
    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']


    # Optimization loop
    for k in range(self.params['num_opt']):

      # Sample control noise
      self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      
      # Rollout and compute mean or cvar
      self.rollout_det_dyn_w_speed_map_numba[self.num_control_rollouts, 1](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.risk_traction_map_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        self.lin_tdm.obstacle_map_d, # just use values from lin_tdm since ang_tdm produces the same values
        self.lin_tdm.unknown_map_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        obs_cost_d, 
        unknown_cost_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        x0_d,
        dt_d,
        dist_weight,
        self.noise_samples_d,
        self.u_cur_d,

        # results
        self.costs_d
      )
      self.u_prev_d = self.u_cur_d

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )

    return self.u_cur_d.copy_to_host()


  def solve_det_dyn(self):
    """
    Launch GPU kernels that use worst-case expected dynamics.
    
    Proposed in "Probabilistic Traversability Model for Risk-Aware Motion Planning in Off-Road Environments" (submitted to ICRA 2023)
    When cvar_alpha=1.0, this approach is equivalent to "WayFAST: Navigation with Predictive Traversability in the Field" (RA-L 2022).
    """

    res_d, xlimits_d, ylimits_d, vrange_d, wrange_d, xgoal_d, \
           v_post_rollout_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, cvar_alpha_d, x0_d, dt_d, obs_cost_d, unknown_cost_d = self.move_mppi_task_vars_to_device()
    
    # No need to pass alpha_dyn, since dynamics have been initialized to only allow sampling of the value that corresponds to cvar(alpha_dyn)
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids() # TDM is configured s.t. only the bin corresponding to CVaR of traction will be sampled
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids() # TDM is configured s.t. only the bin corresponding to CVaR of traction will be sampled

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']


    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      # Rollout and compute mean or cvar
      self.rollout_det_dyn_numba[self.num_control_rollouts, 1](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        self.lin_tdm.obstacle_map_d,
        self.lin_tdm.unknown_map_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        obs_cost_d, 
        unknown_cost_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        x0_d,
        dt_d,
        dist_weight,
        self.noise_samples_d,
        self.u_cur_d,

        # results
        self.costs_d
      )
      self.u_prev_d = self.u_cur_d
      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )

    # Full control sequence copied from GPU
    return self.u_cur_d.copy_to_host()


  def solve_stochastic(self):
    """
    Launch GPU kernels that use sampled traction grids from TDM.
    
    Proposed in "Probabilistic Traversability Model for Risk-Aware Motion Planning in Off-Road Environments" (submitted to ICRA 2023)
    This approach evaluates each control sequence over M sampled traction maps by using the worst-case expectation.
    """

    res_d, xlimits_d, ylimits_d, vrange_d, wrange_d, xgoal_d, \
           v_post_rollout_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, cvar_alpha_d, x0_d, dt_d, obs_cost_d, unknown_cost_d = self.move_mppi_task_vars_to_device()
    
    
    # Sample environment realizations for estimating cvar
    # The argument alpha_dyn=1.0 means that we sample from the full traction distribution.
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids(
      1.0 if 'alpha_dyn' not in self.params else self.params['alpha_dyn'])
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids(
      1.0 if 'alpha_dyn' not in self.params else self.params['alpha_dyn'])

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)

      # Use trick to dynamically allocate shared array in kernel: # https://stackoverflow.com/questions/30510580/numba-cuda-shared-memory-size-at-runtime
      stream = 0
      shared_array_size = self.num_grid_samples * np.float32().itemsize # how many bytes of shared array size?
      # Rollout and compute mean or cvar
      self.rollout_numba[self.num_control_rollouts, self.num_grid_samples, stream, shared_array_size](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        self.lin_tdm.obstacle_map_d,
        self.lin_tdm.unknown_map_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        obs_cost_d, 
        unknown_cost_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        cvar_alpha_d,
        x0_d,
        dt_d,
        dist_weight,
        self.noise_samples_d,
        self.u_cur_d,
        self.costs_d # results
      )

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )

    # Full control sequence copied from GPU
    return self.u_cur_d.copy_to_host()


  def solve_stochastic_oversized(self):
    """
    Launch GPU kernels that use sampled traction grids from TDM.
    Handle the case where each block handles more than max_threads_per_block. Each thread needs to handle >1 sampled traction maps (slow).

    Proposed in "Probabilistic Traversability Model for Risk-Aware Motion Planning in Off-Road Environments" (submitted to ICRA 2023)
    This approach evaluates each control sequence over M sampled traction maps by using the worst-case expectation.
    """
    assert self.num_grid_samples > self.cfg.max_threads_per_block

    res_d, xlimits_d, ylimits_d, vrange_d, wrange_d, xgoal_d, \
        v_post_rollout_d, goal_tolerance_d, lambda_weight_d, \
        u_std_d, cvar_alpha_d, x0_d, dt_d, obs_cost_d, unknown_cost_d = self.move_mppi_task_vars_to_device()
    
    
    # Sample environment realizations for estimating cvar. 
    # The argument alpha_dyn=1.0 means that we sample from the full traction distribution.
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids(
      1.0 if 'alpha_dyn' not in self.params else self.params['alpha_dyn'])
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids(
      1.0 if 'alpha_dyn' not in self.params else self.params['alpha_dyn'])

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)

      # Use trick to dynamically allocate shared array in kernel: # https://stackoverflow.com/questions/30510580/numba-cuda-shared-memory-size-at-runtime
      stream = 0
      shared_array_size = self.num_grid_samples * np.float32().itemsize # how many bytes of shared array size?
      # Rollout and compute mean or cvar
      # Getting errors: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES 
      # Somehow have to limit the GPU register usage https://stackoverflow.com/questions/68658970/why-launching-a-numba-cuda-kernel-works-with-up-to-640-threads-but-fails-with-6
      self.rollout_oversized_numba[self.num_control_rollouts, self.max_threads_per_block, stream, shared_array_size](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        self.lin_tdm.obstacle_map_d,
        self.lin_tdm.unknown_map_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        obs_cost_d, 
        unknown_cost_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        cvar_alpha_d,
        x0_d,
        dt_d,
        dist_weight,
        self.noise_samples_d,
        self.u_cur_d,
        self.costs_d # results
      )

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )

    # Full control sequence copied from GPU
    return self.u_cur_d.copy_to_host()


  def shift_and_update(self, new_x0, u_cur, num_shifts=1):
    self.params["x0"] = new_x0.copy()
    self.shift_optimal_control_sequence(u_cur, num_shifts)


  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = cuda.to_device(u_cur_shifted.astype(np.float32))


  def get_state_rollout(self):
    """
    Generate state sequences based on the current optimal control sequence.
      - For use_tdm=True, use pre-sampled traction grids.
      - For other methods, when num_vis_state_rollouts=1, the state sequence of the 
        optimal control is visualized. If num_vis_state_rollouts>1, optimal sequence and
        other randomly sampled controls are visualized.
    
    """

    assert self.params_set, "MPPI parameters are not set"
    assert self.tdm_set, "MPPI has not received TDMs"

    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot run mppi.")
      return
    
    # Move things to GPU
    res_d = np.float32(self.lin_tdm.res) # no need to move int
    xlimits_d = cuda.to_device(self.lin_tdm.padded_xlimits.astype(np.float32))
    ylimits_d = cuda.to_device(self.lin_tdm.padded_ylimits.astype(np.float32))
    vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])

    # Use pre-sampled environment realizations for rollouts
    lin_sample_grid_batch_d = self.lin_tdm.sample_grid_batch_d
    ang_sample_grid_batch_d = self.ang_tdm.sample_grid_batch_d

    if self.det_dyn:
      self.get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
            self.state_rollout_batch_d, # where to store results
            lin_sample_grid_batch_d,
            ang_sample_grid_batch_d,
            self.lin_tdm.bin_values_bounds_d,
            self.ang_tdm.bin_values_bounds_d,
            res_d, 
            xlimits_d, 
            ylimits_d, 
            x0_d, 
            dt_d,
            self.noise_samples_d,
            vrange_d,
            wrange_d,
            self.u_prev_d,
            self.u_cur_d,
            )
    else:
      # Get the state rollouts from the optimal control sequence via different sample environments
      self.get_state_rollout_across_envs_numba[1, self.num_vis_state_rollouts](
            self.state_rollout_batch_d, # where to store results
            lin_sample_grid_batch_d,
            ang_sample_grid_batch_d,
            self.lin_tdm.bin_values_bounds_d,
            self.ang_tdm.bin_values_bounds_d,
            res_d, 
            xlimits_d, 
            ylimits_d, 
            x0_d, 
            dt_d,
            self.u_cur_d)
    
    return self.state_rollout_batch_d.copy_to_host()


  """GPU kernels from here on"""

  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          obstacle_map_d,
          unknown_map_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          obs_cost_d, 
          unknown_cost_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          cvar_alpha_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    Every thread in each block considers different traction grids but the same control sequence.
    Each block produces a single result (reduce a shared list to produce CVaR or mean. Is there a more efficient way to do this?)
    """
    # Basic info
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    block_width = cuda.blockDim.x
    timesteps = len(u_cur_d)

    # Create shared array for saving temporary costs
    # Use trick to dynamically allocate shared memory (only can be 1d)
    # https://stackoverflow.com/questions/30510580/numba-cuda-shared-memory-size-at-runtime
    thread_cost_shared = cuda.shared.array(shape=0, dtype=numba.float32)
    # if (thread_cost_shared.shape[0]!=block_width):
    #   print("WARNING: shared array doesn't have the correct size. Return")
    #   return

    thread_cost_shared[tid] = 0.0
    cuda.syncthreads() # Sync before moving on

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    # height, width = lin_sample_grid_batch_d[tid].shape
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3): 
      x_curr[i] = x0_d[i]

    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom =v_noisy = w_nom = w_noisy = 0.0

    lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
    ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])
    
    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      vtraction = lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
      wtraction = ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

      # Nominal noisy control
      v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t,0]
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t,1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_noisy

      dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
      thread_cost_shared[tid] += stage_cost(dist_to_goal2, dt_d, dist_weight_d)
      
      # Separate obstacle and unknown cost
      thread_cost_shared[tid] += obstacle_map_d[yi, xi]*obs_cost_d
      thread_cost_shared[tid] += unknown_map_d[yi, xi]*unknown_cost_d
      

      if dist_to_goal2<= goal_tolerance_d2:
        goal_reached = True
        break

    for t in range(timesteps):
      thread_cost_shared[tid] += lambda_weight_d*(
        (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid,t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid,t, 1])
    
    # Accumulate terminal cost 
    thread_cost_shared[tid] += term_cost(dist_to_goal2, v_post_rollout_d, goal_reached)

    # Reduce thread_cost_shared to a single value (mean or CVaR)
    cuda.syncthreads()  # make sure all threads have produced costs

    numel = block_width
    if cvar_alpha_d<1:
      # numel = math.ceil(block_width*cvar_alpha_d)
      # --- CVaR requires sorting the elements ---
      # First sort the costs from descending order via parallel bubble sort
      # https://stackoverflow.com/questions/42620649/sorting-algorithm-with-cuda-inside-or-outside-kernels
      for i in range(math.ceil(block_width/2)):
        # Odd
        if (tid%2==0) and ((tid+1)!=block_width):
          if thread_cost_shared[tid+1]>thread_cost_shared[tid]:
            # swap
            temp = thread_cost_shared[tid]
            thread_cost_shared[tid] = thread_cost_shared[tid+1]
            thread_cost_shared[tid+1] = temp
        cuda.syncthreads()
        # Even
        if (tid%2==1) and ((tid+1)!=block_width):
          if thread_cost_shared[tid+1]>thread_cost_shared[tid]:
            # swap
            temp = thread_cost_shared[tid]
            thread_cost_shared[tid] = thread_cost_shared[tid+1]
            thread_cost_shared[tid+1] = temp
        cuda.syncthreads()

    # Average reduction based on quantile (all elements for cvar_alpha_d==1)
    # The mean of the first alpha% will be the CVaR
    numel = math.ceil(block_width*cvar_alpha_d)
    s = 1
    while s < numel:
      if (tid % (2 * s) == 0) and ((tid + s) < numel):
        # Stride by `s` and add
        thread_cost_shared[tid] += thread_cost_shared[tid + s]
      s *= 2
      cuda.syncthreads()

    # After the loop, the zeroth  element contains the sum
    if tid == 0:
      costs_d[bid] = thread_cost_shared[0]/numel



  # If memory allocation error try reducing max registers
  @staticmethod
  @cuda.jit(fastmath=True, max_registers=60)
  # @cuda.jit(fastmath=True)
  def rollout_oversized_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          obstacle_map_d,
          unknown_map_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          obs_cost_d, 
          unknown_cost_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          cvar_alpha_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    Every thread in each block considers different traction grids but the same control sequence.
    Each block produces a single result (reduce a shared list to produce CVaR or mean. Is there a more efficient way to do this?)
    """

    # Basic info
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    num_threads = cuda.blockDim.x
    num_grid_samples_d = lin_sample_grid_batch_d.shape[0]
    repeat = numba.int32(math.ceil(num_grid_samples_d/num_threads))

    # Figure out the start and end indices for environmental rollouts
    gap = math.ceil(num_grid_samples_d / num_threads)
    start_grid_i = min(tid*gap, num_grid_samples_d) # Cannot exceed num grids, but this should never happen
    end_grid_i = min(start_grid_i+gap, num_grid_samples_d)

    # Create shared array for saving temporary costs
    # Use trick to dynamically allocate shared memory (only can be 1d)
    # https://stackoverflow.com/questions/30510580/numba-cuda-shared-memory-size-at-runtime
    thread_cost_shared = cuda.shared.array(shape=0, dtype=numba.float32)

    for i in range(start_grid_i, end_grid_i):
      thread_cost_shared[i] = 0.0
    cuda.syncthreads() # Sync before moving on

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    # height, width = lin_sample_grid_batch_d[tid].shape
    for gi in range(start_grid_i, end_grid_i):
      # Reset the initial state
      x_curr = cuda.local.array(3, numba.float32)
      for i in range(3):
        x_curr[i] = x0_d[i]
      
      goal_reached = False
      goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
      dist_to_goal2 = 1e9
      v_nom = v_noisy = w_nom = w_noisy = 0.0

      lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
      ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])
      
      for t in range(len(u_cur_d)):
        # Look up the traction parameters from map
        xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
        yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

        vtraction = lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[gi, yi, xi]
        wtraction = ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[gi, yi, xi]

        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t,0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t,1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
        
        # Forward simulate
        x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*wtraction*w_noisy

        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        thread_cost_shared[gi] += stage_cost(dist_to_goal2, dt_d, dist_weight_d)
        
        # Separate obstacle and unknown cost
        thread_cost_shared[gi] += obstacle_map_d[yi, xi]*obs_cost_d
        thread_cost_shared[gi] += unknown_map_d[yi, xi]*unknown_cost_d
        
        if dist_to_goal2<= goal_tolerance_d2:
          goal_reached = True
          break
      
      
      for t in range(len(u_cur_d)):
        thread_cost_shared[gi] += lambda_weight_d*(
          (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid,t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid,t, 1])
      
      # Accumulate terminal cost 
      thread_cost_shared[gi] += term_cost(dist_to_goal2, v_post_rollout_d, goal_reached)

    # Only need to sync after all grid samples assigned to this thread have been finished.
    # Next we'll reduce thread_cost_shared to a single value (mean or CVaR)
    cuda.syncthreads()  # make sure all threads have produced costs

    # ----------- Sort the costs across threads -----------
    if cvar_alpha_d<1:
      # sort
      for i in range(math.ceil(num_grid_samples_d/2)):
        # Odd
        for ri in range(repeat):
          gi = tid + ri*num_threads
          if (gi%2==0) and ((gi+1)!=num_grid_samples_d):
            # Swap
            temp = thread_cost_shared[gi]
            thread_cost_shared[gi] = thread_cost_shared[gi+1]
            thread_cost_shared[gi+1] = temp
        cuda.syncthreads()

        # Even
        for ri in range(repeat):
          gi = tid + ri*num_threads
          if (gi%2==1) and ((gi+1)!=num_grid_samples_d):
            # Swap
            temp = thread_cost_shared[gi]
            thread_cost_shared[gi] = thread_cost_shared[gi+1]
            thread_cost_shared[gi+1] = temp
        cuda.syncthreads()
    # --------------------------------------------

    # ------------- Mean reduction for alpha quantile -------------
    numel = math.ceil(num_grid_samples_d*cvar_alpha_d)
    s = 1
    while s < numel:
      for ri in range(repeat):
        gi = tid + ri*num_threads
        if (gi % (2 * s) == 0) and ((gi + s) < numel):
          thread_cost_shared[gi] += thread_cost_shared[gi+s]
      s *= 2
      cuda.syncthreads()
    # --------------------------------------------
    
    # After the loop, the zeroth  element contains the sum
    if tid == 0:
      costs_d[bid] = thread_cost_shared[0]/numel


  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_det_dyn_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          obstacle_map_d,
          unknown_map_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          obs_cost_d, 
          unknown_cost_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    The grid sample represents the worst-case expectation of the traction parameters (or mean if alpha_cvar=1).
    """
    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    # height, width = lin_sample_grid_batch_d[tid].shape
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3): 
      x_curr[i] = x0_d[i]

    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom =v_noisy = w_nom = w_noisy = 0.0

    lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
    ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])

    # printed = False
    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      # if not printed and (xi<0 or yi<0):
      #   print("Out of bound")
      #   printed = True

      vtraction = lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
      wtraction = ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

      # Nominal noisy control
      v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_noisy


      dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
      costs_d[bid]+=stage_cost(dist_to_goal2, dt_d, dist_weight_d)

      # Separate obstacle and unknown cost
      costs_d[bid] += obstacle_map_d[yi, xi]*obs_cost_d
      costs_d[bid] += unknown_map_d[yi, xi]*unknown_cost_d

      if dist_to_goal2<= goal_tolerance_d2:
        goal_reached = True
        break
    
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, v_post_rollout_d, goal_reached)

    for t in range(timesteps):
      costs_d[bid] += lambda_weight_d*(
        (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])



  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_det_dyn_w_speed_map_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_risk_traction_map_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          obstacle_map_d,
          unknown_map_d,
          res_d, 
          xlimits_d,
          ylimits_d,
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          obs_cost_d, 
          unknown_cost_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    The grid sample represents the nominal traction parameters (assuming last PMF bin represents traction=1.0), but the time cost is adjusted based on speed.
    """


    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3): 
      x_curr[i] = x0_d[i]

    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom =v_noisy = w_nom = w_noisy = 0.0

    lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
    ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])

    # printed=False
    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      # Still need to look up dynamics (nominal) to make sure states don't out of allocated bounds
      vtraction = lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
      wtraction = ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

      # if not printed and (xi<0 or yi<0):
      #   print("Out of bound")
      #   printed = True

      # Nominal noisy control
      v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_noisy

      # If else statements will be expensive
      dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
      
      effective_speed = lin_bin_values_bounds_d[0]+lin_ratio*lin_risk_traction_map_d[0, yi, xi]
      costs_d[bid]+= stage_cost(dist_to_goal2, dt_d/(effective_speed+1e-6), dist_weight_d)

      # Separate obstacle and unknown cost
      costs_d[bid] += obstacle_map_d[yi, xi]*obs_cost_d
      costs_d[bid] += unknown_map_d[yi, xi]*unknown_cost_d

      if dist_to_goal2<= goal_tolerance_d2:
        goal_reached = True
        break
    
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, v_post_rollout_d, goal_reached)

    for t in range(timesteps):
      costs_d[bid] += lambda_weight_d*(
              (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

  @staticmethod
  @cuda.jit(fastmath=True)
  def update_useq_numba(
        lambda_weight_d,
        costs_d,
        noise_samples_d,
        weights_d,
        vrange_d,
        wrange_d,
        u_cur_d):
    """
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """

    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    numel = len(noise_samples_d)
    gap = int(math.ceil(numel / num_threads))

    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    if starti<numel:
      weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
      weights_d[starti] = min(weights_d[starti], costs_d[i])
    cuda.syncthreads()

    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
      s *= 2
      cuda.syncthreads()

    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
      weights_d[i] = math.exp(-1./lambda_weight_d*(costs_d[i]-beta))
    cuda.syncthreads()

    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
      costs_d[i] = weights_d[i]
    cuda.syncthreads()
    for i in range(starti+1, endi):
      costs_d[starti] += costs_d[i]
    cuda.syncthreads()
    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        costs_d[starti] += costs_d[starti + s]
      s *= 2
      cuda.syncthreads()

    for i in range(starti, endi):
      weights_d[i] /= costs_d[0]
    cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
      for i in range(starti, endi):
        cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*noise_samples_d[i, t, 0])
        cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*noise_samples_d[i, t, 1])
    cuda.syncthreads()

    # Blocks crop the control together
    tgap = int(math.ceil(timesteps / num_threads))
    starti = min(tid*tgap, timesteps)
    endi = min(starti+tgap, timesteps)
    for ti in range(starti, endi):
      u_cur_d[ti, 0] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 0]))
      u_cur_d[ti, 1] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 1]))


  @staticmethod
  @cuda.jit(fastmath=True)
  def get_state_rollout_across_control_noise(
          state_rollout_batch_d, # where to store results
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          x0_d, 
          dt_d,
          noise_samples_d,
          vrange_d,
          wrange_d,
          u_prev_d,
          u_cur_d):
    """
    Do a fixed number of rollouts for visualization across blocks.
    Assume kernel is launched as get_state_rollout_across_control_noise[num_blocks, 1]
    The block with id 0 will always visualize the best control sequence. Other blocks will visualize random samples.
    """
    
    # Use block id
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    timesteps = len(u_cur_d)


    if bid==0:
      # Visualize the current best 
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(3, numba.float32)
      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]

      lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
      ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])
      
      for t in range(timesteps):
        # Look up the traction parameters from map
        xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
        yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

        vtraction = lin_bin_values_bounds_d[0] + lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
        wtraction = ang_bin_values_bounds_d[0] + ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

        # # Nominal noisy control
        v_nom = u_cur_d[t, 0]
        w_nom = u_cur_d[t, 1]
        
        # Forward simulate
        x_curr[0] += dt_d*vtraction*v_nom*math.cos(x_curr[2])
        x_curr[1] += dt_d*vtraction*v_nom*math.sin(x_curr[2])
        x_curr[2] += dt_d*wtraction*w_nom

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]
    else:
      
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(3, numba.float32)
      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]

      lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
      ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])
      
      for t in range(timesteps):
        # Look up the traction parameters from map
        xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
        yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

        vtraction = lin_bin_values_bounds_d[0] + lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
        wtraction = ang_bin_values_bounds_d[0] + ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

        # Nominal noisy control
        v_nom = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))

        # # Nominal noisy control
        v_nom = u_prev_d[t, 0]
        w_nom = u_prev_d[t, 1]
        
        # Forward simulate
        x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*wtraction*w_noisy

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]


  @staticmethod
  @cuda.jit(fastmath=True)
  def get_state_rollout_across_envs_numba(
          state_rollout_batch_d, # where to store results
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          x0_d, 
          dt_d,
          u_cur_d):
    """
    Do a fixed number of rollouts for visualization within one block and many threads.
    Assume kernel is launched as get_state_rollout_across_envs_numba[1, NUM_THREADS].
    The optimal control sequence will be rolled out over multiple sampled traction maps.
    """
    
    tid = cuda.grid(1)
    timesteps = len(u_cur_d)

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3): 
      x_curr[i] = x0_d[i]
      state_rollout_batch_d[tid,0,i] = x0_d[i]

    lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
    ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])
    
    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      vtraction = lin_bin_values_bounds_d[0] + lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
      wtraction = ang_bin_values_bounds_d[0] + ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]

      # # Nominal noisy control
      v_nom = u_cur_d[t, 0]
      w_nom = u_cur_d[t, 1]
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_nom*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_nom*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_nom

      # Save state
      state_rollout_batch_d[tid,t+1,0] = x_curr[0]
      state_rollout_batch_d[tid,t+1,1] = x_curr[1]
      state_rollout_batch_d[tid,t+1,2] = x_curr[2]


  @staticmethod
  @cuda.jit(fastmath=True)
  def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
    """
    Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
    noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    Assume each thread corresponds to one time step
    For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.

    This will not work if time steps are more than max_threads_per_block (usually 1024)
    """
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    abs_thread_id = cuda.grid(1)

    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)

