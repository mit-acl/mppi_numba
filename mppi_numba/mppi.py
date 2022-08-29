#!/usr/bin/env python3
import numpy as np
import math
import copy
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


# TODO: import the control configurations from a config file?
from .config import Config
cfg = Config()
NUM_GRID_SAMPLES = cfg.num_grid_samples # Has to be a constant when used in kernel (cannot be class property)
NUM_STEPS = cfg.num_steps
NUM_VIS_STATE_ROLLOUTS = cfg.num_vis_state_rollouts
# Sanity checks
assert cfg.num_steps <= 1024, "NUM_STEPS={} current cannot be more than 1024".format(cfg.num_steps)


class MPPI_Numba(object):

  # NUM_CONTROL_ROLLOUTS = cfg.num_control_rollouts
  # NUM_STEPS = cfg.num_steps
  # NUM_GRID_SAMPLES = cfg.num_grid_samples


  def __init__(self):
    self.params = None
    self.params_set = False

    # Hold reference to the current linear and angular TDM
    self.lin_tdm = None
    self.ang_tdm = None
    self.tdm_set = False

    # Initialize device variables
    self.noise_samples_d = None
    self.u_cur_d = None
    self.costs_d = None
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient
    self.det_dyn = False
    self.device_var_initialized = False


  def is_within_bound(self, v, vbounds):
    return v>=vbounds[0] and v<=vbounds[1]

  def setup(self, params, lin_tdm, ang_tdm):
    self.set_tdm(lin_tdm, ang_tdm)
    self.set_params(params)

  def set_params(self, params):
    # Check conditions
    assert self.tdm_set, "TDM must be set before MPPI parameters"
    assert self.is_within_bound(params['x0'][0], self.lin_tdm.xlimits), "x0[0] is not within xlimits."
    assert self.is_within_bound(params['x0'][1], self.lin_tdm.ylimits), "x0[1] is not within ylimits."

    # Options to run variants of MPPI
    assert not (params['use_det_dynamics'] and params['use_nom_dynamics_with_speed_map']) , \
      "Cannot set both 'use_det_dynamics' and 'use_nom_dynamics_with_speed_map' to True."

    self.params = copy.deepcopy(params)
    self.det_dyn = self.params['use_det_dynamics'] or self.params['use_nom_dynamics_with_speed_map']
    self.params_set = True

  def update_params(self, sub_params):
    # Update a subset of the parameters (e.g., x0, xgoal)
    for key, val in sub_params.items():
      if key in self.params:
        self.params[key] = val

  def set_tdm(self, lin_tdm, ang_tdm):
    self.lin_tdm = lin_tdm
    self.ang_tdm = ang_tdm
    self.tdm_set = True


  def init_device_vars_before_solving(self):
    # noise, sol, 
    if not self.params_set:
      assert False, "Params not set. Cannot initialize GPU memory for noise samples and current control sequence."

    if not self.device_var_initialized:
      self.noise_samples_d = cuda.device_array((cfg.num_control_rollouts, cfg.num_steps, 2), dtype=np.float32) # to be sampled collaboratively via GPU
      self.costs_d = cuda.device_array((cfg.num_control_rollouts), dtype=np.float32)
      self.weights_d = cuda.device_array((cfg.num_control_rollouts), dtype=np.float32)
      self.rng_states_d = create_xoroshiro128p_states([cfg.num_control_rollouts*cfg.num_steps], seed=1)
      self.u_cur_d = cuda.to_device(np.asarray(self.params['u_seq0']).astype(np.float32)) # likely reused
      if not self.det_dyn:  
        self.state_rollout_batch_d = cuda.device_array((cfg.num_vis_state_rollouts, cfg.num_steps+1, 3), dtype=np.float32)
      else:
        self.state_rollout_batch_d = cuda.device_array((1, cfg.num_steps+1, 3), dtype=np.float32)
      
      self.device_var_initialized = True


  def solve(self):
    if self.params['use_det_dynamics']:
      return self.solve_det_dyn()
    elif self.params['use_nom_dynamics_with_speed_map']:
      return self.solve_nom_dyn_w_speed_map()
    else:
      return self.solve_stochastic()


  def solve_nom_dyn_w_speed_map(self):
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
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    v_post_rollout_d = np.float32(self.params['v_post_rollout'])
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    cvar_alpha_d = np.float32(self.params['cvar_alpha'])
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    
    
    # Sample environment realizations for estimating cvar
    self.lin_tdm.init_device_vars_before_sampling(det_dyn=True)
    self.ang_tdm.init_device_vars_before_sampling(det_dyn=True)
    # lin_sample_grid_batch_d = self.lin_tdm.sample_grids(det_dyn=True) # get ref to device samples
    # ang_sample_grid_batch_d = self.ang_tdm.sample_grids(det_dyn=True) # get ref to device samples

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[cfg.num_control_rollouts, cfg.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      
      # Rollout and compute mean or cvar
      self.rollout_det_dyn_w_speed_map_numba[cfg.num_control_rollouts, 1](
        # lin_sample_grid_batch_d,
        # ang_sample_grid_batch_d,
        self.lin_tdm.risk_traction_map_d,
        # self.ang_tdm.risk_traction_map_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        # cvar_alpha_d,
        x0_d,
        dt_d,
        self.noise_samples_d,
        self.u_cur_d,
        # results
        self.costs_d
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


  def solve_det_dyn(self):
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
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    v_post_rollout_d = np.float32(self.params['v_post_rollout'])
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    cvar_alpha_d = np.float32(self.params['cvar_alpha'])
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    
    
    # Sample environment realizations for estimating cvar
    self.lin_tdm.init_device_vars_before_sampling(det_dyn=True)
    self.ang_tdm.init_device_vars_before_sampling(det_dyn=True)
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids(det_dyn=True) # get ref to device samples
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids(det_dyn=True) # get ref to device samples

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[cfg.num_control_rollouts, cfg.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      
      # Rollout and compute mean or cvar
      self.rollout_det_dyn_numba[cfg.num_control_rollouts, 1](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        # cvar_alpha_d,
        x0_d,
        dt_d,
        self.noise_samples_d,
        self.u_cur_d,
        # results
        self.costs_d
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


  def solve_stochastic(self):
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
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    v_post_rollout_d = np.float32(self.params['v_post_rollout'])
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    cvar_alpha_d = np.float32(self.params['cvar_alpha'])
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    
    
    # Sample environment realizations for estimating cvar
    self.lin_tdm.init_device_vars_before_sampling()
    self.ang_tdm.init_device_vars_before_sampling()
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids() # get ref to device samples
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids() # get ref to device samples

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      self.sample_noise_numba[cfg.num_control_rollouts, cfg.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      
      # Rollout and compute mean or cvar
      self.rollout_numba[cfg.num_control_rollouts, cfg.num_grid_samples](
        lin_sample_grid_batch_d,
        ang_sample_grid_batch_d,
        self.lin_tdm.bin_values_bounds_d,
        self.ang_tdm.bin_values_bounds_d,
        res_d,
        xlimits_d,
        ylimits_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        v_post_rollout_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        cvar_alpha_d,
        x0_d,
        dt_d,
        self.noise_samples_d,
        self.u_cur_d,
        # results
        self.costs_d
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
    # shift the optimal control sequence via
    self.shift_optimal_control_sequence(u_cur, num_shifts)

  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    # shift the optimal control sequence
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = cuda.to_device(u_cur_shifted.astype(np.float32))

  def get_state_rollout(self):

    assert self.params_set, "MPPI parameters are not set"
    assert self.tdm_set, "MPPI has not received TDMs"

    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot run mppi.")
      return
    
    # Move things to GPU
    res_d = np.float32(self.lin_tdm.res) # no need to move int
    xlimits_d = cuda.to_device(self.lin_tdm.padded_xlimits.astype(np.float32))
    ylimits_d = cuda.to_device(self.lin_tdm.padded_ylimits.astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])

    # Sample environment realizations for estimating cvar
    # TODO: can we save computation by not resampling for visualization?
    self.lin_tdm.init_device_vars_before_sampling(det_dyn=self.det_dyn)
    self.ang_tdm.init_device_vars_before_sampling(det_dyn=self.det_dyn)
    lin_sample_grid_batch_d = self.lin_tdm.sample_grids(det_dyn=self.det_dyn) # get ref to device samples
    ang_sample_grid_batch_d = self.ang_tdm.sample_grids(det_dyn=self.det_dyn) # get ref to device samples
    if self.det_dyn:
      self.get_state_rollout_numba[1, 1](
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
    else:
      self.get_state_rollout_numba[1, NUM_VIS_STATE_ROLLOUTS](
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



  """GPU kernels"""

  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          cvar_alpha_d, 
          x0_d, 
          dt_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    Every thread in each block considers different traction grids but the same control sequence.
    Each block produces a single result (reduce a shared list to produce CVaR or mean. Is there a more efficient way to do this?)
    """
    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block

    # Create shared array for saving temporary costs
    block_width = cuda.blockDim.x
    thread_cost_shared = cuda.shared.array(NUM_GRID_SAMPLES, dtype=numba.float32)
    thread_cost_shared[tid] = 0.0

    # Move control sequence to shared array as well
    timesteps = len(u_cur_d)
    useq_shared = cuda.shared.array((NUM_STEPS, 2), dtype=numba.float32) # can only initialize using constants..
    noise_shared = cuda.shared.array((NUM_STEPS, 2), dtype=numba.float32) # can only initialize using constants..
    
    # Since each thread uses the same sequence, construct in shared memory
    num = math.ceil(timesteps/block_width)
    tstart = min(tid*num, timesteps)
    tend = min(tstart+num, timesteps)
    for i in range(tstart, tend):
      useq_shared[i,0] = u_cur_d[i,0]
      useq_shared[i,1] = u_cur_d[i,1]
      noise_shared[i,0] = noise_samples_d[bid, i,0]
      noise_shared[i,1] = noise_samples_d[bid, i,1]
    
    # Sync before moving on
    cuda.syncthreads()

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    height, width = lin_sample_grid_batch_d[tid].shape
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3): 
      x_curr[i] = x0_d[i]

    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom =v_noisy = w_nom = w_noisy = 0.0

    lin_ratio = 0.01*(lin_bin_values_bounds_d[1]-lin_bin_values_bounds_d[0])
    ang_ratio = 0.01*(ang_bin_values_bounds_d[1]-ang_bin_values_bounds_d[0])

    # PI2 = numba.float32(math.pi*2.0)
    
    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      # vtraction = lin_bin_values_bounds_d[0] + lin_ratio*lin_sample_grid_batch_d[tid, yi, xi]
      # wtraction = ang_bin_values_bounds_d[0] + ang_ratio*ang_sample_grid_batch_d[tid, yi, xi]
      vtraction = lin_bin_values_bounds_d[0] + (lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[tid, yi, xi])
      wtraction = ang_bin_values_bounds_d[0] + (ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[tid, yi, xi])

      # Nominal noisy control
      v_nom = useq_shared[t, 0] + noise_shared[t, 0]
      w_nom = useq_shared[t, 1] + noise_shared[t, 1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_noisy

      # Clip angle values within [0, 2pi] (Hmm don't think is needed)
      # x_curr[2] = math.fmod(math.fmod(x_curr[2], PI2)+PI2, PI2)

      # Accumulate cost starting at the initial state
      thread_cost_shared[tid]+=dt_d
      if not goal_reached:
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        if dist_to_goal2<= goal_tolerance_d2:
          goal_reached = True
          break
      
    # Accumulate terminal cost 
    if not goal_reached:
      thread_cost_shared[tid] += math.sqrt(dist_to_goal2)/v_post_rollout_d

    # Accumulate the missing stage cost
    for t in range(timesteps):
      thread_cost_shared[tid] += lambda_weight_d*(
              (useq_shared[t,0]/(u_std_d[0]**2))*noise_shared[t,0] + (useq_shared[t,1]/(u_std_d[1]**2))*noise_shared[t, 1])

    # Reudce thread_cost_shared to a single value (mean or CVaR)
    cuda.syncthreads()  # make sure all threads have produced costs

    numel = block_width
    if cvar_alpha_d<1:
      numel = math.ceil(block_width*cvar_alpha_d)
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


  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_det_dyn_numba(
          lin_sample_grid_batch_d,
          ang_sample_grid_batch_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          res_d, 
          xlimits_d, 
          ylimits_d, 
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          # cvar_alpha_d, # not used
          x0_d, 
          dt_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    Every thread in each block considers different traction grids but the same control sequence.
    Each block produces a single result (reduce a shared list to produce CVaR or mean. Is there a more efficient way to do this?)
    """
    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    height, width = lin_sample_grid_batch_d[tid].shape
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

    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      vtraction = lin_bin_values_bounds_d[0] + (lin_bin_values_bounds_d[0]+lin_ratio*lin_sample_grid_batch_d[tid, yi, xi])
      wtraction = ang_bin_values_bounds_d[0] + (ang_bin_values_bounds_d[0]+ang_ratio*ang_sample_grid_batch_d[tid, yi, xi])

      # Nominal noisy control
      v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*vtraction*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*vtraction*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*wtraction*w_noisy

      # Accumulate cost starting at the initial state
      costs_d[bid]+=dt_d
      if not goal_reached:
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        if dist_to_goal2<= goal_tolerance_d2:
          goal_reached = True
          break
      
    # Accumulate terminal cost 
    if not goal_reached:
      costs_d[bid] += math.sqrt(dist_to_goal2)/v_post_rollout_d

    # Accumulate the missing stage cost
    for t in range(timesteps):
      costs_d[bid] += lambda_weight_d*(
              (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])


  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_det_dyn_w_speed_map_numba(
          #lin_sample_grid_batch_d,
          #ang_sample_grid_batch_d,
          lin_risk_traction_map_d,
          #ang_risk_traction_map_d,
          lin_bin_values_bounds_d,
          ang_bin_values_bounds_d,
          res_d, 
          xlimits_d,
          ylimits_d,
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          v_post_rollout_d, 
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          # cvar_alpha_d, # not used
          x0_d, 
          dt_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    Every thread in each block considers different traction grids but the same control sequence.
    Each block produces a single result (reduce a shared list to produce CVaR or mean. Is there a more efficient way to do this?)
    """
    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    # height, width = lin_sample_grid_batch_d[tid].shape
    height, width = lin_risk_traction_map_d[0].shape
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

    for t in range(timesteps):
      # Look up the traction parameters from map
      xi = numba.int32((x_curr[0]-xlimits_d[0])//res_d)
      yi = numba.int32((x_curr[1]-ylimits_d[0])//res_d)

      # Nominal noisy control
      v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      
      # Forward simulate
      x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*w_noisy

      # Accumulate (risk-speed adjusted) cost starting at the initial state
      vtraction = lin_bin_values_bounds_d[0] + (lin_bin_values_bounds_d[0]+lin_ratio*lin_risk_traction_map_d[0, yi, xi])
      costs_d[bid]+=(dt_d/(vtraction+1e-3)) # avoid div by 0

      if not goal_reached:
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        if dist_to_goal2<= goal_tolerance_d2:
          goal_reached = True
          break
      
    # Accumulate terminal cost 
    if not goal_reached:
      costs_d[bid] += math.sqrt(dist_to_goal2)/v_post_rollout_d

    # Accumulate the missing stage cost
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
    # Assume the function is invoked as update_useq_numba[1, NUM_THREADS]
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
  def get_state_rollout_numba(
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
    Assume kernel is launched as get_state_rollout_numba[1, NUM_THREADS]
    """
    tid = cuda.grid(1)
    block_width = cuda.blockDim.x

    # Move control sequence to shared array as well
    timesteps = len(u_cur_d)
    useq_shared = cuda.shared.array((NUM_STEPS, 2), dtype=numba.float32) # can only initialize using constants..
    
    # Since each thread uses the same sequence, construct in shared memory
    num = math.ceil(timesteps/block_width)
    tstart = min(tid*num, timesteps)
    tend = min(tstart+num, timesteps)
    for i in range(tstart, tend):
      useq_shared[i,0] = u_cur_d[i,0]
      useq_shared[i,1] = u_cur_d[i,1]
    
    # Sync before moving on
    cuda.syncthreads()

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    height, width = lin_sample_grid_batch_d[tid].shape
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

      # Nominal noisy control
      v_nom = useq_shared[t, 0]
      w_nom = useq_shared[t, 1]
      
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
    # sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS]
    # noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    # Assume each thread corresponds to one time step
    # For consistency, each block samples a sequence, and threads (not too many) work together over num_steps
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    abs_thread_id = cuda.grid(1)

    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)

