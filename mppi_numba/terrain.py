#!/usr/bin/env python3
"""
Class definitions for "Terrain", "TDM_Numba", and "TractionGrid".

"TDM_Numba": Traction Distribution Map (TDM) implemented using numba, including core functions 
    for sampling traction maps, representing worst-case expected tractions, and more.
    The underlyin pmf_grid has shape (num_bins, height, width), where at each location, bins sum up to 100 (int8).

"TractionGrid": Deterministic traction map, typically sampled from TDM_Numba for simulation / visualization. 
    Can be used for simulating fixed but unknown terrain tractions from a ground truth distribution.

"Terrain": A semantic terrain type and contains distribution for linear and angular tractions.
    Only used if "TDM_Numba" is initialized from semantic types instead of PMFs directly.
"""

import numpy as np
import time
import math
import copy
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


class Terrain(object):

    """Terrain type has linear and angular traction parameters."""

    def __init__(self, name, rgb, lin_density, ang_density, cvar_alpha=0.1, cvar_front=True, num_saved_samples=1e4):
        self.name = name
        self.lin_density = lin_density
        self.ang_density = ang_density
        self.num_saved_samples = num_saved_samples
        self.lin_saved_samples = self.lin_density.sample(num_saved_samples)
        self.ang_saved_samples = self.ang_density.sample(num_saved_samples)
        
        self.cvar_alpha = cvar_alpha
        self.cvar_front = cvar_front
        self.rgb = rgb
        
        # Save statistics
        self.lin_mean = self.lin_density.mean(self.lin_saved_samples)
        self.lin_var = self.lin_density.var(self.lin_saved_samples)
        self.lin_std = np.sqrt(self.lin_var)
        self.lin_cvar, self.lin_cvar_thres = self.lin_density.cvar(self.cvar_alpha, samples=self.lin_saved_samples, front=cvar_front)

        self.ang_mean = self.ang_density.mean(self.ang_saved_samples)
        self.ang_var = self.ang_density.var(self.ang_saved_samples)
        self.ang_std = np.sqrt(self.ang_var)
        self.ang_cvar, self.ang_cvar_thres = self.ang_density.cvar(self.cvar_alpha, samples=self.ang_saved_samples, front=cvar_front)

    def update_cvar_alpha(self, alpha):
        assert alpha>0 and alpha<=1.0
        self.cvar_alpha = alpha
        self.lin_cvar, self.lin_cvar_thres = self.lin_density.cvar(self.cvar_alpha, samples=self.lin_saved_samples, front=self.cvar_front)
        self.ang_cvar, self.ang_cvar_thres = self.ang_density.cvar(self.cvar_alpha, samples=self.ang_saved_samples, front=self.cvar_front)
    
    def sample_traction(self, num_samples):
        lin_samples = self.lin_density.sample(num_samples)
        ang_samples = self.ang_density.sample(num_samples)
        return lin_samples, ang_samples
    
    def __repr__(self):
        return "Terrain {} has the following properties for linear and angular tractions.\n".format(self.name) + \
                "mean=({:.2f}, {:.2f}), std=({:.2f}, {:.2f}), cvar({:.2f})=({:.2f}, {:.2f}) (computed from {} saved samples)".format(
                    self.lin_mean, self.ang_mean, self.lin_std, self.ang_std, self.cvar_alpha, self.lin_cvar, self.ang_cvar, self.num_saved_samples
                )


class TDM_Numba(object):

    """
    Traction Distribution Map (TDM) leveraging Numba to pre-allocate memory on GPU.
    Internal storage is in the form of (num_bins, height, width) int8 0~100 normalized between min and max traction values (typically 0~1).
    Current implementation assumes unicycle's traction parameters range from 0 to 1.0 (not always explicitly checked).
    Certain functions will break if the previous assumption is not true.


    Typical workflow: 
        1. Initialize object with config that allows pre-initialization of GPU memory
        2. reset()
        3. set_TDM_from_semantic_grid(...) or set_TDM_from_PMF_grid(...)
        4. Pass this object to MPPI_Numba 
        5. Repeat from 2 if traction map changes during replanning
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


        # Other TDM specific params from cfg
        self.thread_dim = self.tdm_sample_thread_dim
        self.block_dim = (1, self.num_grid_samples)
        self.total_threads = self.num_grid_samples*self.thread_dim[0]*self.thread_dim[1]


        # Reuseable device memory
        self.sample_grid_batch_d = None
        self.risk_traction_map_d = None # for adjusting time cost
        self.obstacle_map_d = None # Indicators map for obstacle
        self.unknown_map_d = None # Indicators map for unknown
        self.rng_states_d = None

        # Other task specific params that can be changed for this object
        self.device_var_initialized = False
        self.reset()


    def reset(self):
        # For initialization from semantic grid (for sim benchmarks only)
        self.semantic_grid = None # semantic_grid # semantic ids
        self.semantic_grid_initialized = False
        self.id2name = None # dict[semantic_id]=>name
        self.name2terrain = None # dict[name]=>Terrain object
        self.id2terrain_fn = None
        self.terrain2pmf = None

        # Set the properties for pmf_grid. For now, assume all pmf has the same range
        self.pmf_grid = None
        self.bin_values = None
        self.bin_values_bounds = None
        self.pmf_grid_d = None
        self.bin_values_d = None
        self.bin_values_bounds_d = None
        self.num_pmf_bins = None
        self.xlimits = None
        self.ylimits = None
        self.padded_xlimits = None
        self.padded_ylimits = None
        self.pad_cells = None
        self.res = None
        self.pmf_grid_initialized = False

        self.risk_traction_map_d = None
        self.obstacle_map = None
        self.obstacle_map_d = None
        self.unknown_map = None
        self.unknown_map_d = None

        # For visualization
        self.cell_dimensions = None
        self.figsize = None

        # Initialize batch_sample variables
        self.init_device_vars_before_sampling()


    def init_device_vars_before_sampling(self):
        if not self.device_var_initialized:
            t0 = time.time()
            
            ## Allocate more than needed space to account for varying map size (crop if larger than expected)
            ## Note that uninitialized values are not necessarily 0.
            rows, cols = self.max_map_dim
        
            if not self.det_dyn:
                self.rng_states_d = create_xoroshiro128p_states(self.total_threads, seed=self.seed)
                self.sample_grid_batch_d = cuda.device_array((self.num_grid_samples, rows, cols), dtype=np.int8)
            else:
                self.rng_states_d = create_xoroshiro128p_states(self.thread_dim[0]*self.thread_dim[1], seed=self.seed)
                self.sample_grid_batch_d = cuda.device_array((1, rows, cols), dtype=np.int8)

            self.device_var_initialized = True
            print("TDM has initialized GPU memory after {} s".format(time.time()-t0))


    def set_TDM_from_semantic_grid(self, sg, res, num_pmf_bins, bin_values, bin_values_bounds,
                                  xlimits, ylimits, id2name, name2terrain, terrain2pmf,
                                  det_dynamics_cvar_alpha=None, # What's the alpha quantile for worst-case dynamics?
                                  obstacle_map=None,
                                  unknown_map=None):

        """
        Mainly used for simulation benchmark where semantics and their ground truth properties are known.
        Save semantic grid and initialize visualization parameters. Initialize the PMF grid and copy to device. 
        Optionally obstacle_map and unknown_map can be processed and padded to represent regions to avoid.
        """

        if det_dynamics_cvar_alpha is None:
            assert self.use_tdm or self.use_costmap
        else:
            assert det_dynamics_cvar_alpha >0 and det_dynamics_cvar_alpha<=1.0

        # Based on semantics, construct the grid 
        self.semantic_grid = sg.copy()
        self.id2name = id2name # dict[semantic_id]=>name
        self.name2terrain = name2terrain # dict[name]=>Terrain object
        self.id2terrain_fn = lambda semantic_id: self.name2terrain[self.id2name[semantic_id]]
        self.terrain2pmf = terrain2pmf
        self.semantic_grid_initialized = True
        self.cell_dimensions = (res, res)
        self.xlimits = xlimits
        self.ylimits = ylimits
        num_rows, num_cols = sg.shape
        self.num_pmf_bins = num_pmf_bins
        self.bin_values = np.asarray(bin_values).astype(np.float32)
        self.bin_values_bounds = np.asarray(bin_values_bounds).astype(np.float32)
        self.res = res
        # self.det_dynamics_cvar_alpha = det_dynamics_cvar_alpha

        assert bin_values[0]==0, "Assume minimum bin value is 0 for now"
        assert bin_values_bounds[0]==0, "Assume minimum traction is 0 for now"

        # Initialize pmf grid and account for padding
        self.pmf_grid = np.zeros((self.num_pmf_bins, num_rows, num_cols), dtype=np.int8)

        # Use dynamics computed based on cvar_alpha
        # Use CVaR dynamics (alpha=1 ==> mean dynamics)
        if self.use_det_dynamics:
            for ri in range(num_rows):
                for ci in range(num_cols):

                    # Handle alpha==1 separately due to numerical errors
                    if det_dynamics_cvar_alpha==1.0:
                        # Simple weighted sum. 
                        terrain = self.id2terrain_fn(self.semantic_grid[ri, ci])
                        values, pmf = self.terrain2pmf[terrain]
                        expected_traction = 0.0
                        for val, m in zip(values, pmf):
                            expected_traction += m*val
                        # Find the corresponding bin for the traction
                        for bin_i, vval in enumerate(values):
                            if expected_traction <= vval:
                                self.pmf_grid[bin_i, ri, ci] = np.int8(100)
                                break
                    else:
                        # Expected value in the west-alpha percentile
                        terrain = self.id2terrain_fn(self.semantic_grid[ri, ci])
                        values, pmf = self.terrain2pmf[terrain]
                        cum_sum = 0.0
                        expected_traction = 0.0
                        for val, m in zip(values, pmf):
                            cum_sum += m
                            expected_traction += m*val
                            if cum_sum >= det_dynamics_cvar_alpha:
                                if cum_sum > 0:
                                    expected_traction /= cum_sum
                                # Find the corresponding bin for the traction
                                for bin_i, vval in enumerate(values):
                                    if expected_traction <= vval:
                                        self.pmf_grid[bin_i, ri, ci] = np.int8(100)
                                        break
                                break
                    assert sum(self.pmf_grid[:,ri, ci])==100


        # Initialize PMFs with nominal dynamics, and create risk_traction_map (or the worst-case speed map)
        elif self.use_nom_dynamics_with_speed_map:

            self.pmf_grid[-1,:,:] = np.int8(100)
            num_rows, num_cols = self.semantic_grid.shape
            unique_ids = np.unique(self.semantic_grid)
            id_2_values_and_pmf = {sid: self.terrain2pmf[self.id2terrain_fn(sid)] 
                                    for sid in unique_ids}
            num_layers = len(id_2_values_and_pmf[unique_ids[0]][1])
            pmf_grid = np.zeros((num_layers, num_rows, num_cols), dtype=float) # Here each axis=0 actually sums to 1 (float) like normal PMF
            bin_values_grid = np.zeros((num_layers, num_rows, num_cols), dtype=float) 
            # fill pmf_grid with values
            for id in unique_ids:
                values, pmf = id_2_values_and_pmf[id]
                pmf_grid[:,self.semantic_grid==id] = np.reshape(pmf, (num_layers, 1))
                bin_values_grid[:,self.semantic_grid==id] = np.reshape(values, (num_layers, 1))
            
            pmf_cumsum = pmf_grid.cumsum(axis=0)
            weighted_values = pmf_grid * bin_values_grid
            weighted_v_cumsum = np.cumsum(weighted_values, axis=0)
            traction_range = self.bin_values_bounds[1] - self.bin_values_bounds[0]
            
            if det_dynamics_cvar_alpha == 1.0:
                # risk_traction_map = np.int8(100) * np.ones((1, num_rows, num_cols), dtype=np.int8)
                # Find mean traction values
                risk_traction_map = np.reshape(
                    100*(weighted_v_cumsum[-1]-self.bin_values_bounds[0])/traction_range,
                    (1, num_rows, num_cols)
                ).astype(np.int8)

            else:
                # Up to which PMF bin?
                which_layer = np.argmax(pmf_cumsum>=det_dynamics_cvar_alpha, axis=0)
                l_indices = which_layer.ravel()
                r_indices = np.repeat(np.arange(num_rows), num_cols)
                c_indices = np.tile(np.arange(num_cols), num_rows)
                # Conditional mean
                cvars = weighted_v_cumsum[l_indices, r_indices, c_indices] / pmf_cumsum[l_indices, r_indices, c_indices].ravel()
                risk_traction_map = np.reshape(
                    100*np.asarray((cvars.reshape(num_rows, num_cols)-self.bin_values_bounds[0])/traction_range),
                    (1,num_rows,num_cols)
                ).astype(np.int8)
            
            # Padd the worst case risk
            padded_risk_traction_map, _, _ = self.set_padding_risk_traction(risk_traction_map, self.max_speed_padding, self.dt, res, xlimits, ylimits)
            self.risk_traction_map_d = cuda.to_device(padded_risk_traction_map)


        # Capture the entire PMF of terrain
        elif self.use_tdm:

            # Stochastic dynamics
            for ri in range(num_rows):
                for ci in range(num_cols):
                    terrain = self.id2terrain_fn(self.semantic_grid[ri, ci])
                    values, pmf = self.terrain2pmf[terrain]
                    # self.pmf_grid[:, ri, ci] = np.rint(pmf*100).astype(np.int8)
                    self.pmf_grid[:, ri, ci] = np.int8(pmf*100)
                    # Make sure cum sum is 100
                    self.pmf_grid[-1, ri, ci] = np.int8(100)-np.sum(self.pmf_grid[:-1, ri, ci])

                    assert sum(self.pmf_grid[:,ri, ci])==100

        else:
            assert False, "TDM cannot be set up"
        
        padded_pmf_grid, self.padded_xlimits, self.padded_ylimits \
            = self.set_padding(self.pmf_grid, self.max_speed_padding, self.dt, res, xlimits, ylimits)
        self.pmf_grid_d = cuda.to_device(padded_pmf_grid)
        self.bin_values_d = cuda.to_device(bin_values)
        self.bin_values_bounds_d = cuda.to_device(bin_values_bounds)

        self.prepare_obstacle_and_unknown_map(obstacle_map, unknown_map, num_rows, num_cols, res)
        
        # Crop the original semantic map to fit in memory
        num_rows, num_cols = self.pmf_grid_d.shape[1:]
        original_semantic_grid = copy.deepcopy(self.semantic_grid)
        self.semantic_grid = original_semantic_grid[:num_rows-2*self.pad_cells, :num_cols - 2*self.pad_cells]

        self.pmf_grid_initialized = True


    def get_padded_grid_xy_dim(self):
        if self.pmf_grid_initialized:
            return self.pmf_grid_d.shape[1:]
        else:
            print("Padded grid has not been initialized yet.")
            return None

    
    def prepare_obstacle_and_unknown_map(self, obstacle_map, unknown_map, num_rows, num_cols, res):
                # Vars for holding obstacles and unknown masks
        if obstacle_map is not None:
            assert obstacle_map.shape==(num_rows, num_cols), "obstacle_map does not have the same XY dim as pmf grid."
            self.obstacle_map = np.asarray(obstacle_map).astype(np.int8).reshape(num_rows, num_cols)
        else:
            self.obstacle_map = np.zeros((num_rows, num_cols), dtype=np.int8)

        if unknown_map is not None:
            assert unknown_map.shape==(num_rows, num_cols), "unknown_map does not have the same XY dim as pmf grid."
            self.unknown_map = np.asarray(unknown_map).astype(np.int8).reshape(num_rows, num_cols)
        else:
            self.unknown_map = np.zeros((num_rows, num_cols), dtype=np.int8)
        
        # Also pad the obstacle and unknown mask
        padded_obstacle_map = self.set_padding_2d(self.obstacle_map, self.max_speed_padding, self.dt, res)
        padded_unknown_map = self.set_padding_2d(self.unknown_map, self.max_speed_padding, self.dt, res)
        self.obstacle_map_d = cuda.to_device(padded_obstacle_map)
        self.unknown_map_d = cuda.to_device(padded_unknown_map)


    def print_bin_values_bounds(self, obj_name):
        if self.bin_values_bounds_d is None:
            print("{}: Bin value is None".format(obj_name))
        else:
            print("{}: bin values bounds are ".format(obj_name), self.bin_values_bounds_d.copy_to_host())
        
    def set_TDM_from_PMF_grid(self, pmf_grid, tdm_dict, obstacle_map=None, unknown_map=None):
        
        """
        Initialize TDM from PMF grid that represents tractions. 
        Dimension for pmf_grid: (num_bins, height, width). At each location, bins sum up to 100 (int8)
        Optionally obstacle_map and unknown_map can be processed and padded to represent regions to avoid.
        """

        # Code for interfacing PMF values from cpp interface
        # Input pmf_grid has shape (num_bins, num_rows, num_cols), where all bins sum to 100 for (row, col)
        if not (tdm_dict["det_dynamics_cvar_alpha"]>0 and tdm_dict["det_dynamics_cvar_alpha"]<=1.0 ):
            print("WARNING: TDM cannot be setup since alpha is not in (0,1]")
        assert tdm_dict["det_dynamics_cvar_alpha"]>0
        assert tdm_dict["det_dynamics_cvar_alpha"]<=1.0
        assert len(pmf_grid.shape)==3, "PMF grid must have 3 dimensions"
        self.num_pmf_bins, num_rows, num_cols = pmf_grid.shape
        self.res = res = tdm_dict["res"]
        self.cell_dimensions = (res, res)
        self.xlimits = tdm_dict["xlimits"]
        self.ylimits = tdm_dict["ylimits"]

        self.bin_values = np.asarray(tdm_dict["bin_values"]).astype(np.float32)
        self.bin_values_bounds = np.asarray(tdm_dict["bin_values_bounds"]).astype(np.float32)
        assert self.bin_values[0]==0, "Assume minimum bin value is 0 for now"
        assert self.bin_values_bounds[0]==0, "Assume minimum traction is 0 for now"
        self.bin_values_d = cuda.to_device(self.bin_values)
        self.bin_values_bounds_d = cuda.to_device(self.bin_values_bounds)

        if self.use_det_dynamics:
            if (np.sum(pmf_grid, axis=0)!=100).any():
                print("WARNING: the provided PMF has columns that don't sum up to 100: {}".fromat(
                    np.argwhere(np.sum(pmf_cumsum, axis=0)!=100)))
            
            # Use modified PMF that has 100% prob mass at the bin that approximately equals cvar
            # Use dynamics computed based on cvar_alpha
            # Use CVaR dynamics (alpha=1 ==> mean dynamics)
            self.pmf_grid = np.zeros((self.num_pmf_bins, num_rows, num_cols), dtype=np.int8)

            # Process the incoming data
            pmf_cumsum = 0.01*pmf_grid.cumsum(axis=0).astype(float) # summed to 1 (float)
            weighted_values = 0.01* (pmf_grid.astype(float)) * self.bin_values.reshape((-1,1,1)) 
            weighted_v_cumsum = np.cumsum(weighted_values, axis=0) # sum to 1 (float)

            r_indices = np.repeat(np.arange(num_rows), num_cols)
            c_indices = np.tile(np.arange(num_cols), num_rows)
            
            # Why handling cvar_alpha==1.0 separately? Sometimes computed float that's close to 1.0 does not get recognized as >=1
            if (tdm_dict["det_dynamics_cvar_alpha"]==1.0):
                # Compute the expected dynamics (true mean, not scaled by 100)
                means = weighted_v_cumsum[-1]

                # Compute which bin is the approximation (which layer in each x y location)
                which_layer = np.argmax(means <=self.bin_values.reshape((-1, 1, 1)), axis=0)
                l_indices = which_layer.ravel()
                self.pmf_grid[l_indices, r_indices, c_indices] = np.int8(100)

            else:
                # Find up to which bins the CVaR values should be computed
                # upto_which_layer_to_compute_cvar = np.argmax(pmf_cumsum<=tdm_dict["det_dynamics_cvar_alpha"], axis=0)
                upto_which_layer_to_compute_cvar = np.argmax(pmf_cumsum>=tdm_dict["det_dynamics_cvar_alpha"], axis=0)
                l_indices_to_compute_cvar = upto_which_layer_to_compute_cvar.ravel()
                # Compute the CVaR by dividing the total mass of the worst-percentiles
                cvars = (weighted_v_cumsum[l_indices_to_compute_cvar, r_indices, c_indices] / \
                    (pmf_cumsum[l_indices_to_compute_cvar, r_indices, c_indices]+1e-6)).reshape((num_rows, num_cols))

                # Find the bins that approximate this CVaR
                which_layer = np.argmax(cvars <=self.bin_values.reshape((-1, 1, 1)), axis=0)
                l_indices = which_layer.ravel()
                self.pmf_grid[l_indices, r_indices, c_indices] = np.int8(100)

            # assert (np.sum(self.pmf_grid, axis=0)==(np.ones((num_rows, num_cols))*100)).all()
            if (np.sum(self.pmf_grid, axis=0)!=(np.ones((num_rows, num_cols))*100)).all():
                print("WARNING: pmf_grid not properly set in set_TDM_from_PMF_grid. Values don't' sum to 100")


        elif self.use_nom_dynamics_with_speed_map:
            if (np.sum(pmf_grid, axis=0)!=100).any():
                print("WARNING: the provided PMF has columns that don't sum up to 100: {}".fromat(
                    np.argwhere(np.sum(pmf_cumsum, axis=0)!=100)))
            
            # Use modified PMF that has 100% prob mass at the bin that approximately equals cvar
            # Use dynamics computed based on cvar_alpha
            # Use CVaR dynamics (alpha=1 ==> mean dynamics)
            self.pmf_grid = np.zeros((self.num_pmf_bins, num_rows, num_cols), dtype=np.int8)
            self.pmf_grid[-1] = np.int8(100) # last layer == 100 implies that nominal dynamics (traction==1)will be used

            # Process the incoming data
            pmf_cumsum = 0.01*pmf_grid.cumsum(axis=0).astype(float) # summed to 1 (float)
            weighted_values = 0.01* (pmf_grid.astype(float)) * self.bin_values.reshape((-1,1,1)) 
            weighted_v_cumsum = np.cumsum(weighted_values, axis=0) # sum to 1 (float)

            traction_range = self.bin_values_bounds[1] - self.bin_values_bounds[0]

            # Compute 
            if tdm_dict["det_dynamics_cvar_alpha"]==1.0:
                # Compute the expected dynamics (true mean, not scaled by 100)
                risk_traction_map = np.reshape(
                    100*(weighted_v_cumsum[-1]-self.bin_values_bounds[0])/traction_range,
                    (1, num_rows, num_cols)).astype(np.int8)
                    
            else:
                # Up to which PMF bin?                
                which_layer = np.argmax(pmf_cumsum>=tdm_dict["det_dynamics_cvar_alpha"], axis=0)
                l_indices = which_layer.ravel()
                r_indices = np.repeat(np.arange(num_rows), num_cols)
                c_indices = np.tile(np.arange(num_cols), num_rows)
                # Conditional mean
                cvars = weighted_v_cumsum[l_indices, r_indices, c_indices] / (pmf_cumsum[l_indices, r_indices, c_indices].ravel()+1e-6)
                risk_traction_map = np.reshape(
                    100*np.asarray((cvars.reshape(num_rows, num_cols)-self.bin_values_bounds[0])/traction_range),
                    (1,num_rows,num_cols)).astype(np.int8)

            # Padd the worst case risk
            padded_risk_traction_map, _, _ = self.set_padding_risk_traction(risk_traction_map, self.max_speed_padding, self.dt, res, self.xlimits, self.ylimits)
            # padded_risk_traction_map = self.set_padding_2d(risk_traction_map, self.max_speed_padding, self.dt, res, pad_val=0)
            self.risk_traction_map_d = cuda.to_device(padded_risk_traction_map)

        else:
            # For proposed method, use the existing PMF
            self.pmf_grid = np.asarray(pmf_grid).astype(np.int8)
            
        if (np.sum(self.pmf_grid, axis=0)!=100).any():
            print("WARNING: some PMF columns do not sum to 100: {}".format(np.argwhere(np.sum(self.pmf_grid, axis=0)!=100)))

        padded_pmf_grid, self.padded_xlimits, self.padded_ylimits = self.set_padding(self.pmf_grid, self.max_speed_padding, self.dt, res,
                                                            self.xlimits, self.ylimits)
        self.pmf_grid_d = cuda.to_device(padded_pmf_grid)
        self.prepare_obstacle_and_unknown_map(obstacle_map, unknown_map, num_rows, num_cols, res)
        self.pmf_grid_initialized = True


    def set_padding_risk_traction(self, grid, max_speed_padding, dt, res, xlimits, ylimits):
        valid_rows, valid_cols, pad_cells = self.get_padding_info(grid.shape, max_speed_padding, dt, res)
        self.pad_cells = pad_cells

        padded_xlimits = np.array([xlimits[0]-pad_cells*res, xlimits[0]+(valid_cols+pad_cells)*res])
        padded_ylimits = np.array([ylimits[0]-pad_cells*res, ylimits[0]+(valid_rows+pad_cells)*res])
        
        # Extract the valid submap from the provided pmf_grid
        padded_grid = np.zeros((1, valid_rows+2*pad_cells, valid_cols+2*pad_cells), dtype=np.int8)
        padded_grid[:, pad_cells:(pad_cells+valid_rows), pad_cells:(pad_cells+valid_cols)] = grid[:,:valid_rows, :valid_cols]
        
        return padded_grid, padded_xlimits, padded_ylimits


    def set_padding(self, pmf_grid,  max_speed_padding, dt, res, xlimits, ylimits):
        """
        Padd the nominal pmf grid with 0_traction components (assumed to be the first bin values)
        This function also checks the size of allocated GPU memory and crop the provided PMF accordingly 
        (from bottom left origin) while leaving enough memory for padding.
        """

        valid_rows, valid_cols, pad_cells = self.get_padding_info(pmf_grid.shape, max_speed_padding, dt, res)
        self.pad_cells = pad_cells

        padded_xlimits = np.array([xlimits[0]-pad_cells*res, xlimits[0]+(valid_cols+pad_cells)*res])
        padded_ylimits = np.array([ylimits[0]-pad_cells*res, ylimits[0]+(valid_rows+pad_cells)*res])
        
        # Extract the valid submap from the provided pmf_grid
        padded_pmf_grid = np.zeros((self.num_pmf_bins, valid_rows+2*pad_cells, valid_cols+2*pad_cells), dtype=np.int8)
        padded_pmf_grid[0] = np.int8(100) # Fill the probability mass associated with 0 traction
        padded_pmf_grid[:, pad_cells:(pad_cells+valid_rows), pad_cells:(pad_cells+valid_cols)] = pmf_grid[:,:valid_rows, :valid_cols]
        
        return padded_pmf_grid, padded_xlimits, padded_ylimits


    def set_padding_2d(self, map, max_speed_padding, dt, res, pad_val=0):
        """
        Padd the nominal pmf grid with 0_traction components (assumed to be the first bin values)
        This function also checks the size of allocated GPU memory and crop the provided PMF accordingly 
        (from bottom left origin) while leaving enough memory for padding.
        """
        valid_rows, valid_cols, pad_cells = self.get_padding_info(map.shape, max_speed_padding, dt, res)
        self.pad_cells = pad_cells

        # Extract the valid submap the traction and unknown maps 
        padded_map = pad_val*np.ones((valid_rows+2*pad_cells, valid_cols+2*pad_cells), dtype=np.int8)
        padded_map[pad_cells:(pad_cells+valid_rows), pad_cells:(pad_cells+valid_cols)] = map[:valid_rows, :valid_cols]

        return padded_map


    def get_padding_info(self, grid_shape, max_speed_padding, dt, res):
        if len(grid_shape)==3:
            _, rows, cols = grid_shape
        elif len(grid_shape)==2:
            rows, cols = grid_shape
        pad_cells = int(np.ceil(max_speed_padding*dt/res))

        # Based on allocated GPU mem
        max_rows = self.max_map_dim[0]-2*pad_cells
        max_cols = self.max_map_dim[1]-2*pad_cells
        if max_rows < 1 or max_cols < 1:
            print("While padding the TDM, the max_allowed rows {} or cols {} are below 1.\nAllocated GPU array size: {}".format(
                max_rows, max_cols, 
                [1 if self.det_dyn else self.num_grid_samples, self.max_map_dim[0], self.max_map_dim[1]]))
            assert False
        
        valid_rows = min(max_rows, rows)
        valid_cols = min(max_cols, cols)
        if valid_rows<rows or valid_cols<cols:
            print("WARNING: While padding the TDM, original PMF is cropped from ({}, {}) to ({}, {})to fit in allocated GPU memory.".format(
                rows, cols, valid_rows , valid_cols ))
        return valid_rows, valid_cols, pad_cells

    
    def sample_grids_true_dist(self):
        # TODO: This has access to both linear and angular distributions.. Maybe confusing since this class is supposed to be either linear or angular
        # Get a single sample from the true underlying distribution instead of the PMF
        # Count number of each semantic types

        sid2num = dict()
        for sid in self.semantic_grid.flatten():
            if sid not in sid2num:
                sid2num[sid]=1
            else:
                sid2num[sid]+=1
        sid2tractions = dict()
        for sid, num in sid2num.items():
            sid2tractions[sid] = self.id2terrain_fn(sid).sample_traction(num)
        lins = np.zeros_like(self.semantic_grid, dtype=float)
        angs = np.zeros_like(self.semantic_grid, dtype=float)

        for sid, num in sid2num.items():
            mask = (self.semantic_grid==sid)
            lins[mask] = sid2tractions[sid][0] 
            angs[mask] = sid2tractions[sid][1]

        return TractionGrid(lins, angs)

    def sample_grids(self, alpha_dyn=1.0):
        # Invoke the GPU kernels to sample from the PMF approximation
        if not self.det_dyn:
            self.sample_grids_numba[self.block_dim, self.thread_dim](
                self.sample_grid_batch_d, self.pmf_grid_d, self.rng_states_d,
                self.bin_values_d, self.bin_values_bounds_d, alpha_dyn
            )
        else:
            self.sample_grids_numba[(1,1), self.thread_dim](
                self.sample_grid_batch_d, self.pmf_grid_d, self.rng_states_d,
                self.bin_values_d, self.bin_values_bounds_d, alpha_dyn
            )
        return self.sample_grid_batch_d


    def int8_grid_to_float32(self, int8grid):
        # int8 value between 0 and 100 represent some value within bin_values_bounds
        ratio = np.asarray(int8grid.copy()).astype(np.float32)/100.
        return ratio*(self.bin_values_bounds[1]-self.bin_values_bounds[0])+self.bin_values_bounds[0]


    """ GPU kernels """

    @staticmethod
    @cuda.jit(fastmath=True)
    def sample_grids_numba(grid_batch_d, pmf_grid_d, rng_states_d,
                     bin_values_d, bin_values_bounds_d, alpha_dyn):
        # pmf_grid_d has been cropped in xy dimension and properly padded, 
        # which can only have smaller size than grid_batch_d's xy shape

        # Each 2D block samples a single grid
        # Only consider a single row of block (1, num_blocks)
        # Every thread takes care of a small section of the grid
        # Return a reference to sampled grids on GPU

        threads_x = cuda.blockDim.x # row
        threads_y = cuda.blockDim.y # col
        blocks_x = cuda.gridDim.x # row
        blocks_y = cuda.gridDim.y # col
        num_bins, grid_rows, grid_cols = pmf_grid_d.shape
        num_col_entries_per_thread = math.ceil(grid_cols/threads_y)
        num_rows_entries_per_thread = math.ceil(grid_rows/threads_x)
        
        # thread info
        block_id = cuda.blockIdx.y#
        tid_x = cuda.threadIdx.x # index within block
        tid_y = cuda.threadIdx.y # index within block
        abs_tid_x, abs_tid_y = cuda.grid(2) # absolute x, y index
        thread_id = abs_tid_x*threads_y*blocks_y + abs_tid_y
        # print(thread_id)
        # cuda.syncthreads()
        # a, b, c = grid_batch_d.shape
        # print("grid_batch.shape", a, b, c, " pmf_grid.shape", num_bins, grid_rows, grid_cols)

        # Compute horizontal and vertical index range
        ri_start = min(tid_x*num_rows_entries_per_thread, grid_rows)
        ri_end = min(ri_start+num_rows_entries_per_thread, grid_rows)
        ci_start = min(tid_y*num_col_entries_per_thread, grid_cols)
        ci_end = min(ci_start+num_col_entries_per_thread, grid_cols)

        if (ri_end>grid_batch_d.shape[1]):
            print("row idx out of bound ")

        if (ci_end>grid_batch_d.shape[2]):
            print("col idx out of bound ")

        traction_range = bin_values_bounds_d[1]-bin_values_bounds_d[0]
        # cum_pmf = np.int8(0)
        # sampled_cum_pmf = np.int8(0)
        for ri in range(ri_start, ri_end):
            for ci in range(ci_start, ci_end):
                # Check which bin this belongs to
                rand_num = xoroshiro128p_uniform_float32(rng_states_d, thread_id)
                sampled_cum_pmf = np.int8(math.ceil(rand_num*100.0*alpha_dyn))
                cum_pmf = np.int8(0)
                for bi in range(num_bins):
                    cum_pmf += pmf_grid_d[bi, ri, ci]
                    if sampled_cum_pmf <= cum_pmf:
                        before = grid_batch_d[block_id, ri, ci] # should be np.int8
                        grid_batch_d[block_id, ri, ci] = np.int8(100.*(bin_values_d[bi]-bin_values_bounds_d[0])/traction_range)
                        
                        if grid_batch_d[block_id, ri, ci]<0:
                            print("TDM sample_grids_numba experiences values < 0. Before=", before, "now", grid_batch_d[block_id, ri, ci], "before in8 cast", 100.*(bin_values_d[bi]-bin_values_bounds_d[0])/traction_range)
                            print("bin_values[", bi,"]=", bin_values_d[bi], " bin_values_bounds", bin_values_bounds_d[0], bin_values_bounds_d[1], "traction_range", traction_range)
                        break
        # cuda.syncthreads()


    # def set_TDM_from_costmap(self, costmap_dict, obstacle_map=None, unknown_map=None):
    #     assert self.use_costmap, "set_TDM_from_costmap is invoked when self.use_costmap is not True"
    #     # print('in set_TDM_from_Costmap')
    #     start_t = time.time()
    #     costmap = costmap_dict["costmap"]

    #     # Based on semantics, construct the grid 
    #     res = costmap_dict["res"]
    #     self.res = res
    #     self.cell_dimensions = (res, res)
    #     self.xlimits = costmap_dict["xlimits"]
    #     self.ylimits = costmap_dict["ylimits"]

    #     num_rows, num_cols = costmap_dict["costmap"].shape
    #     self.num_pmf_bins = 2 # costmap_dict["num_pmf_bins"]
    #     self.bin_values = np.array([0.0, 1.0], dtype=np.float32) # costmap_dict["bin_values"].astype(np.float32)
    #     self.bin_values_bounds = np.array([min(self.bin_values), max(self.bin_values)], dtype=np.float32) # costmap_dict["bin_values_bounds"].astype(np.float32)

    #     # TODO: check and pad obstacle and unknown map
    #     # # Initialize pmf grid
    #     # Account for padding
    #     self.pmf_grid = np.zeros((self.num_pmf_bins, num_rows, num_cols), dtype=np.int8)
    #     self.pmf_grid[-1] = np.int8(100) # Set PMF to have 1 in the last bin

    #     print("Took {} to initialize pmf grid".format(time.time()- start_t))
    #     start_t = time.time()

    #     # Generate the risk speed map (CVaR)
    #     risk_traction_map = 100*np.ones((num_rows, num_cols), dtype=np.int8)
    #     no_info_mask = (costmap==255)
    #     lethal_mask = ((costmap > costmap_dict["costmap_lethal_threshold"]) & (~no_info_mask))
    #     risk_traction_map[no_info_mask] = -1
    #     risk_traction_map[lethal_mask] = -2
    #     self.risk_traction_map_d = cuda.to_device(risk_traction_map.reshape((1, num_rows, num_cols)))
    #     print("Took {} to handle costmap values iteratively".format(time.time()- start_t))

    #     start_t = time.time()
    #     padded_pmf_grid, self.padded_xlimits, self.padded_ylimits = self.set_padding(self.pmf_grid, self.max_speed_padding, self.dt, res,
    #                                                         self.xlimits, self.ylimits)
    #     self.pmf_grid_d = cuda.to_device(padded_pmf_grid)
    #     self.bin_values_d = cuda.to_device(self.bin_values)
    #     self.bin_values_bounds_d = cuda.to_device(self.bin_values_bounds)
        
    #     # How to ensure the padded map fits properly in the allocated device memory?
    #     # Check dimension 
    #     # Get max col and max rows to sample over (must be the min of allocated and provided pmf)
    #     # Store this info in class variable 

    #     self.pmf_grid_initialized = True
    #     print("Took {} to transfer procesed PMF grid to device".format(time.time()- start_t))


class TractionGrid(object):
    
    """A deterministic grid with traction coefficients (can be generated by SDM)"""
    
    def __init__(self, lin_traction, ang_traction, res=1.0, use_int8=False, xlimits=None, ylimits=None):
        if use_int8:
          # Use int 0-100 to represent values between 0 and 1
          self.lin_traction = (100*lin_traction).astype(np.int8)
          self.ang_traction = (100*ang_traction).astype(np.int8)
        else:
          self.lin_traction = lin_traction
          self.ang_traction = ang_traction
        self.res = res
        self.height, self.width = self.lin_traction.shape
        if xlimits is None:
            self.xlimits = (0, self.res*self.width)
        else:
            self.xlimits = xlimits
        
        if ylimits is None:
            self.ylimits = (0, self.res*self.height)
        else:
            self.ylimits = ylimits
        

    def get(self, x,y):
        # If within bounds, return queried value. Otherwise, return 0
        xi = int((x-self.xlimits[0])//self.res)
        yi = int((y-self.ylimits[0])//self.res)
        if (xi<0) or (xi>=self.width) or (yi<0) or (yi>=self.height):
            return 0, 0
        else:
            return self.lin_traction[yi, xi], self.ang_traction[yi, xi]
    
    def get_grids(self):
        return self.lin_traction, self.ang_traction
