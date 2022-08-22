
#!/usr/bin/env python3
import numpy as np
import math
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


# TODO: import the control configurations from a config file?
from config import Config
cfg = Config()

# Terrain type has linear and angular traction parameters
class Terrain(object):
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
        # return "Terrain {} has mean={:.2f}, std={:.2f}, cvar({:.2f})={:.2f} (computed from {} saved samples)".format(
        #     self.name, self.mean, self.std, self.cvar_alpha, self.cvar, self.num_saved_samples)
        return "Terrain {} has the following properties for linear and angular tractions.\n".format(self.name) + \
                "mean=({:.2f}, {:.2f}), std=({:.2f}, {:.2f}), cvar({:.2f})=({:.2f}, {:.2f}) (computed from {} saved samples)".format(
                    self.lin_mean, self.ang_mean, self.lin_std, self.ang_std, self.cvar_alpha, self.lin_cvar, self.ang_cvar, self.num_saved_samples
                )



"""
Traction Distribution Map (TDM) leveraging Numba to pre-allocate memory on GPU.
Internal storage is in the form of (num_bins, height, width) int8 0~100 normalized between min and max traction values (typically 0~1).
"""
class TDM_Numba(object):

    NUM_GRID_SAMPLES = cfg.num_grid_samples
    BLOCK_DIM = (1, NUM_GRID_SAMPLES)
    THREAD_DIM = cfg.tdm_sample_thread_dim
    TOTAL_THREADS = NUM_GRID_SAMPLES*THREAD_DIM[0]*THREAD_DIM[1]

    def __init__(self, max_speed, dt=cfg.dt):

        # Used for padding 0 traction regions around the map
        self.max_speed = max_speed
        self.dt = dt
        self.num_cells_to_pad = None


        # For initialization from semantic grid (for sim benchmarks only)
        self.semantic_grid = None # semantic_grid # semantic ids
        self.semantic_grid_initialized = False
        self.id2name = None # dict[semantic_id]=>name
        self.name2terrain = None # dict[name]=>Terrain object
        self.id2terrain_fn = None
        self.terrain2pmf = None

        # Set the properties for pmf_grid.
        # For now, assume all pmf has the same range
        self.pmf_grid = None
        self.bin_values = None
        self.bin_values_bounds = None
        self.pmf_grid_d = None # data on device
        self.bin_values_d = None
        self.bin_values_bounds_d = None
        self.num_pmf_bins = None
        self.num_cols = None
        self.num_rows = None
        self.padded_num_cols = None
        self.padded_num_rows = None
        self.xlimits = None
        self.ylimits = None
        self.padded_xlimits = None
        self.padded_ylimits = None
        self.pad_width = None
        self.res = None
        self.pmf_grid_initialized = False

        # For variants of MPPI
        self.use_det_dynamics = None
        self.det_dynamics_cvar_alpha = None
        self.use_nom_dynamics_with_speed_map = None

        # Initialize batch_sample variables
        self.sample_grid_batch_d = None
        self.risk_traction_map_d = None # for adjusting time cost
        self.rng_states_d = None
        self.device_var_initialized = False

        # For visualization
        self.cell_dimensions = None
        self.figsize = None

    def set_TDM_from_semantic_grid(self, sg, res, num_pmf_bins, bin_values, bin_values_bounds,
                                  xlimits, ylimits, id2name, name2terrain, terrain2pmf,
                                  use_det_dynamics=False, det_dynamics_cvar_alpha=None, use_nom_dynamics_with_speed_map=False):
        """
        Save semantic grid and initialize visualization parameters.
        initialize the PMF grid and copy to device. 
        Return: (pmf_grid, pmf_grid_d)
        """
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
        self.num_rows, self.num_cols = sg.shape
        self.num_pmf_bins = num_pmf_bins
        self.bin_values = np.asarray(bin_values).astype(np.float32)
        self.bin_values_bounds = np.asarray(bin_values_bounds).astype(np.float32)
        self.res = res
        assert bin_values[0]==0, "Assume minimum bin value is 0 for now"
        assert bin_values_bounds[0]==0, "Assume minimum traction is 0 for now"
        assert not (use_det_dynamics and use_nom_dynamics_with_speed_map), \
            "In 'set_TDM_from_semantic_grid', cannot set both use_det_dynamics and use_nom_dynamics_with_speed_map to True"
        assert (not use_det_dynamics) or (det_dynamics_cvar_alpha is not None), \
            "When using deterministic dynamics, det_dynamics_cvar_alpha must be set."

        self.use_det_dynamics = use_det_dynamics
        self.det_dynamics_cvar_alpha = det_dynamics_cvar_alpha
        self.use_nom_dynamics_with_speed_map = use_nom_dynamics_with_speed_map

        # Initialize pmf grid
        # Account for padding
        self.pmf_grid = np.zeros((self.num_pmf_bins, self.num_rows, self.num_cols), dtype=np.int8)
        
        if use_det_dynamics:
            # Use dynamics computed based on cvar_alpha
            # Use CVaR dynamics (alpha=1 ==> mean dynamics)
            for ri in range(self.num_rows):
                for ci in range(self.num_cols):
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
                    # assert sum(self.pmf_grid[:,ri, ci])==100
            
        elif use_nom_dynamics_with_speed_map:
            # Set PMF to have 1 in the last bin
            self.pmf_grid[-1,:,:] = np.int8(100)

            # Generate the risk speed map (CVaR)
            risk_traction_map = np.zeros((1, self.num_rows, self.num_cols), dtype=np.int8)
            traction_range = self.bin_values_bounds[1] - self.bin_values_bounds[0]
            for ri in range(self.num_rows):
                for ci in range(self.num_cols):
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
                            risk_traction_map[0, ri, ci] = np.int8(100*(expected_traction-self.bin_values_bounds[0])/traction_range)
                            break

            self.risk_traction_map_d = cuda.to_device(risk_traction_map)

        else:

            # Stochastic dynamics
            for ri in range(self.num_rows):
                for ci in range(self.num_cols):
                    terrain = self.id2terrain_fn(self.semantic_grid[ri, ci])
                    values, pmf = self.terrain2pmf[terrain]
                    self.pmf_grid[:, ri, ci] = np.rint(pmf*100).astype(np.int8)
                    # # Make sure cum sum is 100
                    # self.pmf_grid[-1, ri, ci] = np.int8(100)-np.sum(self.pmf_grid[:-1, ri, ci])
        
        padded_pmf_grid, padded_xlimits, padded_ylimits = self.set_padding(self.pmf_grid, self.max_speed, self.dt, res,
                                                            xlimits, ylimits)
        self.pmf_grid_d = cuda.to_device(padded_pmf_grid)
        self.padded_xlimits = padded_xlimits
        self.padded_ylimits = padded_ylimits
        _, self.padded_num_cols, self.padded_num_rows = padded_pmf_grid.shape
        self.bin_values_d = cuda.to_device(bin_values)
        self.bin_values_bounds_d = cuda.to_device(bin_values_bounds)
        self.pmf_grid_initialized = True

        
    # def set_TDM_from_PMF_grid(self, pmf_grid, res, xlimits, ylimits, bin_values, bin_values_bounds):
    #     # TODO: make sure parameters are all set properly


    #     assert len(pmf_grid.shape)==3, "PMF grid must have 3 dimensions"
    #     self.num_pmf_bins, self.num_rows, self.num_cols = pmf_grid.shape
    #     self.cell_dimensions = (res, res)
    #     self.xlimits = xlimits
    #     self.ylimits = ylimits

    #     self.pmf_grid = np.asarray(pmf_grid).astype(np.int8)
    #     self.bin_values = np.asarray(bin_values).astype(np.float32)
    #     self.bin_values_bounds = np.asarray(bin_values_bounds).astype(np.float32)
    #     assert bin_values[0]==0, "Assume minimum bin value is 0 for now"
    #     assert bin_values_bounds[0]==0, "Assume minimum traction is 0 for now"
    #     # self.pmf_grid_d = cuda.to_device(self.pmf_grid)
    #     self.bin_values_d = cuda.to_device(bin_values)
    #     self.bin_values_bounds_d = cuda.to_device(bin_values_bounds)
    #     self.res = res

    #     padded_pmf_grid, padded_xlimits, padded_ylimits = self.set_padding(self.pmf_grid, self.max_speed, self.dt, res,
    #                                                         xlimits, ylimits)

    #     self.pmf_grid_d = cuda.to_device(padded_pmf_grid)
    #     self.padded_xlimits = padded_xlimits
    #     self.padded_ylimits = padded_ylimits
    #     _, self.padded_num_rows, self.padded_num_cols = padded_pmf_grid.shape
    #     self.pmf_grid_initialized = True 

        

    def set_padding(self, pmf_grid, max_speed, dt, res, xlimits, ylimits):
        # Padd the nominal pmf grid with 0_traction components (assumed to be the first bin values)
        _, original_height, original_width = pmf_grid.shape
        self.pad_width = pad_width = int(np.ceil(max_speed*dt/res))
        self.padded_num_cols = self.num_cols+2*pad_width
        self.padded_num_rows = self.num_rows+2*pad_width
        padded_xlimits = np.array([xlimits[0]-pad_width*res, xlimits[0]+pad_width*res])
        padded_ylimits = np.array([ylimits[0]-pad_width*res, ylimits[0]+pad_width*res])

        padded_pmf_grid = np.zeros((self.num_pmf_bins, original_height+int(2*pad_width), original_width+int(2*pad_width)), dtype=np.int8)
        padded_pmf_grid[0] = 100 # Fill the probability mass associated with 0 traction
        padded_pmf_grid[:, pad_width:(pad_width+original_height), pad_width:(pad_width+original_width)] = pmf_grid
        
        return padded_pmf_grid, padded_xlimits, padded_ylimits


    def init_device_vars_before_sampling(self, det_dyn=False, seed=1):
        # num_samples = number of grids
        if not self.device_var_initialized:
            _, rows, cols = self.pmf_grid_d.shape
            if not det_dyn:
                self.rng_states_d = create_xoroshiro128p_states(self.TOTAL_THREADS, seed=seed)
                self.sample_grid_batch_d = cuda.device_array((self.NUM_GRID_SAMPLES, rows, cols), dtype=np.int8)
            else:
                self.rng_states_d = create_xoroshiro128p_states(self.THREAD_DIM[0]*self.THREAD_DIM[1], seed=seed)
                self.sample_grid_batch_d = cuda.device_array((1, rows, cols), dtype=np.int8)

            self.device_var_initialized = True
    
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

    def sample_grids(self, det_dyn=False):
        # Invoke the GPU kernels to sample from the PMF approximation
        if not det_dyn:
            self.sample_grids_numba[self.BLOCK_DIM, self.THREAD_DIM](
                self.sample_grid_batch_d, self.pmf_grid_d, self.rng_states_d,
                self.bin_values_d, self.bin_values_bounds_d
            )
        else:
            self.sample_grids_numba[(1,1), self.THREAD_DIM](
                self.sample_grid_batch_d, self.pmf_grid_d, self.rng_states_d,
                self.bin_values_d, self.bin_values_bounds_d
            )
        return self.sample_grid_batch_d


    def int8_grid_to_float32(self, int8grid):
        # int8 value between 0 and 100 represent some value within bin_values_bounds
        ratio = np.asarray(int8grid.copy()).astype(np.float32)/100.
        return ratio*(self.bin_values_bounds[1]-self.bin_values_bounds[0])+self.bin_values_bounds[0]

    """GPU kernels"""

    @staticmethod
    @cuda.jit(fastmath=True)
    def realized_states():
        # TODO: given the current optimal, compute the rollouts with sampled dynamics
        pass

    @staticmethod
    @cuda.jit(fastmath=True)
    def sample_grids_numba(grid_batch_d, pmf_grid_d, rng_states_d,
                     bin_values_d, bin_values_bounds_d):
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

        # Compute horizontal and vertical index range
        ri_start = min(tid_x*num_rows_entries_per_thread, grid_rows)
        ri_end = min(ri_start+num_rows_entries_per_thread, grid_rows)
        ci_start = min(tid_y*num_col_entries_per_thread, grid_cols)
        ci_end = min(ci_start+num_col_entries_per_thread, grid_cols)

        traction_range = bin_values_bounds_d[1]-bin_values_bounds_d[0]
        cum_pmf = np.int8(0)
        sampled_cum_pmf = np.int8(8)
        for ri in range(ri_start, ri_end):
            for ci in range(ci_start, ci_end):
                # Check which bin this belongs to
                rand_num = xoroshiro128p_uniform_float32(rng_states_d, thread_id)
                sampled_cum_pmf = rand_num*100.0
                cum_pmf = 0
                for bi in range(num_bins):
                    cum_pmf += pmf_grid_d[bi, ri, ci]
                    if sampled_cum_pmf <= cum_pmf:
                        grid_batch_d[block_id, ri, ci] = np.int8(100.*(bin_values_d[bi]-bin_values_bounds_d[0])/traction_range)
                        if grid_batch_d[block_id, ri, ci]<0:
                            print("<0")
                        break
        cuda.syncthreads()



# A deterministic grid with traction coefficients (can be generated by SDM)
class TractionGrid(object):
    def __init__(self, lin_traction, ang_traction, res=1.0, use_int8=False):
        if use_int8:
          # Use int 0-100 to represent values between 0 and 1
          self.lin_traction = (100*lin_traction).astype(np.int8)
          self.ang_traction = (100*ang_traction).astype(np.int8)
        else:
          self.lin_traction = lin_traction
          self.ang_traction = ang_traction
        self.res = res
        self.height, self.width = self.lin_traction.shape
        self.xlimits = (0, self.res*self.width)
        self.ylimits = (0, self.res*self.height)
        

    def get(self, x,y):
        # If within bounds, return queried value. Otherwise, return 0
        xi = int((x-self.xlimits[0])//self.res)
        yi = int((y-self.ylimits[0])//self.res)
        # print(yi, xi)
        if (xi<0) or (xi>=self.width) or (yi<0) or (yi>=self.height):
            return 0, 0
        else:
            return self.lin_traction[yi, xi], self.ang_traction[yi, xi]
    
    def get_grids(self):
        return self.lin_traction, self.ang_traction
