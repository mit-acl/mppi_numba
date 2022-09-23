#!/usr/bin/env python3
"""
Utils for visualizing semantic maps
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

class TDM_Visualizer(object):

    """Visualizer for a given traction distribution map."""

    PREFERRED_MAX_FIG_WIDTH = 12
    PREFERRED_MAX_FIG_HEIGHT = 8

    def __init__(self, tdm, tdm_contains_semantic_grid=True):
        
        if tdm_contains_semantic_grid:
        
            self.semantic_grid_initialized = tdm.semantic_grid_initialized
            self.semantic_grid = copy.deepcopy(tdm.semantic_grid)
            self.id2name = copy.deepcopy(tdm.id2name)
            self.name2terrain = copy.deepcopy(tdm.name2terrain)
            self.id2terrain_fn = copy.deepcopy(tdm.id2terrain_fn)
            self.id2rgb = {sid: self.id2terrain_fn(sid).rgb for sid in self.id2name}
            self.terrain2pmf = copy.deepcopy(tdm.terrain2pmf)
            self.semantic_grid_initialized = tdm.semantic_grid_initialized
            self.id2name[-1] = "Padding"
            self.id2rgb[-1] = (0,0,0,)
        else:
            self.semantic_grid_initialized= False
        
        
        self.ylimits = copy.deepcopy(tdm.ylimits)
        self.xlimits = copy.deepcopy(tdm.xlimits)

        
        self.cell_dimensions = copy.deepcopy(tdm.cell_dimensions)
        self.xlimits = copy.deepcopy(tdm.padded_xlimits)
        self.ylimits = copy.deepcopy(tdm.padded_ylimits)
        self.num_pmf_bins = copy.deepcopy(tdm.num_pmf_bins)
        self.bin_values = copy.deepcopy(tdm.bin_values)
        self.bin_values_bounds = copy.deepcopy(tdm.bin_values_bounds)
        
        # If padded, create new semantic_grid and update color (using black?)
        self.pad_width = tdm.pad_cells

        return_v = tdm.get_padded_grid_xy_dim()
        assert return_v is not None, "Cannot get padded grid dimension from TDM."
        self.num_rows, self.num_cols = copy.deepcopy(return_v)
        
        if tdm_contains_semantic_grid:
            original_semantic_grid = copy.deepcopy(self.semantic_grid)
            self.semantic_grid = -1*np.ones((self.num_rows, self.num_cols))
            self.semantic_grid[self.pad_width:(self.num_rows-self.pad_width), self.pad_width:(self.num_cols-self.pad_width)] = original_semantic_grid[:self.num_rows-2*self.pad_width, :self.num_cols - 2*self.pad_width]
            

    def draw(self, figsize=(10,10), ax=None, semantic_grid=None, id2rgb_map=None):
        if (not self.semantic_grid_initialized) and (semantic_grid is None) and (id2rgb_map is None):
            print("Semantic grid not initialized. Cannot invoke draw() function")
            return

        if figsize is None and ax is None:
            # Use user supplied axis
            fig, ax = self.draw_base_grid(ax=ax)
        elif figsize is None:
            self.figsize = self.calc_auto_figsize(self.xlimits, self.ylimits)
            fig, ax = self.draw_base_grid(self.figsize, ax=ax)
        else:
            fig, ax = self.draw_base_grid(figsize, ax=ax)

        if self.semantic_grid_initialized:
            self.draw_semantic_patches(ax)
        elif (semantic_grid is not None) and (id2rgb_map is not None):
            original_semantic_grid = copy.deepcopy(semantic_grid)
            semantic_grid = -1*np.ones((self.num_rows, self.num_cols))
            semantic_grid[self.pad_width:(self.num_rows-self.pad_width), self.pad_width:(self.num_cols-self.pad_width)] = original_semantic_grid[:self.num_rows-2*self.pad_width, :self.num_cols - 2*self.pad_width]
            self.draw_semantic_patches(ax, semantic_grid=semantic_grid, id2rgb_map=id2rgb_map)
        else:
            print("Colors not shown as semantic grid is not initialized.")
        return fig, ax
    
    def draw_base_grid(self, figsize, ax=None):
        cols, rows = self.num_cols, self.num_rows
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits

        width, height = self.cell_dimensions

        x = list(map(lambda i: minx + width*i, range(cols+1)))
        y = list(map(lambda i: miny + height*i, range(rows+1)))
        hlines = np.column_stack(np.broadcast_arrays(x[0], y, x[-1], y))
        vlines = np.column_stack(np.broadcast_arrays(x, y[0], x, y[-1]))
        lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
        line_collection = LineCollection(lines, color="black", linewidths=0.5, alpha=0.5)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax = plt.gca()
            ax.add_collection(line_collection)
            ax.set_xlim(x[0]-1, x[-1]+1)
            ax.set_ylim(y[0]-1, y[-1]+1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            return fig, ax
        else:
            ax.add_collection(line_collection)
            ax.set_xlim(x[0]-1, x[-1]+1)
            ax.set_ylim(y[0]-1, y[-1]+1)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            return plt.gcf(), ax

    def calc_auto_figsize(self, xlimits, ylimits):
        (minx, maxx) = xlimits
        (miny, maxy) = ylimits
        width, height = maxx - minx, maxy - miny
        if width > height:
            figsize = (self.PREFERRED_MAX_FIG_WIDTH, height * self.PREFERRED_MAX_FIG_WIDTH / width)
        else:
            figsize = (width * self.PREFERRED_MAX_FIG_HEIGHT / height, self.PREFERRED_MAX_FIG_HEIGHT)
        return figsize

    def cell_verts(self, ix, iy):
        width, height = self.cell_dimensions
        x, y = self.cell_xy(ix, iy)
        verts = [(x + ofx*0.5*width, y + ofy*0.5*height) for ofx, ofy in [(-1,-1),(-1,1),(1,1),(1,-1)]]
        return verts

    def cell_xy(self, ix, iy):
        """Returns the center xy point of the cell."""
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits
        width, height = self.cell_dimensions
        return minx + (ix+0.5) * width, miny + (iy+0.5) * height

    def draw_semantic_patches(self, ax, semantic_grid=None, id2rgb_map=None):
        if (semantic_grid is None) or (id2rgb_map is None):
            collection_recs = PolyCollection(self.get_all_cell_verts(), facecolors=self.get_terrain_rgbs())
        else:
            collection_recs = PolyCollection(self.get_all_cell_verts(semantic_grid=semantic_grid),
             facecolors=self.get_terrain_rgbs(id2rgb_map=id2rgb_map,semantic_grid=semantic_grid))

        ax.add_collection(collection_recs)
        
    def get_all_cell_verts(self, semantic_grid=None):
        if semantic_grid is None:
            num_rows, num_cols = self.semantic_grid.shape
        else:
            num_rows, num_cols = semantic_grid.shape
        return [self.cell_verts(ix, iy) for iy in range(num_rows) for ix in range(num_cols) ]
    
    def get_terrain_rgbs(self, id2rgb_map=None, semantic_grid=None):
        if (id2rgb_map is None) or (semantic_grid is None):
            return [self.id2rgb[sid] for sid in self.semantic_grid.reshape(-1)]
        else:
            return [id2rgb_map[sid] for sid in semantic_grid.reshape(-1)]
    


def vis_density(ax, density, terrain, vis_cvar_alpha=0.3, show_cvar=False, color='b', show_legend=True, title=None, hist_alpha=0.5, fontsize=12):
    """Visualization function for a given density."""

    cvar, thres = density.cvar(alpha=vis_cvar_alpha)
    if density.sample_initialized:
        ax.hist(density.samples, bins=100, density=True, color=color, alpha=hist_alpha, label=terrain.name)
    if show_cvar:
        ax.plot([thres, thres], [0,5], 'k--', label='{}-th Percentile'.format(int(vis_cvar_alpha*100.0)),
           linewidth=2)
    if density.sample_bounds is not None:
        ax.set_xlim(density.sample_bounds)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
        
        
    ax.set_xlabel("Traction", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    if show_legend:
        ax.legend(fontsize=fontsize)
    return ax

def vis_density_as_pmf(ax, density, terrain, num_bins, include_min_max=True, color='b', title=None, hist_alpha=0.5):
    """Visualization function for a given density approximated as PMF."""
    values, pmf = density.get_pmf(num_bins=num_bins, include_min_max=include_min_max)
    markerline, stemlines, baseline  = ax.stem(values, pmf, label=terrain.name)
    markerline.set_color(color)
    stemlines.set_color(color)
    baseline.set_color('r')
    if density.pmf_bounds is not None:
        ax.set_xlim(density.pmf_bounds)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Traction")
    ax.set_ylabel("PMF")
    ax.legend()
    return ax

