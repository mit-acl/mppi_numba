#!/usr/bin/env python3
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Ellipse

class TDM_Visualizer(object):


    PREFERRED_MAX_FIG_WIDTH = 12
    PREFERRED_MAX_FIG_HEIGHT = 8

    def __init__(self, tdm):
        self.semantic_grid_initialized = tdm.semantic_grid_initialized
        self.num_rows = tdm.padded_num_rows
        self.num_cols = tdm.padded_num_cols
        self.ylimits = copy.deepcopy(tdm.ylimits)
        self.xlimits = copy.deepcopy(tdm.xlimits)

        self.semantic_grid = copy.deepcopy(tdm.semantic_grid)
        self.id2name = copy.deepcopy(tdm.id2name)
        self.name2terrain = copy.deepcopy(tdm.name2terrain)
        self.id2terrain_fn = copy.deepcopy(tdm.id2terrain_fn)
        self.id2rgb = {sid: self.id2terrain_fn(sid).rgb for sid in self.id2name}
        self.terrain2pmf = copy.deepcopy(tdm.terrain2pmf)
        self.semantic_grid_initialized = tdm.semantic_grid_initialized
        self.cell_dimensions = copy.deepcopy(tdm.cell_dimensions)
        self.xlimits = copy.deepcopy(tdm.padded_xlimits)
        self.ylimits = copy.deepcopy(tdm.padded_ylimits)
        self.num_pmf_bins = copy.deepcopy(tdm.num_pmf_bins)
        self.bin_values = copy.deepcopy(tdm.bin_values)
        self.bin_values_bounds = copy.deepcopy(tdm.bin_values_bounds)
        
        # If padded, create new semantic_grid and update color (using black?)
        self.pad_width = tdm.pad_width
        self.id2name[-1] = "Padding"
        self.id2rgb[-1] = (0,0,0,)

        original_semantic_grid = copy.deepcopy(self.semantic_grid)
        self.semantic_grid = -1*np.ones((self.num_rows, self.num_cols))
        self.semantic_grid[self.pad_width:(self.num_rows-self.pad_width), self.pad_width:(self.num_cols-self.pad_width)] = original_semantic_grid
            

    def draw(self, figsize=(10,10)):
        if not self.semantic_grid_initialized:
            print("Semantic grid not initialized. Cannot invoke draw() function")
            return

        if figsize is None:
            self.figsize = self.calc_auto_figsize(self.xlimits, self.ylimits)
            fig, ax = self.draw_base_grid(self.figsize)
        else:
            fig, ax = self.draw_base_grid(figsize)

        if self.semantic_grid_initialized:
            self.draw_semantic_patches(ax)
        else:
            print("Colors not shown as semantic grid is not initialized.")
        return fig, ax
    
    def draw_base_grid(self, figsize):
        cols, rows = self.num_cols, self.num_rows
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits

        width, height = self.cell_dimensions

        x = list(map(lambda i: minx + width*i, range(cols+1)))
        y = list(map(lambda i: miny + height*i, range(rows+1)))

        fig = plt.figure(figsize=figsize)

        hlines = np.column_stack(np.broadcast_arrays(x[0], y, x[-1], y))
        vlines = np.column_stack(np.broadcast_arrays(x, y[0], x, y[-1]))
        lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
        line_collection = LineCollection(lines, color="black", linewidths=0.5)
        ax = plt.gca()
        ax.add_collection(line_collection)
        ax.set_xlim(x[0]-1, x[-1]+1)
        ax.set_ylim(y[0]-1, y[-1]+1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        return fig, plt.gca()

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

    def draw_semantic_patches(self, ax):
        collection_recs = PolyCollection(self.get_all_cell_verts(), facecolors=self.get_terrain_rgbs())
        ax.add_collection(collection_recs)
        
    def get_all_cell_verts(self):
        num_rows, num_cols = self.semantic_grid.shape
        return [self.cell_verts(ix, iy) for iy in range(num_rows) for ix in range(num_cols) ]
    
    def get_terrain_rgbs(self):
        return [self.id2rgb[sid] for sid in self.semantic_grid.reshape(-1)]
    


def vis_density(ax, density, terrain, vis_cvar_alpha, color='b', title=None, hist_alpha=0.5):
    cvar, thres = density.cvar(alpha=vis_cvar_alpha)
    if density.sample_initialized:
        ax.hist(density.samples, bins=100, density=True, color=color, alpha=hist_alpha, label=terrain.name)
    ax.plot([thres, thres], [0,5], 'k--', label='{}-th Percentile'.format(int(vis_cvar_alpha*100.0)),
           linewidth=2)
    if density.sample_bounds is not None:
        ax.set_xlim(density.sample_bounds)
    if title is not None:
        ax.set_title(title)
        
        
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Density")
        
    ax.legend()
    return ax

def vis_density_as_pmf(ax, density, terrain, num_bins, include_min_max=True, color='b', title=None, hist_alpha=0.5):
    values, pmf = density.get_pmf(num_bins=num_bins, include_min_max=include_min_max)
    markerline, stemlines, baseline  = ax.stem(values, pmf, label=terrain.name)
    markerline.set_color(color)
    stemlines.set_color(color)
    baseline.set_color('r')
    if density.pmf_bounds is not None:
        ax.set_xlim(density.pmf_bounds)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Density")
    ax.legend()
    return ax

