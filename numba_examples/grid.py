from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Ellipse


class WrongGridFormat(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return "Wrong grid format. Use 0 for free region and 1 for obstacles. Use the same number of columns always"
    
PREFERRED_MAX_FIG_WIDTH = 12
PREFERRED_MAX_FIG_HEIGHT = 8



class Grid(object):
    def __init__(self, num_cols=10, num_rows=10, xy_limits=None, figsize=None):
        self.generate_grid(num_cols, num_rows)

        if not xy_limits:
            xy_limits = (0, num_cols), (0, num_rows)
        self.set_xy_limits(*xy_limits)
        if not figsize:
            figsize = self.calc_auto_figsize(xy_limits)
        self.figsize = figsize
        
    @property
    def size(self):
        return self.grid_array.shape

    def generate_grid(self, num_cols, num_rows):
        self.grid_array = np.zeros([num_cols, num_rows])

    def set_xy_limits(self, xlimits, ylimits):
        num_cols, num_rows = self.size
        if not isinstance(xlimits,tuple) or not len(xlimits)==2 \
           or not isinstance(ylimits,tuple) or not len(ylimits)==2 \
           or not xlimits[0] < xlimits[1] or not ylimits[0] < ylimits[1]:
            raise ValueError('Specified xlimits or ylimits are not valid.')
        self.xlimits, self.ylimits = xlimits, ylimits
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits
        self.cell_dimensions = (maxx-minx) / num_cols, (maxy-miny) / num_rows

    def calc_auto_figsize(self, xy_limits):
        (minx, maxx), (miny, maxy) = xy_limits
        width, height = maxx - minx, maxy - miny
        if width > height:
            figsize = (PREFERRED_MAX_FIG_WIDTH, height * PREFERRED_MAX_FIG_WIDTH / width)
        else:
            figsize = (width * PREFERRED_MAX_FIG_HEIGHT / height, PREFERRED_MAX_FIG_HEIGHT)
        return figsize

    @classmethod
    def create_from_file(grid_class, grid_file, xy_limits=None, figsize=None):
        gfile = open(grid_file)
        grid = grid_class.create_from_str(gfile.read(), xy_limits=xy_limits, figsize=figsize)
        gfile.close()
        return grid
    @classmethod
    def create_from_str(grid_class, grid_str, xy_limits=None, figsize=None):
        lines = list(map(str.split, filter(lambda s:not s.startswith('#') and len(s)>0, map(str.strip,grid_str.split('\n')))))
        num_rows = len(lines)
        num_cols = len(lines[0])

        for line in lines:
            if not num_cols == len(line):
                raise WrongGridFormat
        grid_array = np.zeros([num_cols,num_rows])
        for row in range(num_rows):
            for col in range(num_cols):
                value = lines[row][col]
                if not value == '0' and not value =='1':
                    raise WrongGridFormat
                grid_array[col,num_rows-1 - row] = value

        grid = grid_class(num_cols, num_rows, xy_limits, figsize)
        grid.grid_array = grid_array

        return grid

    def add_random_obstacles(self, num_obs):
        """Clear grid and add random obstacles"""
        free_idx = list(zip(*np.where(self.grid_array == 0)))
        num_free = len(free_idx)
        num_obs = min(num_free, num_obs)
        for i in range(num_obs):
            obs_idx = np.random.randint(0, num_free-1)
            self.grid_array[free_idx.pop(obs_idx)] = 1
            num_free = num_free - 1

    def mark_obstacle_cell(self, x, y):
        self.grid_array[x, y] = 1

    def mark_free_cell(self, x, y):
        self.grid_array[x, y] = 0

    def clear(self):
        self.grid_array = np.zeros([self.num_cols, self.num_rows])

    def get_obstacles(self):
        return list(zip(*np.where(self.grid_array > 0)))

    def cell_xy(self, ix, iy):
        """Returns the center xy point of the cell."""
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits
        width, height = self.cell_dimensions
        return minx + (ix+0.5) * width, miny + (iy+0.5) * height

    def cell_verts(self, ix, iy):
        width, height = self.cell_dimensions
        x, y = self.cell_xy(ix, iy)
        verts = [(x + ofx*0.5*width, y + ofy*0.5*height) for ofx, ofy in [(-1,-1),(-1,1),(1,1),(1,-1)]]
        return verts

    def export_to_dict(self):
        export_dict = {}
        export_dict['grid'] = self.grid_array.tolist()
        return export_dict

    def load_from_dict(self, grid_dict):
        self.grid_array = np.array(grid_dict['grid'])

    def draw(self):
        cols, rows = self.size
        minx, maxx = self.xlimits
        miny, maxy = self.ylimits

        width, height = self.cell_dimensions

        x = list(map(lambda i: minx + width*i, range(cols+1)))
        y = list(map(lambda i: miny + height*i, range(rows+1)))

        f = plt.figure(figsize=self.figsize)

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
        self.draw_obstacles(plt.gca())

        return plt.gca()

    def draw_obstacles(self, axes):
        verts = [self.cell_verts(ix, iy) for ix,iy in self.get_obstacles()]
        collection_recs = PolyCollection(verts, facecolors='r')
        axes.add_collection(collection_recs)

    def draw_cell_circle(self, axes, xy, size=0.5, **kwargs):
        ix, iy = xy
        x, y = self.cell_xy(ix, iy)
        xr, yr = 0.5 * self.cell_dimensions[0], 0.5 * self.cell_dimensions[1]
        axes.add_patch(Ellipse((x,y), xr, yr, **kwargs))

    def draw_path(self, axes, path, *args, **kwargs):
        xy_coords = list(map(lambda idx: self.cell_xy(*idx), path))
        xx, yy = list(zip(*xy_coords))
        axes.plot(xx, yy, *args, **kwargs)


