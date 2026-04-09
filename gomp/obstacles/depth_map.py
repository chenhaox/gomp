"""
Depth-map based obstacle model for GOMP.

From GOMP Section IV-C: pick-and-place robots can operate with a simplified
task-specific obstacle model — specifically a depth map. This avoids the
complexities of GJK/EPA collision detection.

The depth map represents the obstacle surface as a 2D grid of z-heights.
For a bin-picking scenario, the obstacles are the bin walls and surrounding
workspace elements.
"""

import numpy as np
from typing import Tuple


class DepthMapObstacle:
    """
    A 2D depth map representing obstacle heights in the workspace.

    The depth map covers a rectangular region in the xy-plane, with each
    cell storing the z-height of the obstacle surface at that location.
    The robot must keep its end-effector (and optionally other links)
    above this height at all times.

    Parameters
    ----------
    grid : np.ndarray, shape (nx, ny)
        2D array of z-heights (meters).
    resolution : float
        Grid resolution in meters per cell.
    origin : np.ndarray, shape (2,)
        World-frame (x, y) position of the grid's (0, 0) corner.
    """

    def __init__(self, grid: np.ndarray, resolution: float,
                 origin: np.ndarray):
        self.grid = grid.astype(float)
        self.resolution = resolution
        self.origin = np.asarray(origin, dtype=float)
        self.nx, self.ny = grid.shape

    def get_obstacle_height(self, x: float, y: float) -> float:
        """
        Look up the obstacle z-height at world position (x, y).

        Uses bilinear interpolation for sub-cell positions.
        Returns 0.0 if the position is outside the grid.

        Parameters
        ----------
        x, y : float
            World-frame position.

        Returns
        -------
        z_obstacle : float
            Height of the obstacle surface.
        """
        # Convert to grid coordinates
        gx = (x - self.origin[0]) / self.resolution
        gy = (y - self.origin[1]) / self.resolution

        # Check bounds
        if gx < 0 or gx >= self.nx - 1 or gy < 0 or gy >= self.ny - 1:
            return 0.0  # No obstacle outside the grid

        # Bilinear interpolation
        ix = int(gx)
        iy = int(gy)
        fx = gx - ix
        fy = gy - iy

        z00 = self.grid[ix, iy]
        z10 = self.grid[ix + 1, iy]
        z01 = self.grid[ix, iy + 1]
        z11 = self.grid[ix + 1, iy + 1]

        z = (z00 * (1 - fx) * (1 - fy) +
             z10 * fx * (1 - fy) +
             z01 * (1 - fx) * fy +
             z11 * fx * fy)
        return z

    def get_obstacle_height_batch(self, xy: np.ndarray) -> np.ndarray:
        """
        Batch lookup of obstacle heights.

        Parameters
        ----------
        xy : np.ndarray, shape (N, 2)
            Array of (x, y) positions.

        Returns
        -------
        z : np.ndarray, shape (N,)
            Obstacle heights.
        """
        return np.array([self.get_obstacle_height(p[0], p[1]) for p in xy])

    @classmethod
    def create_bin_obstacle(cls, bin_center: np.ndarray,
                            bin_size: Tuple[float, float, float],
                            wall_height: float,
                            resolution: float = 0.01,
                            margin: float = 0.05) -> 'DepthMapObstacle':
        """
        Create a depth map for a rectangular bin with walls.

        The bin interior has z_obstacle = 0 (or a small value),
        while the walls and exterior have z = wall_height.

        Parameters
        ----------
        bin_center : np.ndarray, shape (3,)
            World-frame center of the bin.
        bin_size : tuple (width, depth, height)
            Inner dimensions of the bin.
        wall_height : float
            Height of the bin walls above the table.
        resolution : float
            Grid cell size in meters.
        margin : float
            Extra area around the bin to include in the grid.

        Returns
        -------
        DepthMapObstacle
            The depth map obstacle.
        """
        width, depth, height = bin_size
        # Grid covers bin + margin on all sides
        total_w = width + 2 * margin
        total_d = depth + 2 * margin

        nx = int(total_w / resolution) + 1
        ny = int(total_d / resolution) + 1

        origin = np.array([
            bin_center[0] - total_w / 2,
            bin_center[1] - total_d / 2
        ])

        # Initialize grid with wall height everywhere
        grid = np.full((nx, ny), wall_height)

        # Carve out the bin interior (z = 0 inside)
        for i in range(nx):
            for j in range(ny):
                x = origin[0] + i * resolution
                y = origin[1] + j * resolution
                in_x = abs(x - bin_center[0]) < width / 2
                in_y = abs(y - bin_center[1]) < depth / 2
                if in_x and in_y:
                    grid[i, j] = 0.0  # Inside the bin, no obstacle

        return cls(grid=grid, resolution=resolution, origin=origin)

    @classmethod
    def create_flat_obstacle(cls, height: float = 0.0,
                             center: np.ndarray = None,
                             size: float = 2.0,
                             resolution: float = 0.01) -> 'DepthMapObstacle':
        """
        Create a flat obstacle (table surface) at a given height.

        Parameters
        ----------
        height : float
            Height of the flat surface.
        center : np.ndarray
            Center of the grid. Default: origin.
        size : float
            Side length of the square grid.
        resolution : float
            Grid resolution.

        Returns
        -------
        DepthMapObstacle
        """
        if center is None:
            center = np.zeros(2)

        nx = int(size / resolution) + 1
        ny = nx
        origin = center - size / 2
        grid = np.full((nx, ny), height)
        return cls(grid=grid, resolution=resolution, origin=origin)
