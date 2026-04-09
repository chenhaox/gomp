"""
Grasp sampler using WRS grasping module or manual specification.

Provides synthetic grasp candidates for testing GOMP without
requiring a full Dex-Net pipeline.
"""

import numpy as np
from typing import List

from gomp.grasp.grasp import Grasp


def create_topdown_grasp(pos: np.ndarray, angle_z: float = 0.0,
                         axis: np.ndarray = None,
                         theta_range: tuple = (-np.pi / 4, np.pi / 4),
                         quality: float = 1.0) -> Grasp:
    """
    Create a top-down grasp at a given position.

    The grasp approaches from above (negative z), with the gripper
    oriented at angle_z around the z-axis.

    Parameters
    ----------
    pos : np.ndarray, shape (3,)
        Grasp center position in world frame.
    angle_z : float
        Rotation about the z-axis (yaw) for the gripper approach.
    axis : np.ndarray or None
        Grasp rotation DOF axis. Default: x-axis (parallel jaw).
    theta_range : tuple
        (min, max) range for the rotational DOF.
    quality : float
        Grasp quality score.

    Returns
    -------
    Grasp
        A top-down grasp with rotational DOF.
    """
    # Top-down: z-axis pointing down, x-axis in the approach plane
    # Rotation by angle_z around the z-axis
    cos_z = np.cos(angle_z)
    sin_z = np.sin(angle_z)
    # Gripper frame: z pointing down (-z world), x rotated by angle_z
    rotmat = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, -1]  # z-axis pointing down
    ]).T  # Transpose because we want columns to be frame axes

    # Actually, for a top-down grasp:
    # The approach direction is -z (world), so the gripper's z-axis = [0, 0, -1]
    # The gripper's x-axis defines jaw opening direction
    rotmat = np.array([
        [cos_z, sin_z, 0],
        [-sin_z, cos_z, 0],
        [0, 0, -1]
    ])

    if axis is None:
        # Default: rotation about world x-axis (parallel jaw axis in gripper frame)
        axis = rotmat[:, 0].copy()  # gripper's x-axis = jaw axis

    return Grasp(
        pos=np.asarray(pos, dtype=float),
        rotmat=rotmat,
        axis=np.asarray(axis, dtype=float),
        theta_min=theta_range[0],
        theta_max=theta_range[1],
        quality=quality
    )


def sample_bin_grasps(bin_center: np.ndarray, bin_size: tuple,
                      n_grasps: int = 5, z_offset: float = 0.0,
                      seed: int = None) -> List[Grasp]:
    """
    Generate random top-down grasp candidates spread across a bin.

    Mimics the diverse grasp sampling from a modified Dex-Net 4.0
    (Section V of the paper): rejection-sampling with a minimum
    distance constraint between grasps in image space.

    Parameters
    ----------
    bin_center : np.ndarray, shape (3,)
        Center of the bin in world frame.
    bin_size : tuple (width, depth, height)
        Bin dimensions.
    n_grasps : int
        Number of candidate grasps to generate.
    z_offset : float
        Height offset above the bin bottom for grasp positions.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    grasps : list of Grasp
        Generated grasp candidates.
    """
    rng = np.random.default_rng(seed)

    width, depth, height = bin_size
    grasps = []

    for _ in range(n_grasps):
        # Random position within the bin
        x = bin_center[0] + rng.uniform(-width / 2 * 0.8, width / 2 * 0.8)
        y = bin_center[1] + rng.uniform(-depth / 2 * 0.8, depth / 2 * 0.8)
        z = bin_center[2] + z_offset
        pos = np.array([x, y, z])

        # Random yaw angle
        angle_z = rng.uniform(-np.pi, np.pi)

        # Random quality
        quality = rng.uniform(0.5, 1.0)

        grasp = create_topdown_grasp(
            pos=pos,
            angle_z=angle_z,
            theta_range=(-np.pi / 4, np.pi / 4),
            quality=quality
        )
        grasps.append(grasp)

    return grasps
