"""
Grasp representation with rotational degree of freedom.

From GOMP Section III: A parallel-jaw grasp has a rotational DOF
around the grasp axis (the vector connecting the two contact points).
This allows the optimizer to rotate the grasp to find faster motions.

For suction grippers, the DOF is around the contact normal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

import gomp  # noqa: F401
from wrs.basis import robot_math as rm


@dataclass
class Grasp:
    """
    A grasp pose with an associated rotational degree of freedom.

    The grasp is defined by:
    - A top-down pose (pos, rotmat) in SE(3), denoted g^{z-} in the paper
    - An axis of rotation (the grasp axis for parallel-jaw, or contact normal for suction)
    - A range of allowed rotation angles around that axis

    The set of all valid grasps forms a 1-DOF family:
        G = { R_a(θ) · g^{z-} | θ ∈ [θ_min, θ_max] }

    Attributes
    ----------
    pos : np.ndarray, shape (3,)
        Position of the grasp center in world frame.
    rotmat : np.ndarray, shape (3, 3)
        Rotation matrix of the top-down grasp orientation.
    axis : np.ndarray, shape (3,)
        Unit vector defining the rotational DOF axis.
    theta_min : float
        Minimum rotation angle (rad). Default: -π/2.
    theta_max : float
        Maximum rotation angle (rad). Default: π/2.
    quality : float
        Grasp quality score (e.g., from Dex-Net). Higher is better.
    """
    pos: np.ndarray
    rotmat: np.ndarray
    axis: np.ndarray
    theta_min: float = -np.pi / 2
    theta_max: float = np.pi / 2
    quality: float = 1.0

    def __post_init__(self):
        """Normalize the axis vector."""
        self.axis = self.axis / np.linalg.norm(self.axis)

    def rotated_pose(self, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotation R_a(θ) about the grasp axis.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        pos : np.ndarray, shape (3,)
            Grasp position (unchanged by rotation about the grasp center).
        rotmat : np.ndarray, shape (3, 3)
            Rotated orientation matrix.
        """
        R_theta = rm.rotmat_from_axangle(self.axis, theta)
        return self.pos.copy(), R_theta @ self.rotmat

    def sample_poses(self, n_samples: int = 10) -> list:
        """
        Sample n poses evenly across the rotational DOF range.

        Returns
        -------
        poses : list of (pos, rotmat, theta) tuples
        """
        thetas = np.linspace(self.theta_min, self.theta_max, n_samples)
        poses = []
        for theta in thetas:
            pos, rotmat = self.rotated_pose(theta)
            poses.append((pos, rotmat, theta))
        return poses

    @property
    def homomat(self) -> np.ndarray:
        """Return the 4×4 homogeneous transformation matrix for the top-down pose."""
        mat = np.eye(4)
        mat[:3, :3] = self.rotmat
        mat[:3, 3] = self.pos
        return mat

    def rotated_homomat(self, theta: float) -> np.ndarray:
        """Return the 4×4 homogeneous matrix for a rotated grasp."""
        pos, rotmat = self.rotated_pose(theta)
        mat = np.eye(4)
        mat[:3, :3] = rotmat
        mat[:3, 3] = pos
        return mat
