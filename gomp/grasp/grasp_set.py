"""
GraspSet: Generate and manage sets of IK-resolved grasp configurations.

From GOMP Section III & IV-D: The start/goal grasps form sets G_start and G_goal,
each parameterized by a rotation angle θ around the grasp axis. For each candidate
grasp and each θ, we compute an IK solution to get a joint configuration.
"""

import numpy as np
from typing import List, Optional, Tuple

from gomp.grasp.grasp import Grasp
from gomp.robot_adapter import RobotAdapter

import gomp  # noqa: F401
from wrs.basis import robot_math as rm


class GraspSet:
    """
    A set of grasp configurations resolved through IK for a given grasp.

    For GOMP, the start and goal poses are not fixed to a single configuration —
    they can vary over the rotational DOF of the grasp. This class precomputes
    IK solutions for the top-down pose and provides the constraint formulation
    matrices R_0 and ε_0 needed by the SQP.

    Parameters
    ----------
    grasp : Grasp
        The grasp with rotational DOF.
    robot : RobotAdapter
        Robot adapter for IK computation.
    """

    def __init__(self, grasp: Grasp, robot: RobotAdapter):
        self.grasp = grasp
        self.robot = robot
        self._ik_cache = {}  # theta -> jnt_values

    def get_ik(self, theta: float = 0.0,
               seed_jnt_values: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get IK solution for the grasp at rotation angle theta.

        Parameters
        ----------
        theta : float
            Rotation about the grasp axis.
        seed_jnt_values : np.ndarray or None
            Seed for IK solver.

        Returns
        -------
        jnt_values : np.ndarray or None
            Joint configuration, or None if IK fails.
        """
        # Check cache
        cache_key = round(theta, 6)
        if cache_key in self._ik_cache:
            return self._ik_cache[cache_key]

        pos, rotmat = self.grasp.rotated_pose(theta)
        result = self.robot.inverse_kinematics(pos, rotmat,
                                               seed_jnt_values=seed_jnt_values)
        if result is not None:
            self._ik_cache[cache_key] = result
        return result

    def get_topdown_ik(self, seed_jnt_values=None) -> Optional[np.ndarray]:
        """Get IK for the top-down (θ=0) grasp pose."""
        return self.get_ik(theta=0.0, seed_jnt_values=seed_jnt_values)

    def compute_constraint_rotation(self) -> np.ndarray:
        """
        Compute the rotation matrix R_0 that aligns one Jacobian component
        with the grasp rotational DOF axis.

        From Section IV-D: R_0 rotates the coordinate frame so that a single
        component of the Jacobian corresponds to the degree of freedom.

        Returns
        -------
        R_0 : np.ndarray, shape (6, 6)
            Block diagonal rotation: [R_3x3, 0; 0, R_3x3] that aligns
            the grasp axis with a coordinate axis.
        """
        # Find rotation that maps grasp axis to z-axis
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = self.grasp.axis

        if np.allclose(axis, z_axis) or np.allclose(axis, -z_axis):
            R3 = np.eye(3)
        else:
            cross = np.cross(axis, z_axis)
            cross_norm = np.linalg.norm(cross)
            dot = np.dot(axis, z_axis)
            R3 = rm.rotmat_from_axangle(cross / cross_norm,
                                        np.arccos(np.clip(dot, -1, 1)))

        # Build 6×6 block diagonal
        R_0 = np.zeros((6, 6))
        R_0[:3, :3] = R3
        R_0[3:, 3:] = R3
        return R_0

    def compute_epsilon(self, free_tol: float = 1e3,
                        pos_tol: float = 0.001,
                        orient_tol: float = 0.01) -> np.ndarray:
        """
        Compute the tolerance vector ε_0 for the grasp DOF constraint.

        From Section IV-D: ε_0 is a vector where the coefficient corresponding
        to the DOF (the grasp axis direction after R_0 rotation) is large,
        and remaining values are small.

        After R_0 rotation, the DOF aligns with the z-axis, so:
        - ε[5] (angular z) is large (free) when the grasp has rotational DOF
        - ε[5] stays tight when theta_min ≈ theta_max (single fixed pose)
        - Position components (ε[0:3]) use a tight positional tolerance
        - Orientation components (ε[3:6]) use a slightly looser tolerance
          for numerical stability of the linearized constraint

        Parameters
        ----------
        free_tol : float
            Large tolerance for the free DOF.
        pos_tol : float
            Tolerance for position components (meters). Default 1mm.
        orient_tol : float
            Tolerance for orientation components (radians). Default ~0.6°.

        Returns
        -------
        epsilon : np.ndarray, shape (6,)
            Tolerance vector.
        """
        epsilon = np.empty(6)
        epsilon[:3] = pos_tol      # position: tight (1mm)
        epsilon[3:] = orient_tol   # orientation: small but allows linearization error
        # After R_0 rotation, the grasp rotation axis aligns with z.
        # Only free the angular-z component if the grasp has a non-trivial
        # rotational DOF range; otherwise keep it tight (single fixed pose).
        theta_range = abs(self.grasp.theta_max - self.grasp.theta_min)
        if theta_range > 1e-6:
            epsilon[5] = free_tol  # Angular z is the rotation DOF
        return epsilon

    def compute_constraint_bounds(self, q_current: np.ndarray,
                                  R_0: np.ndarray,
                                  fast_kin=None) -> np.ndarray:
        """
        Compute the constraint bound vector b_0^{(k+1)} for SQP iteration.

        From Section IV-D, Equation (10):
            b_0^{(k+1)} = R_0 * (g_start^{z-} - p(q_0^{(k)}) + J^{(k)} * q_0^{(k)})

        The pose error g - p(q) is computed as a 6-vector:
        - Position: simple subtraction (R^3)
        - Orientation: log-map of R_target @ R_current^T (rotation vector)

        Parameters
        ----------
        q_current : np.ndarray, shape (n_dof,)
            Current joint configuration at this waypoint.
        R_0 : np.ndarray, shape (6, 6)
            The DOF-alignment rotation matrix.
        fast_kin : FastKinematics or None
            Pre-compiled fast kinematics. If provided, uses it for FK+Jacobian.

        Returns
        -------
        b : np.ndarray, shape (6,)
            Constraint bound vector.
        """
        # Current FK + Jacobian (use fast_kin if available)
        if fast_kin is not None:
            pos_current, rotmat_current, J = fast_kin.fk_and_jacobian(q_current)
        else:
            pos_current, rotmat_current = self.robot.forward_kinematics(q_current)
            J = self.robot.jacobian(q_current)

        # Target pose (top-down grasp)
        pos_target = self.grasp.pos
        rotmat_target = self.grasp.rotmat

        # Position error (R^3 subtraction — always exact)
        pos_error = pos_target - pos_current

        # Orientation error via robust log-map: log(R_target @ R_current^T)
        orient_error = self._logmap_rotation_error(rotmat_target, rotmat_current)

        # 6-vector: [position_error; orientation_error]
        pose_error = np.concatenate([pos_error, orient_error])

        # b = R_0 * (pose_error + J * q_current)
        b = R_0 @ (pose_error + J @ q_current)
        return b

    @staticmethod
    def _logmap_rotation_error(R_target: np.ndarray,
                               R_current: np.ndarray) -> np.ndarray:
        """
        Compute the rotation vector (axis-angle) error between two rotations.

        Uses the logarithmic map: log(R_target @ R_current^T).
        Handles three regimes to avoid numerical singularities:
        1. angle ≈ 0  → return zeros
        2. 0 < angle < π-δ → standard log-map via skew-symmetric extraction
        3. angle ≈ π  → extract axis from (R + I) to avoid sin(angle)→0

        Parameters
        ----------
        R_target : np.ndarray, shape (3, 3)
        R_current : np.ndarray, shape (3, 3)

        Returns
        -------
        orient_error : np.ndarray, shape (3,)
            Rotation vector (axis × angle).
        """
        R_err = R_target @ R_current.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle < 1e-10:
            # Near zero rotation — first-order approximation
            return np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ]) * 0.5

        if angle > np.pi - 1e-6:
            # Near π: sin(angle) ≈ 0, so the standard formula is unstable.
            # Use: R = I + 2 * (a @ a.T) - 2 * I  when angle=π
            #   => R + I = 2 * a @ a.T  (symmetric, rank-1)
            # The rotation axis is the column of (R+I) with largest norm.
            S = R_err + np.eye(3)
            col_norms = np.linalg.norm(S, axis=0)
            best_col = np.argmax(col_norms)
            axis = S[:, best_col]
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                return np.zeros(3)
            axis = axis / norm
            return axis * angle

        # Standard log-map: angle / (2 sin(angle)) * vex(R - R^T)
        sin_angle = np.sin(angle)
        return (angle / (2.0 * sin_angle)) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])

    def check_orientation_error(self, q: np.ndarray) -> float:
        """
        Compute the orientation error magnitude for diagnostics.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        angle_deg : float
            Orientation error in degrees.
        """
        _, rotmat_current = self.robot.forward_kinematics(q)
        R_err = self.grasp.rotmat @ rotmat_current.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
