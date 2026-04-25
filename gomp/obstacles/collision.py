"""
Linearized obstacle avoidance constraints for the SQP.

From GOMP Section IV-C: The obstacle avoidance constraint is linearized
using the robot's current configuration and Jacobian:

    z_obstacle - p(q_i^{(k)}) + J_z^{(k)} * q_i^{(k)} ≤ J_z^{(k)} * q_i^{(k+1)}

where J_z is the z-translation row of the Jacobian.

Performance: Uses FastKinematics for batched FK+Jacobian computation,
replacing the serial per-waypoint Python loop that was the #1 bottleneck
(87% of total planning time in profiling).
"""

import numpy as np
from typing import Tuple

from gomp.robot_adapter import RobotAdapter
from gomp.obstacles.depth_map import DepthMapObstacle


class ObstacleConstraint:
    """
    Computes linearized obstacle avoidance constraints for each waypoint.

    For each waypoint, the constraint ensures the end-effector stays above
    the depth map surface. The constraint is linearized around the current
    configuration and updated at each SQP iteration.

    Parameters
    ----------
    robot : RobotAdapter
        Robot model for FK and Jacobian.
    depth_map : DepthMapObstacle
        The obstacle depth map.
    safety_margin : float
        Additional clearance above the depth map surface (meters).
    fast_kin : FastKinematics or None
        Pre-compiled fast kinematics engine. If None, falls back to
        robot_adapter FK/Jacobian (slower).
    """

    def __init__(self, robot: RobotAdapter, depth_map: DepthMapObstacle,
                 safety_margin: float = 0.01, fast_kin=None):
        self.robot = robot
        self.depth_map = depth_map
        self.safety_margin = safety_margin
        self.fast_kin = fast_kin

    def compute_for_waypoint(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the linearized obstacle constraint for a single waypoint.

        The constraint is:
            z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)} ≤ J_z * q^{(k+1)}

        Rearranged for the QP: J_z * q ≥ z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)}

        Which becomes:  -J_z * q ≤ -(z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)})

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Current joint configuration at this waypoint.

        Returns
        -------
        J_z : np.ndarray, shape (n_dof,)
            The z-row of the Jacobian (linear velocity z-component).
        rhs : float
            The right-hand side of the linearized constraint:
            z_obstacle + margin - p_z(q^{(k)}) + J_z * q^{(k)}
        """
        if self.fast_kin is not None:
            pos, _, J = self.fast_kin.fk_and_jacobian(q)
        else:
            # Forward kinematics at current config
            pos, _ = self.robot.forward_kinematics(q)
            # Jacobian
            J = self.robot.jacobian(q)

        p_z = pos[2]  # z-position of end-effector
        p_x, p_y = pos[0], pos[1]

        # Get obstacle height at the (x, y) position
        z_obstacle = self.depth_map.get_obstacle_height(p_x, p_y)

        # Jacobian z-row (translation z)
        J_z = J[2, :]  # z-translation row

        # Right-hand side: z_obstacle + margin - p_z + J_z @ q
        rhs = z_obstacle + self.safety_margin - p_z + J_z @ q

        return J_z, rhs

    def compute_all_waypoints(self, waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearized obstacle constraints for all waypoints.

        Uses batch FK+Jacobian computation for significant speedup.

        Parameters
        ----------
        waypoints : np.ndarray, shape (H+1, n_dof)
            Joint configurations at each waypoint.

        Returns
        -------
        J_z_all : np.ndarray, shape (H+1, n_dof)
            J_z row for each waypoint.
        rhs_all : np.ndarray, shape (H+1,)
            Right-hand side for each waypoint.
        """
        H_plus_1, n = waypoints.shape

        if self.fast_kin is not None:
            # Batched computation using fast kinematics
            positions, _, jacobians = self.fast_kin.batch_fk_and_jacobian(waypoints)

            # Extract z-rows of Jacobians: shape (H+1, n)
            J_z_all = jacobians[:, 2, :]

            # Get obstacle heights for all xy positions
            z_obstacles = np.array([
                self.depth_map.get_obstacle_height(positions[i, 0], positions[i, 1])
                for i in range(H_plus_1)
            ])

            # Vectorized RHS computation
            p_z_all = positions[:, 2]
            # rhs = z_obstacle + margin - p_z + J_z @ q  (per waypoint)
            Jz_dot_q = np.sum(J_z_all * waypoints, axis=1)
            rhs_all = z_obstacles + self.safety_margin - p_z_all + Jz_dot_q

            return J_z_all, rhs_all
        else:
            # Fallback: serial computation
            J_z_all = np.zeros((H_plus_1, n))
            rhs_all = np.zeros(H_plus_1)
            for i in range(H_plus_1):
                J_z_all[i], rhs_all[i] = self.compute_for_waypoint(waypoints[i])
            return J_z_all, rhs_all

    def check_violation(self, q: np.ndarray) -> float:
        """
        Check how much a waypoint violates the obstacle constraint.

        Returns
        -------
        violation : float
            Positive if the constraint is violated (EE below obstacle).
            Negative if satisfied (EE above obstacle).
        """
        if self.fast_kin is not None:
            pos, _ = self.fast_kin.fk(q)
        else:
            pos, _ = self.robot.forward_kinematics(q)
        z_obstacle = self.depth_map.get_obstacle_height(pos[0], pos[1])
        return z_obstacle + self.safety_margin - pos[2]
