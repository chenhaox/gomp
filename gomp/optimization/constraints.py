"""
Constraint builder for the GOMP QP formulation.

Implements all constraint types from GOMP Sections IV-B, IV-C, IV-D as
linear inequalities Ax ≤ b (or equality constraints encoded as pairs).

Constraint types:
1. Joint position limits (Eq. 2): q_min ≤ q_i ≤ q_max
2. Velocity limits (Eq. 3): v_min ≤ v_i ≤ v_max
3. Acceleration limits (Eq. 4): -a_max ≤ (v_{i+1} - v_i)/t_step ≤ a_max
4. Dynamics (Sec. IV-B): q_{i+1} = q_i + v_i * t_step
5. Zero velocity at endpoints: v_0 = 0, v_H = 0
6. Obstacle avoidance (Sec. IV-C): linearized depth-map constraint
7. Start/goal grasp constraints (Sec. IV-D): linearized with DOF
8. Trust regions: box constraints around current solution
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional

from gomp.optimization.qp_builder import QPBuilder
from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.collision import ObstacleConstraint


class ConstraintBuilder:
    """
    Builds the A matrix and (l, u) bound vectors for OSQP.

    OSQP uses the form: l ≤ Ax ≤ u
    This is more general than Ax ≤ b, and can encode both
    inequality and equality constraints.

    Parameters
    ----------
    qp : QPBuilder
        The QP layout manager.
    t_step : float
        Time step between waypoints.
    robot : RobotAdapter
        Robot model for FK/Jacobian.
    """

    def __init__(self, qp: QPBuilder, t_step: float, robot: RobotAdapter):
        self.qp = qp
        self.t_step = t_step
        self.robot = robot
        self.n = qp.n
        self.H = qp.H
        self.m = qp.n_waypoints  # H + 1

    def build_static_constraints(self) -> tuple:
        """
        Build constraints that don't change between SQP iterations:
        - Joint position limits
        - Velocity limits
        - Acceleration limits
        - Dynamics constraints
        - Zero velocity at endpoints

        Returns
        -------
        A : scipy.sparse.csc_matrix
            Constraint matrix.
        l : np.ndarray
            Lower bounds.
        u : np.ndarray
            Upper bounds.
        """
        constraints = []

        # 1. Joint position limits: q_min ≤ q_i ≤ q_max for all waypoints
        constraints.append(self._joint_position_limits())

        # 2. Velocity limits: -v_max ≤ v_i ≤ v_max for all waypoints
        constraints.append(self._velocity_limits())

        # 3. Acceleration limits: -a_max ≤ (v_{i+1} - v_i)/t_step ≤ a_max
        constraints.append(self._acceleration_limits())

        # 4. Dynamics: q_{i+1} = q_i + v_i * t_step
        constraints.append(self._dynamics_constraints())

        # 5. Zero velocity at endpoints: v_0 = 0, v_H = 0
        constraints.append(self._zero_velocity_endpoints())

        return self._stack_constraints(constraints)

    def build_dynamic_constraints(self, waypoints: np.ndarray,
                                  obstacle_constraint: Optional[ObstacleConstraint],
                                  start_grasp_set: Optional[GraspSet],
                                  goal_grasp_set: Optional[GraspSet],
                                  trust_region: float = 0.5) -> tuple:
        """
        Build constraints that are updated at each SQP iteration:
        - Obstacle avoidance (linearized)
        - Start/goal grasp constraints (linearized)
        - Trust regions

        Parameters
        ----------
        waypoints : np.ndarray, shape (H+1, n)
            Current waypoint configurations (linearization point).
        obstacle_constraint : ObstacleConstraint or None
            Obstacle model.
        start_grasp_set : GraspSet or None
            Start grasp set with DOF.
        goal_grasp_set : GraspSet or None
            Goal grasp set with DOF.
        trust_region : float
            Trust region radius (infinity norm).

        Returns
        -------
        A : scipy.sparse.csc_matrix
        l : np.ndarray
        u : np.ndarray
        """
        constraints = []

        # 6. Obstacle avoidance
        if obstacle_constraint is not None:
            constraints.append(
                self._obstacle_constraints(waypoints, obstacle_constraint)
            )

        # 7. Start grasp constraint
        if start_grasp_set is not None:
            constraints.append(
                self._grasp_constraint(waypoints[0], start_grasp_set, waypoint_idx=0)
            )

        # 8. Goal grasp constraint
        if goal_grasp_set is not None:
            constraints.append(
                self._grasp_constraint(waypoints[-1], goal_grasp_set,
                                       waypoint_idx=self.H)
            )

        # 9. Trust regions
        constraints.append(self._trust_region_constraints(waypoints, trust_region))

        return self._stack_constraints(constraints)

    # ----- Individual constraint builders -----

    def _joint_position_limits(self) -> tuple:
        """q_min ≤ q_i ≤ q_max for each waypoint."""
        n = self.n
        m = self.m
        n_vars = self.qp.n_vars

        # Identity for q portion, zero for v portion
        rows = []
        for i in range(m):
            row = sp.lil_matrix((n, n_vars))
            row[:, self.qp.q_idx(i)] = sp.eye(n)
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        l = np.tile(self.robot.q_min, m)
        u = np.tile(self.robot.q_max, m)
        return A, l, u

    def _velocity_limits(self) -> tuple:
        """v_min ≤ v_i ≤ v_max for each waypoint."""
        n = self.n
        m = self.m
        n_vars = self.qp.n_vars

        rows = []
        for i in range(m):
            row = sp.lil_matrix((n, n_vars))
            row[:, self.qp.v_idx(i)] = sp.eye(n)
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        l = np.tile(-self.robot.v_max, m)
        u = np.tile(self.robot.v_max, m)
        return A, l, u

    def _acceleration_limits(self) -> tuple:
        """
        -a_max ≤ (v_{i+1} - v_i) / t_step ≤ a_max

        Rearranged: -a_max * t_step ≤ v_{i+1} - v_i ≤ a_max * t_step
        """
        n = self.n
        H = self.H
        n_vars = self.qp.n_vars

        rows = []
        for i in range(H):  # H differences between H+1 velocities
            row = sp.lil_matrix((n, n_vars))
            # v_{i+1} - v_i
            row[:, self.qp.v_idx(i)] = -sp.eye(n)
            row[:, self.qp.v_idx(i + 1)] = sp.eye(n)
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        a_bound = self.robot.a_max * self.t_step
        l = np.tile(-a_bound, H)
        u = np.tile(a_bound, H)
        return A, l, u

    def _dynamics_constraints(self) -> tuple:
        """
        q_{i+1} = q_i + v_i * t_step  (equality)

        Encoded as: q_{i+1} - q_i - v_i * t_step = 0
        So: l = u = 0
        """
        n = self.n
        H = self.H
        n_vars = self.qp.n_vars

        rows = []
        for i in range(H):
            row = sp.lil_matrix((n, n_vars))
            # q_{i+1} - q_i - v_i * t_step = 0
            row[:, self.qp.q_idx(i)] = -sp.eye(n)
            row[:, self.qp.q_idx(i + 1)] = sp.eye(n)
            row[:, self.qp.v_idx(i)] = -self.t_step * sp.eye(n)
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        l = np.zeros(H * n)
        u = np.zeros(H * n)
        return A, l, u

    def _zero_velocity_endpoints(self) -> tuple:
        """v_0 = 0 and v_H = 0 (equality)."""
        n = self.n
        n_vars = self.qp.n_vars

        # Two blocks of n constraints each
        A = sp.lil_matrix((2 * n, n_vars))
        # v_0 = 0
        A[:n, self.qp.v_idx(0)] = sp.eye(n)
        # v_H = 0
        A[n:, self.qp.v_idx(self.H)] = sp.eye(n)

        A = A.tocsc()
        l = np.zeros(2 * n)
        u = np.zeros(2 * n)
        return A, l, u

    def _obstacle_constraints(self, waypoints: np.ndarray,
                              obs: ObstacleConstraint) -> tuple:
        """
        Linearized obstacle avoidance for each waypoint.

        From Section IV-C:
            z_obs - p_z(q^{(k)}) + J_z * q^{(k)} ≤ J_z * q^{(k+1)}

        In QP form: J_z * q_i ≥ rhs  →  -J_z * q_i ≤ -rhs
        Or equivalently: rhs ≤ J_z * q_i ≤ +inf
        """
        n = self.n
        m = self.m
        n_vars = self.qp.n_vars

        J_z_all, rhs_all = obs.compute_all_waypoints(waypoints)

        rows = []
        for i in range(m):
            row = sp.lil_matrix((1, n_vars))
            row[0, self.qp.q_idx(i)] = J_z_all[i]
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        l = rhs_all  # Lower bound: J_z * q ≥ rhs
        u = np.full(m, np.inf)  # No upper bound
        return A, l, u

    def _grasp_constraint(self, q_current: np.ndarray,
                          grasp_set: GraspSet,
                          waypoint_idx: int) -> tuple:
        """
        Linearized start/goal grasp constraint with rotational DOF.

        From Section IV-D:
            b_0 - ε ≤ R_0 * J * q ≤ b_0 + ε

        where R_0 aligns the Jacobian with the grasp DOF axis,
        and ε is large for the DOF axis, small otherwise.
        """
        n = self.n
        n_vars = self.qp.n_vars

        # Compute constraint matrices
        R_0 = grasp_set.compute_constraint_rotation()  # 6×6
        epsilon = grasp_set.compute_epsilon()  # 6
        b = grasp_set.compute_constraint_bounds(q_current, R_0)  # 6

        # Jacobian at current config
        J = self.robot.jacobian(q_current)  # 6×n

        # Constraint: b - ε ≤ R_0 * J * q ≤ b + ε
        # A_row = R_0 @ J, applied to the appropriate waypoint's q
        A_coeff = R_0 @ J  # 6×n

        A = sp.lil_matrix((6, n_vars))
        A[:, self.qp.q_idx(waypoint_idx)] = A_coeff
        A = A.tocsc()

        l = b - epsilon
        u = b + epsilon
        return A, l, u

    def _trust_region_constraints(self, waypoints: np.ndarray,
                                  delta: float) -> tuple:
        """
        Trust region: ||q_i^{(k+1)} - q_i^{(k)}||_∞ ≤ δ

        In QP form: q_i^{(k)} - δ ≤ q_i ≤ q_i^{(k)} + δ
        """
        n = self.n
        m = self.m
        n_vars = self.qp.n_vars

        rows = []
        for i in range(m):
            row = sp.lil_matrix((n, n_vars))
            row[:, self.qp.q_idx(i)] = sp.eye(n)
            rows.append(row)

        A = sp.vstack(rows, format='csc')
        l = waypoints.ravel() - delta
        u = waypoints.ravel() + delta
        return A, l, u

    # ----- Utility -----

    @staticmethod
    def _stack_constraints(constraint_list: list) -> tuple:
        """Stack multiple (A, l, u) tuples into a single constraint system."""
        if not constraint_list:
            return sp.csc_matrix((0, 0)), np.array([]), np.array([])

        A_list = [c[0] for c in constraint_list]
        l_list = [c[1] for c in constraint_list]
        u_list = [c[2] for c in constraint_list]

        A = sp.vstack(A_list, format='csc')
        l = np.concatenate(l_list)
        u = np.concatenate(u_list)
        return A, l, u
