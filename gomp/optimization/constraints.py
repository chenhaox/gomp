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

Performance: All constraint matrices are built using vectorized COO
construction (pre-computed index arrays) instead of Python-loop
lil_matrix assembly.
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

        # Pre-compute index offsets used by multiple constraint builders
        self._q_offset = 0                    # q block starts at 0
        self._v_offset = self.m * self.n      # v block starts after all q's
        self._n_vars = qp.n_vars

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
                                  trust_region: float = 0.5,
                                  endpoint_jnt_tol: float = 0.02) -> tuple:
        """
        Build constraints that are updated at each SQP iteration:
        - Obstacle avoidance (linearized)
        - Start/goal grasp constraints (linearized FK, for DOF direction)
        - Endpoint joint-space pinning (prevents FK drift)
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
        endpoint_jnt_tol : float
            Joint-space tolerance for endpoint pinning (radians).
            Larger values give more slack for the optimizer; smaller
            values more strictly pin the FK to the grasp pose.

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

        # 7. Start grasp constraint (linearized FK — controls DOF direction)
        if start_grasp_set is not None:
            constraints.append(
                self._grasp_constraint(waypoints[0], start_grasp_set, waypoint_idx=0)
            )
            # 7b. Endpoint joint-space pinning (prevents FK drift)
            constraints.append(
                self._endpoint_pinning_constraint(
                    start_grasp_set, waypoint_idx=0, jnt_tol=endpoint_jnt_tol)
            )

        # 8. Goal grasp constraint (linearized FK — controls DOF direction)
        if goal_grasp_set is not None:
            constraints.append(
                self._grasp_constraint(waypoints[-1], goal_grasp_set,
                                       waypoint_idx=self.H)
            )
            # 8b. Endpoint joint-space pinning (prevents FK drift)
            constraints.append(
                self._endpoint_pinning_constraint(
                    goal_grasp_set, waypoint_idx=self.H, jnt_tol=endpoint_jnt_tol)
            )

        # 9. Trust regions
        constraints.append(self._trust_region_constraints(waypoints, trust_region))

        return self._stack_constraints(constraints)

    # ----- Vectorized constraint builders -----

    def _build_identity_block(self, n_blocks: int, col_offset: int,
                              n_rows_total: int) -> sp.csc_matrix:
        """
        Build a block-diagonal identity: m blocks of I_n placed at
        consecutive column positions starting from col_offset.

        This is the common pattern for joint limits, velocity limits,
        trust regions, etc.

        Parameters
        ----------
        n_blocks : int
            Number of identity blocks (m = H+1 for waypoints).
        col_offset : int
            Column offset for the first block in x.
        n_rows_total : int
            Total number of constraint rows (n_blocks * n).
        """
        n = self.n
        # Row indices: 0, 1, ..., n_blocks*n - 1
        row_idx = np.arange(n_rows_total)
        # Column indices: col_offset, col_offset+1, ..., col_offset+n_blocks*n-1
        col_idx = col_offset + np.arange(n_rows_total)
        data = np.ones(n_rows_total)
        return sp.coo_matrix((data, (row_idx, col_idx)),
                             shape=(n_rows_total, self._n_vars)).tocsc()

    def _joint_position_limits(self) -> tuple:
        """q_min ≤ q_i ≤ q_max for each waypoint.

        Vectorized: single identity block spanning all q variables.
        """
        n = self.n
        m = self.m
        n_rows = m * n

        A = self._build_identity_block(m, self._q_offset, n_rows)
        l = np.tile(self.robot.q_min, m)
        u = np.tile(self.robot.q_max, m)
        return A, l, u

    def _velocity_limits(self) -> tuple:
        """v_min ≤ v_i ≤ v_max for each waypoint.

        Vectorized: single identity block spanning all v variables.
        """
        n = self.n
        m = self.m
        n_rows = m * n

        A = self._build_identity_block(m, self._v_offset, n_rows)
        l = np.tile(-self.robot.v_max, m)
        u = np.tile(self.robot.v_max, m)
        return A, l, u

    def _acceleration_limits(self) -> tuple:
        """
        -a_max ≤ (v_{i+1} - v_i) / t_step ≤ a_max

        Rearranged: -a_max * t_step ≤ v_{i+1} - v_i ≤ a_max * t_step

        Vectorized: two sets of diagonal entries (−I at v_i, +I at v_{i+1}).
        """
        n = self.n
        H = self.H
        n_rows = H * n

        # Each constraint row i*n+j has:
        #   -1 at column v_offset + i*n + j     (v_i[j])
        #   +1 at column v_offset + (i+1)*n + j (v_{i+1}[j])
        row_idx = np.arange(n_rows)

        # Block i maps to v_i starting at v_offset + i*n
        block_idx = np.repeat(np.arange(H), n)  # [0,0,..,0, 1,1,..,1, ...]
        joint_idx = np.tile(np.arange(n), H)     # [0,1,..,n-1, 0,1,..,n-1, ...]

        col_neg = self._v_offset + block_idx * n + joint_idx       # v_i
        col_pos = self._v_offset + (block_idx + 1) * n + joint_idx  # v_{i+1}

        # Stack: negative entries first, then positive
        rows = np.concatenate([row_idx, row_idx])
        cols = np.concatenate([col_neg, col_pos])
        data = np.concatenate([-np.ones(n_rows), np.ones(n_rows)])

        A = sp.coo_matrix((data, (rows, cols)),
                          shape=(n_rows, self._n_vars)).tocsc()

        a_bound = self.robot.a_max * self.t_step
        l = np.tile(-a_bound, H)
        u = np.tile(a_bound, H)
        return A, l, u

    def _dynamics_constraints(self) -> tuple:
        """
        q_{i+1} = q_i + v_i * t_step  (equality)

        Encoded as: q_{i+1} - q_i - v_i * t_step = 0

        Vectorized: three sets of diagonal entries.
        """
        n = self.n
        H = self.H
        n_rows = H * n

        row_idx = np.arange(n_rows)
        block_idx = np.repeat(np.arange(H), n)
        joint_idx = np.tile(np.arange(n), H)

        # q_{i+1}: +1
        col_q_next = self._q_offset + (block_idx + 1) * n + joint_idx
        # q_i: -1
        col_q_curr = self._q_offset + block_idx * n + joint_idx
        # v_i: -t_step
        col_v_curr = self._v_offset + block_idx * n + joint_idx

        rows = np.concatenate([row_idx, row_idx, row_idx])
        cols = np.concatenate([col_q_next, col_q_curr, col_v_curr])
        data = np.concatenate([
            np.ones(n_rows),              # q_{i+1}
            -np.ones(n_rows),             # -q_i
            np.full(n_rows, -self.t_step)  # -v_i * t_step
        ])

        A = sp.coo_matrix((data, (rows, cols)),
                          shape=(n_rows, self._n_vars)).tocsc()
        l = np.zeros(n_rows)
        u = np.zeros(n_rows)
        return A, l, u

    def _zero_velocity_endpoints(self) -> tuple:
        """v_0 = 0 and v_H = 0 (equality).

        Vectorized: two small identity blocks.
        """
        n = self.n
        n_rows = 2 * n

        row_idx = np.arange(n_rows)
        # v_0 occupies rows 0..n-1, v_H occupies rows n..2n-1
        col_idx = np.concatenate([
            self._v_offset + np.arange(n),              # v_0
            self._v_offset + self.H * n + np.arange(n)  # v_H
        ])
        data = np.ones(n_rows)

        A = sp.coo_matrix((data, (row_idx, col_idx)),
                          shape=(n_rows, self._n_vars)).tocsc()
        l = np.zeros(n_rows)
        u = np.zeros(n_rows)
        return A, l, u

    def _obstacle_constraints(self, waypoints: np.ndarray,
                              obs: ObstacleConstraint) -> tuple:
        """
        Linearized obstacle avoidance for each waypoint.

        From Section IV-C:
            z_obs - p_z(q^{(k)}) + J_z * q^{(k)} ≤ J_z * q^{(k+1)}

        In QP form: J_z * q_i ≥ rhs  →  -J_z * q_i ≤ -rhs
        Or equivalently: rhs ≤ J_z * q_i ≤ +inf

        Vectorized: each row i has n entries from J_z_all[i].
        """
        n = self.n
        m = self.m

        J_z_all, rhs_all = obs.compute_all_waypoints(waypoints)

        # Row i (constraint for waypoint i) has n entries at q_idx(i)
        # row_idx: [0,0,..,0, 1,1,..,1, ..., m-1,m-1,..,m-1]
        row_idx = np.repeat(np.arange(m), n)
        # col_idx: [q_off+0, q_off+1, .., q_off+n-1, q_off+n, .., ...]
        block_idx = np.repeat(np.arange(m), n)
        joint_idx = np.tile(np.arange(n), m)
        col_idx = self._q_offset + block_idx * n + joint_idx
        # data: J_z_all flattened
        data = J_z_all.ravel()

        A = sp.coo_matrix((data, (row_idx, col_idx)),
                          shape=(m, self._n_vars)).tocsc()
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

        Vectorized: 6 rows × n columns placed at the waypoint's q block.
        """
        n = self.n

        # Compute constraint matrices
        R_0 = grasp_set.compute_constraint_rotation()  # 6×6
        epsilon = grasp_set.compute_epsilon()  # 6
        b = grasp_set.compute_constraint_bounds(q_current, R_0)  # 6

        # Jacobian at current config
        J = self.robot.jacobian(q_current)  # 6×n

        # A_coeff = R_0 @ J, applied to the appropriate waypoint's q
        A_coeff = R_0 @ J  # 6×n

        # Build sparse: 6 rows, each with n entries
        row_idx = np.repeat(np.arange(6), n)             # [0,0,..,0, 1,.., 5,..,5]
        col_base = self._q_offset + waypoint_idx * n
        col_idx = np.tile(col_base + np.arange(n), 6)    # [c,c+1,..,c+n-1, c,..]
        data = A_coeff.ravel()

        A = sp.coo_matrix((data, (row_idx, col_idx)),
                          shape=(6, self._n_vars)).tocsc()

        l = b - epsilon
        u = b + epsilon
        return A, l, u

    def _trust_region_constraints(self, waypoints: np.ndarray,
                                  delta: float) -> tuple:
        """
        Trust region: ||q_i^{(k+1)} - q_i^{(k)}||_∞ ≤ δ

        In QP form: q_i^{(k)} - δ ≤ q_i ≤ q_i^{(k)} + δ

        Vectorized: single identity block spanning all q variables
        (same structure as joint_position_limits).
        """
        n = self.n
        m = self.m
        n_rows = m * n

        A = self._build_identity_block(m, self._q_offset, n_rows)
        l = waypoints.ravel() - delta
        u = waypoints.ravel() + delta
        return A, l, u

    def _endpoint_pinning_constraint(self, grasp_set: GraspSet,
                                     waypoint_idx: int,
                                     jnt_tol: float = 0.05) -> tuple:
        """
        Pin the endpoint joint configuration near the IK solution.

        The linearized FK constraint (R_0·J·q) is too weak to prevent
        FK drift when joints move significantly. This directly constrains
        each joint at the endpoint to stay within jnt_tol of the IK
        solution, with extra slack along the grasp DOF null-space
        direction so the optimizer can still exploit the rotational DOF.

        Parameters
        ----------
        grasp_set : GraspSet
            The grasp set (provides IK solution and DOF axis).
        waypoint_idx : int
            Index of the endpoint waypoint (0 or H).
        jnt_tol : float
            Joint-space tolerance in radians for non-DOF joints.
        """
        n = self.n

        # Get the IK solution for this grasp
        q_ik = grasp_set.get_topdown_ik()
        if q_ik is None:
            # If no IK cached, fall back to no constraint
            return sp.csc_matrix((0, self._n_vars)), np.array([]), np.array([])

        # Compute which joint direction corresponds to the grasp DOF.
        # The grasp DOF is rotation about grasp.axis. In joint space,
        # this maps to the Jacobian pseudo-inverse of the angular velocity
        # along that axis.
        J = self.robot.jacobian(q_ik)  # 6×n
        axis = grasp_set.grasp.axis    # 3-vector

        # The angular part of the Jacobian (rows 3:6)
        J_ang = J[3:, :]  # 3×n

        # Project: which joint velocity produces rotation about 'axis'?
        # dof_direction = J_ang^+ @ axis (pseudo-inverse mapping)
        dof_direction = np.linalg.lstsq(J_ang, axis, rcond=None)[0]  # n-vector
        dof_norm = np.linalg.norm(dof_direction)
        if dof_norm > 1e-10:
            dof_direction = dof_direction / dof_norm
        else:
            dof_direction = np.zeros(n)

        # Per-joint tolerance: tight for non-DOF joints, loose for DOF joint.
        # Scale tolerance by how much each joint contributes to the DOF.
        grasp_theta_range = abs(grasp_set.grasp.theta_max - grasp_set.grasp.theta_min)
        dof_slack = np.abs(dof_direction) * grasp_theta_range
        per_joint_tol = jnt_tol + dof_slack

        # Build sparse: n identity rows for the waypoint's q block
        row_idx = np.arange(n)
        col_idx = self._q_offset + waypoint_idx * n + np.arange(n)
        data = np.ones(n)

        A = sp.coo_matrix((data, (row_idx, col_idx)),
                          shape=(n, self._n_vars)).tocsc()
        l = q_ik - per_joint_tol
        u = q_ik + per_joint_tol
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
