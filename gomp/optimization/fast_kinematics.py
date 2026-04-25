"""
Fast FK/Jacobian engine for GOMP — avoids WRS per-joint Python loop overhead.

The WRS FK implementation is accurate but slow for batch computation because:
1. Per-joint Python for-loop calling get_motion_homomat
2. rotmat_from_axangle called ~13× per FK (6 joints × ~2 calls each)
3. Each rotmat_from_axangle calls allclose, norm, cross, eye, etc.

This module pre-extracts the kinematic parameters (DH-like: loc_pos, loc_rotmat,
loc_motion_ax, joint type) from a WRS JLChain at init time, then computes FK
and Jacobian using direct numpy operations (Rodrigues formula inlined).

Benchmark target: >10× speedup over WRS jlchain.fk + jlchain.jacobian.
"""

import numpy as np


class FastKinematics:
    """
    Pre-compiled fast FK and Jacobian for a WRS manipulator.

    Extracts kinematic parameters once from a WRS robot, then provides
    FK and Jacobian computation without WRS object overhead.

    Parameters
    ----------
    robot_adapter : RobotAdapter
        GOMP robot adapter (wraps a WRS ManipulatorInterface).
    """

    def __init__(self, robot_adapter):
        robot = robot_adapter.robot
        jlc = robot.jlc
        n = jlc.n_dof

        self.n_dof = n
        self.flange_jnt_id = jlc.flange_jnt_id

        # Pre-extract per-joint kinematic parameters as contiguous numpy arrays
        # Each joint i has: loc_pos (3,), loc_rotmat (3,3), loc_motion_ax (3,)
        self._loc_pos = np.zeros((n, 3))
        self._loc_rotmat = np.zeros((n, 3, 3))
        self._loc_motion_ax = np.zeros((n, 3))

        # Pre-compute per-joint local homomats (4×4) — avoids repeated construction
        self._loc_homomat = np.zeros((n, 4, 4))

        for i in range(n):
            jnt = jlc.jnts[i]
            self._loc_pos[i] = jnt.loc_pos
            self._loc_rotmat[i] = jnt.loc_rotmat
            self._loc_motion_ax[i] = jnt.loc_motion_ax
            # Pre-compute local homomat (translation part only — rotation added at runtime)
            self._loc_homomat[i, :3, :3] = jnt.loc_rotmat
            self._loc_homomat[i, :3, 3] = jnt.loc_pos
            self._loc_homomat[i, 3, 3] = 1.0

        # Anchor base homomat
        anchor_pose = jlc.anchor.gl_flange_pose_list[0]
        self._base_homomat = np.eye(4)
        self._base_homomat[:3, 3] = anchor_pose[0]
        self._base_homomat[:3, :3] = anchor_pose[1]

        # Flange homomat (local)
        self._loc_flange_homomat = np.eye(4)
        self._loc_flange_homomat[:3, 3] = jlc._loc_flange_pos
        self._loc_flange_homomat[:3, :3] = jlc._loc_flange_rotmat

        # TCP offset from ManipulatorInterface
        self._loc_tcp_pos = robot._loc_tcp_pos
        self._loc_tcp_rotmat = robot._loc_tcp_rotmat

        # Pre-compute skew-symmetric matrices for each joint's local motion axis
        # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
        self._K = np.zeros((n, 3, 3))
        self._K2 = np.zeros((n, 3, 3))  # K @ K
        for i in range(n):
            k = self._loc_motion_ax[i]
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            self._K[i] = K
            self._K2[i] = K @ K

    def fk(self, jnt_values):
        """
        Compute FK: returns (tcp_pos, tcp_rotmat).

        Uses Rodrigues formula inline:
            R(k, θ) = I + sin(θ)·K + (1-cos(θ))·K²
        where K is the skew-symmetric matrix of axis k.

        Parameters
        ----------
        jnt_values : np.ndarray, shape (n_dof,)

        Returns
        -------
        tcp_pos : np.ndarray, shape (3,)
        tcp_rotmat : np.ndarray, shape (3, 3)
        """
        homomat = self._base_homomat.copy()
        n = self.flange_jnt_id + 1

        for i in range(n):
            theta = jnt_values[i]
            # Rodrigues: R = I + sin(θ)K + (1-cosθ)K²
            s = np.sin(theta)
            c = np.cos(theta)
            R_motion = np.eye(3) + s * self._K[i] + (1.0 - c) * self._K2[i]

            # Local homomat with motion applied
            # joint_homomat = loc_homomat @ [[R_motion, 0], [0, 1]]
            loc_R = self._loc_rotmat[i]
            loc_p = self._loc_pos[i]

            # Combined: pos contribution from parent frame
            jnt_pos = homomat[:3, 3] + homomat[:3, :3] @ loc_p

            # Rotation: parent_R @ loc_R @ R_motion
            jnt_R = homomat[:3, :3] @ loc_R @ R_motion

            homomat[:3, :3] = jnt_R
            homomat[:3, 3] = jnt_pos

        # Apply flange
        flange_pos = homomat[:3, 3] + homomat[:3, :3] @ self._loc_flange_homomat[:3, 3]
        flange_rotmat = homomat[:3, :3] @ self._loc_flange_homomat[:3, :3]

        # Apply TCP offset
        tcp_pos = flange_pos + flange_rotmat @ self._loc_tcp_pos
        tcp_rotmat = flange_rotmat @ self._loc_tcp_rotmat

        return tcp_pos, tcp_rotmat

    def fk_and_jacobian(self, jnt_values):
        """
        Compute FK and 6×n Jacobian in a single pass.

        Returns
        -------
        tcp_pos : np.ndarray, shape (3,)
        tcp_rotmat : np.ndarray, shape (3, 3)
        J : np.ndarray, shape (6, n_dof)
        """
        n = self.flange_jnt_id + 1
        homomat = self._base_homomat.copy()

        # Store per-joint world positions and motion axes
        jnt_world_pos = np.zeros((n, 3))
        jnt_world_ax = np.zeros((n, 3))

        for i in range(n):
            theta = jnt_values[i]
            s = np.sin(theta)
            c = np.cos(theta)
            R_motion = np.eye(3) + s * self._K[i] + (1.0 - c) * self._K2[i]

            loc_R = self._loc_rotmat[i]
            loc_p = self._loc_pos[i]

            jnt_pos = homomat[:3, 3] + homomat[:3, :3] @ loc_p
            jnt_R = homomat[:3, :3] @ loc_R @ R_motion

            # Store for Jacobian computation (motion axis in world frame)
            jnt_world_pos[i] = jnt_pos
            jnt_world_ax[i] = homomat[:3, :3] @ loc_R @ self._loc_motion_ax[i]

            homomat[:3, :3] = jnt_R
            homomat[:3, 3] = jnt_pos

        # Apply flange
        flange_pos = homomat[:3, 3] + homomat[:3, :3] @ self._loc_flange_homomat[:3, 3]
        flange_rotmat = homomat[:3, :3] @ self._loc_flange_homomat[:3, :3]

        # TCP
        tcp_pos = flange_pos + flange_rotmat @ self._loc_tcp_pos
        tcp_rotmat = flange_rotmat @ self._loc_tcp_rotmat

        # Build Jacobian (all revolute assumed for UR-like robots)
        J = np.zeros((6, self.n_dof))
        for i in range(n):
            j2t = tcp_pos - jnt_world_pos[i]
            J[:3, i] = np.cross(jnt_world_ax[i], j2t)
            J[3:, i] = jnt_world_ax[i]

        return tcp_pos, tcp_rotmat, J

    def batch_fk_and_jacobian(self, waypoints):
        """
        Compute FK and Jacobian for all waypoints at once.

        Parameters
        ----------
        waypoints : np.ndarray, shape (m, n_dof)

        Returns
        -------
        positions : np.ndarray, shape (m, 3)
            TCP positions for each waypoint.
        rotmats : np.ndarray, shape (m, 3, 3)
            TCP rotation matrices for each waypoint.
        jacobians : np.ndarray, shape (m, 6, n_dof)
            Jacobians for each waypoint.
        """
        m = waypoints.shape[0]
        positions = np.zeros((m, 3))
        rotmats = np.zeros((m, 3, 3))
        jacobians = np.zeros((m, 6, self.n_dof))

        for idx in range(m):
            pos, rotmat, J = self.fk_and_jacobian(waypoints[idx])
            positions[idx] = pos
            rotmats[idx] = rotmat
            jacobians[idx] = J

        return positions, rotmats, jacobians

    def batch_fk(self, waypoints):
        """
        Compute FK (position only) for all waypoints.

        Parameters
        ----------
        waypoints : np.ndarray, shape (m, n_dof)

        Returns
        -------
        positions : np.ndarray, shape (m, 3)
        rotmats : np.ndarray, shape (m, 3, 3)
        """
        m = waypoints.shape[0]
        positions = np.zeros((m, 3))
        rotmats = np.zeros((m, 3, 3))

        for idx in range(m):
            pos, rotmat = self.fk(waypoints[idx])
            positions[idx] = pos
            rotmats[idx] = rotmat

        return positions, rotmats
