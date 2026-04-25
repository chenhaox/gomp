"""
Robot Adapter - Thin wrapper around WRS manipulator for GOMP.

Provides a unified interface to access FK, IK, Jacobian, and joint limits
from any WRS manipulator model (default: UR5e).
"""

import numpy as np
from wrs.robot_sim.manipulators.ur5e.ur5e import UR5E


class RobotAdapter:
    """
    Wraps a WRS ManipulatorInterface to expose the kinematic functions
    needed by the GOMP planner.

    Attributes:
        robot: The underlying WRS manipulator instance.
        n_dof: Number of degrees of freedom.
    """

    def __init__(self, manipulator_cls=None, enable_cc=True, **kwargs):
        """
        Parameters
        ----------
        manipulator_cls : class
            WRS manipulator class (default: UR5E).
        enable_cc : bool
            Enable collision checker in the WRS model.
        **kwargs : dict
            Additional arguments passed to the manipulator constructor.
        """
        if manipulator_cls is None:
            manipulator_cls = UR5E
        self.robot = manipulator_cls(enable_cc=enable_cc, **kwargs)
        self.n_dof = self.robot.n_dof

    @classmethod
    def from_wrs_robot(cls, robot):
        """
        Create a RobotAdapter from an existing WRS robot instance.

        This avoids creating a new robot — useful when integrating with
        an existing WRS application.

        Parameters
        ----------
        robot : ManipulatorInterface
            An existing WRS manipulator.

        Returns
        -------
        RobotAdapter
            Adapter wrapping the given robot.
        """
        adapter = cls.__new__(cls)
        adapter.robot = robot
        adapter.n_dof = robot.n_dof
        return adapter

    # ----- Joint limits -----

    @property
    def q_min(self) -> np.ndarray:
        """Joint position lower limits (rad)."""
        return self.robot.jnt_ranges[:, 0]

    @property
    def q_max(self) -> np.ndarray:
        """Joint position upper limits (rad)."""
        return self.robot.jnt_ranges[:, 1]

    @property
    def v_max(self) -> np.ndarray:
        """
        Maximum joint velocities (rad/s).

        UR5e specs: joints 1-3: ±π rad/s, joints 4-6: ±2π rad/s.
        Override this for other robots.
        """
        return np.array([np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])

    @property
    def a_max(self) -> np.ndarray:
        """
        Maximum joint accelerations (rad/s²).

        UR5e data-sheet values: shoulder joints ~25 rad/s², wrist ~40 rad/s².
        """
        return np.array([25.0, 25.0, 25.0, 40.0, 40.0, 40.0])

    # ----- Kinematics -----

    def forward_kinematics(self, q: np.ndarray) -> tuple:
        """
        Compute forward kinematics.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        pos : np.ndarray, shape (3,)
            End-effector position in world frame.
        rotmat : np.ndarray, shape (3, 3)
            End-effector rotation matrix in world frame.
        """
        return self.robot.fk(jnt_values=q, toggle_jacobian=False, update=False)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the 6×n geometric Jacobian at configuration q.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        J : np.ndarray, shape (6, n_dof)
            Geometric Jacobian. Rows 0-2: linear velocity, rows 3-5: angular velocity.
        """
        # Use JLChain.jacobian directly — ManipulatorInterface.jacobian has a bug
        # when called with jnt_values (references non-existent gl_flange_homomat)
        return self.robot.jlc.jacobian(jnt_values=q)

    def inverse_kinematics(self, pos: np.ndarray, rotmat: np.ndarray,
                           seed_jnt_values=None) -> np.ndarray | None:
        """
        Compute inverse kinematics.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Target end-effector position.
        rotmat : np.ndarray, shape (3, 3)
            Target end-effector rotation matrix.
        seed_jnt_values : np.ndarray or None
            Seed configuration for the IK solver.

        Returns
        -------
        q : np.ndarray or None
            Joint configuration achieving the target, or None if no solution.
        """
        return self.robot.ik(tgt_pos=pos, tgt_rotmat=rotmat,
                             seed_jnt_values=seed_jnt_values)

    def goto_given_conf(self, q: np.ndarray):
        """Set the robot to a specific joint configuration (updates internal state)."""
        self.robot.goto_given_conf(jnt_values=q)

    def backup_state(self):
        """Save the current robot state for later restoration."""
        self.robot.backup_state()

    def restore_state(self):
        """Restore the previously saved robot state."""
        self.robot.restore_state()

    def is_collided(self, obstacle_list=None, other_robot_list=None,
                    toggle_contacts=False) -> bool:
        """
        Check if the robot is in self-collision or colliding with obstacles.

        Parameters
        ----------
        obstacle_list : list
            List of WRS CollisionModel obstacles.
        other_robot_list : list
            List of other WRS robot instances.
        toggle_contacts : bool
            If True, return (bool, contact_points) instead of just bool.

        Returns
        -------
        collided : bool or (bool, list)
            Whether the robot is in collision.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []
        return self.robot.is_collided(obstacle_list=obstacle_list,
                                      otherrobot_list=other_robot_list,
                                      toggle_contacts=toggle_contacts)

    def get_link_positions(self, q: np.ndarray) -> list:
        """
        Get the world-frame positions of all links for a given configuration.
        Useful for multi-point obstacle checking.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.

        Returns
        -------
        positions : list of np.ndarray
            World-frame position of each link.
        """
        self.robot.goto_given_conf(jnt_values=q)
        positions = []
        for i in range(len(self.robot.jlc.lnks)):
            positions.append(self.robot.jlc.jnts[i].gl_pos_q.copy()
                             if i < len(self.robot.jlc.jnts)
                             else self.robot.gl_tcp_pos.copy())
        return positions
