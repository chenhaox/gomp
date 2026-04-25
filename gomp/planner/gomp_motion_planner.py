"""
WRS-compatible GOMP Motion Planner.

Provides a `plan()` interface matching WRS's RRT/RRTConnect pattern:
    planner = GOMPMotionPlanner(robot)
    mot_data = planner.plan(start_conf, goal_conf, obstacle_list=[...])

This allows drop-in replacement of RRT planners with GOMP for time-optimal
trajectory generation.

For full GOMP grasp DOF optimization, use `plan_with_grasps()`.
"""

import numpy as np
from typing import Optional, List

import wrs.motion.motion_data as motd

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp import Grasp
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.collision_validator import CollisionValidator
from gomp.planner.gomp_planner import GOMPPlanner
from gomp.planner.trajectory import Trajectory


class GOMPMotionPlanner:
    """
    WRS-compatible GOMP motion planner.

    Mirrors the WRS RRT/RRTConnect interface:
    - `__init__(robot)` — takes a WRS ManipulatorInterface
    - `plan(start_conf, goal_conf, obstacle_list)` — returns MotionData
    - Saves/restores robot state automatically

    Also provides GOMP-specific features:
    - `plan_with_grasps()` — grasp DOF optimization
    - `plan_trajectory()` — returns a Trajectory object instead of MotionData
    - `collision_check` option for full-body validation

    Parameters
    ----------
    robot : ManipulatorInterface
        A WRS manipulator (e.g., UR5E).

    Examples
    --------
    >>> from wrs.robot_sim.manipulators.ur5e.ur5e import UR5E
    >>> from gomp.planner.gomp_motion_planner import GOMPMotionPlanner
    >>> robot = UR5E(enable_cc=True)
    >>> planner = GOMPMotionPlanner(robot)
    >>> mot_data = planner.plan(start_conf, goal_conf)
    """

    def __init__(self, robot):
        """
        Parameters
        ----------
        robot : ManipulatorInterface
            Any WRS manipulator with collision checking enabled.
        """
        self.robot = robot
        self._adapter = RobotAdapter.from_wrs_robot(robot)
        self._validator = CollisionValidator(self._adapter)

        # Default planner settings (override via plan kwargs)
        self.default_t_step = 0.008
        self.default_initial_H = 120
        self.default_min_H = 10
        self.default_H_reduction = 'geometric'
        self.default_H_reduction_factor = 0.85

    def plan(self, start_conf, goal_conf,
             obstacle_list=None,
             other_robot_list=None,
             depth_map_obstacle=None,
             t_step=None,
             initial_H=None,
             min_H=None,
             collision_check=False,
             max_time=15.0,
             toggle_dbg=False,
             **kwargs):
        """
        Plan a trajectory using GOMP optimization.

        Compatible with WRS RRT's `plan()` signature where possible.
        Returns MotionData (same as RRTConnect.plan()).

        Parameters
        ----------
        start_conf : np.ndarray
            Starting joint configuration.
        goal_conf : np.ndarray
            Goal joint configuration.
        obstacle_list : list, optional
            WRS CollisionModel obstacles for collision validation.
        other_robot_list : list, optional
            Other WRS robots for collision validation.
        depth_map_obstacle : DepthMapObstacle, optional
            Depth map for GOMP obstacle constraints during optimization.
            If None, a flat depth map is created (no obstacle avoidance).
        t_step : float, optional
            Time step (defaults to 0.008s).
        initial_H : int, optional
            Initial time horizon (defaults to 120).
        min_H : int, optional
            Minimum time horizon (defaults to 10).
        collision_check : bool
            If True, validate the trajectory with full-body collision
            checking before returning. Returns None if collisions found.
        max_time : float
            Maximum planning time in seconds (not used currently,
            reserved for compatibility).
        toggle_dbg : bool
            If True, print debug info.
        **kwargs
            Additional arguments passed to GOMPPlanner.

        Returns
        -------
        mot_data : MotionData or None
            WRS-compatible motion data, or None if no feasible path found.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []

        # Check start and goal for collisions (matching RRT pattern)
        if toggle_dbg:
            print("GOMP: Checking start robot configuration...")
        if collision_check and self._is_collided(start_conf, obstacle_list,
                                                  other_robot_list):
            print("GOMP: The start robot configuration is in collision!")
            return None

        if toggle_dbg:
            print("GOMP: Checking goal robot configuration...")
        if collision_check and self._is_collided(goal_conf, obstacle_list,
                                                  other_robot_list):
            print("GOMP: The goal robot configuration is in collision!")
            return None

        # Create grasps from start/goal configurations
        # (No grasp DOF — just use FK to get EE poses)
        self._adapter.backup_state()
        try:
            start_pos, start_rotmat = self._adapter.forward_kinematics(start_conf)
            goal_pos, goal_rotmat = self._adapter.forward_kinematics(goal_conf)
        finally:
            self._adapter.restore_state()

        grasp_start = Grasp(
            pos=start_pos, rotmat=start_rotmat,
            axis=start_rotmat[:, 2],  # Use z-axis
            theta_min=0.0, theta_max=0.0  # No DOF — pinned to exact config
        )
        grasp_goal = Grasp(
            pos=goal_pos, rotmat=goal_rotmat,
            axis=goal_rotmat[:, 2],
            theta_min=0.0, theta_max=0.0
        )

        # Run GOMP planner, with IK seeded from the known configs
        trajectory = self._plan_internal(
            grasp_start, grasp_goal,
            depth_map_obstacle=depth_map_obstacle,
            t_step=t_step, initial_H=initial_H, min_H=min_H,
            verbose=toggle_dbg,
            ik_seed_start=np.asarray(start_conf),
            ik_seed_goal=np.asarray(goal_conf),
            **kwargs
        )

        if trajectory is None:
            return None

        # Full-body collision validation
        if collision_check:
            report = self._validator.validate_trajectory(
                trajectory, obstacle_list, other_robot_list,
                interpolation_density=3
            )
            if not report.is_valid:
                if toggle_dbg:
                    print(f"GOMP: Trajectory has collisions at "
                          f"t={report.first_collision_time:.3f}s")
                return None

        # Convert to MotionData
        return trajectory.to_motion_data(self.robot)

    def plan_with_grasps(self, grasp_start, grasp_goal,
                         obstacle_list=None,
                         depth_map_obstacle=None,
                         collision_check=False,
                         **kwargs):
        """
        Plan with GOMP grasp DOF optimization.

        This is the full GOMP algorithm with grasp rotational DOF.

        Parameters
        ----------
        grasp_start : Grasp
            Start grasp with rotational DOF.
        grasp_goal : Grasp
            Goal grasp with rotational DOF.
        obstacle_list : list, optional
            WRS obstacles for collision validation.
        depth_map_obstacle : DepthMapObstacle, optional
            Depth map for optimization constraints.
        collision_check : bool
            If True, validate trajectory with full-body checking.
        **kwargs
            Passed to GOMPPlanner.

        Returns
        -------
        mot_data : MotionData or None
        """
        if obstacle_list is None:
            obstacle_list = []

        trajectory = self._plan_internal(
            grasp_start, grasp_goal,
            depth_map_obstacle=depth_map_obstacle,
            verbose=kwargs.pop('toggle_dbg', False),
            **kwargs
        )

        if trajectory is None:
            return None

        if collision_check:
            report = self._validator.validate_trajectory(
                trajectory, obstacle_list, interpolation_density=3
            )
            if not report.is_valid:
                return None

        return trajectory.to_motion_data(self.robot)

    def plan_trajectory(self, start_conf, goal_conf,
                        depth_map_obstacle=None,
                        toggle_dbg=False,
                        **kwargs):
        """
        Plan and return a Trajectory object (not MotionData).

        Useful when you need access to velocities, accelerations,
        timing, and collision reports.

        Parameters
        ----------
        start_conf : np.ndarray
            Starting joint configuration.
        goal_conf : np.ndarray
            Goal joint configuration.
        depth_map_obstacle : DepthMapObstacle, optional
            Depth map obstacle for optimization.
        toggle_dbg : bool
            If True, print debug info.
        **kwargs
            Passed to GOMPPlanner.

        Returns
        -------
        trajectory : Trajectory or None
        """
        start_conf = np.asarray(start_conf)
        goal_conf = np.asarray(goal_conf)

        # Create grasps from start/goal
        self._adapter.backup_state()
        try:
            start_pos, start_rotmat = self._adapter.forward_kinematics(start_conf)
            goal_pos, goal_rotmat = self._adapter.forward_kinematics(goal_conf)
        finally:
            self._adapter.restore_state()

        grasp_start = Grasp(
            pos=start_pos, rotmat=start_rotmat,
            axis=start_rotmat[:, 2],
            theta_min=0.0, theta_max=0.0
        )
        grasp_goal = Grasp(
            pos=goal_pos, rotmat=goal_rotmat,
            axis=goal_rotmat[:, 2],
            theta_min=0.0, theta_max=0.0
        )

        return self._plan_internal(
            grasp_start, grasp_goal,
            depth_map_obstacle=depth_map_obstacle,
            verbose=toggle_dbg,
            ik_seed_start=start_conf,
            ik_seed_goal=goal_conf,
            **kwargs
        )

    def _plan_internal(self, grasp_start, grasp_goal,
                       depth_map_obstacle=None,
                       t_step=None, initial_H=None, min_H=None,
                       verbose=False,
                       ik_seed_start=None, ik_seed_goal=None,
                       **kwargs):
        """Run the internal GOMP planner."""
        planner = GOMPPlanner(
            robot=self._adapter,
            t_step=t_step or self.default_t_step,
            initial_H=initial_H or self.default_initial_H,
            min_H=min_H or self.default_min_H,
            H_reduction=self.default_H_reduction,
            H_reduction_factor=self.default_H_reduction_factor,
            sqp_kwargs=kwargs.get('sqp_kwargs', {
                'max_iterations': 30,
                'initial_trust_region': 2.0,
                'verbose': verbose,
            }),
            verbose=verbose
        )

        return planner.plan(
            grasp_start=grasp_start,
            grasp_goal=grasp_goal,
            obstacles=depth_map_obstacle,
            ik_seed_start=ik_seed_start,
            ik_seed_goal=ik_seed_goal
        )

    def _is_collided(self, conf, obstacle_list, other_robot_list):
        """Check if a single configuration is in collision."""
        self._adapter.backup_state()
        try:
            self._adapter.goto_given_conf(conf)
            return self._adapter.is_collided(
                obstacle_list=obstacle_list,
                other_robot_list=other_robot_list
            )
        finally:
            self._adapter.restore_state()
