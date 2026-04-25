"""
Full-body collision validator using WRS collision primitives.

Unlike the depth-map-based EE-only constraint used during GOMP optimization,
this validator checks ALL robot links against ALL obstacles for each waypoint
along the trajectory, using WRS's built-in collision detection (BVH/GJK).

Usage:
    validator = CollisionValidator(robot_adapter)
    report = validator.validate_trajectory(
        trajectory, obstacle_list, interpolation_density=3
    )
    if not report.is_valid:
        print(f"Collision at t={report.first_collision_time:.3f}s")
"""

import numpy as np
from typing import List, Optional

from gomp.robot_adapter import RobotAdapter
from gomp.planner.trajectory import Trajectory, CollisionReport


class CollisionValidator:
    """
    Full-body collision checking using WRS collision primitives.

    Validates that ALL robot links are collision-free at each waypoint,
    not just the end-effector. This provides a more thorough check than
    the depth-map-based obstacle constraint used during optimization.

    Parameters
    ----------
    robot : RobotAdapter
        Robot model with collision checking enabled.
    """

    def __init__(self, robot: RobotAdapter):
        self.robot = robot

    def validate_configuration(self, q: np.ndarray,
                               obstacle_list: list = None,
                               other_robot_list: list = None) -> bool:
        """
        Check a single configuration for collisions.

        Parameters
        ----------
        q : np.ndarray, shape (n_dof,)
            Joint configuration.
        obstacle_list : list
            WRS CollisionModel obstacles.
        other_robot_list : list
            Other WRS robot instances.

        Returns
        -------
        is_collided : bool
            True if the robot is in collision.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []

        # Set robot to configuration
        self.robot.goto_given_conf(q)
        return self.robot.is_collided(
            obstacle_list=obstacle_list,
            other_robot_list=other_robot_list
        )

    def validate_trajectory(self, trajectory: Trajectory,
                            obstacle_list: list = None,
                            other_robot_list: list = None,
                            interpolation_density: int = 1,
                            stop_on_first: bool = True) -> CollisionReport:
        """
        Validate a full trajectory for collisions at every waypoint.

        Optionally interpolates additional configurations between waypoints
        for denser collision checking.

        Parameters
        ----------
        trajectory : Trajectory
            GOMP trajectory to validate.
        obstacle_list : list
            WRS CollisionModel obstacles.
        other_robot_list : list
            Other WRS robot instances.
        interpolation_density : int
            Number of configurations to check per trajectory segment.
            1 = waypoints only, 5 = 4 intermediate checks per segment.
        stop_on_first : bool
            If True, stop checking after the first collision is found.

        Returns
        -------
        CollisionReport
            Validation results including collision locations.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []

        collisions = []
        n_checked = 0
        first_collision_time = None

        # Save robot state
        self.robot.backup_state()

        try:
            for i in range(trajectory.n_waypoints):
                q_wp = trajectory.waypoints[i]

                # Check the waypoint itself
                n_checked += 1
                is_col = self.validate_configuration(
                    q_wp, obstacle_list, other_robot_list
                )

                if is_col:
                    t = i * trajectory.t_step
                    collisions.append((i, t, 'waypoint'))
                    if first_collision_time is None:
                        first_collision_time = t
                    if stop_on_first:
                        break

                # Check interpolated points between this and next waypoint
                if interpolation_density > 1 and i < trajectory.H:
                    q_next = trajectory.waypoints[i + 1]
                    for j in range(1, interpolation_density):
                        alpha = j / interpolation_density
                        q_interp = (1 - alpha) * q_wp + alpha * q_next
                        n_checked += 1

                        is_col = self.validate_configuration(
                            q_interp, obstacle_list, other_robot_list
                        )

                        if is_col:
                            t = (i + alpha) * trajectory.t_step
                            collisions.append((i, t, f'interp_{j}'))
                            if first_collision_time is None:
                                first_collision_time = t
                            if stop_on_first:
                                break

                    if stop_on_first and collisions:
                        break

        finally:
            # Restore robot state
            self.robot.restore_state()

        is_valid = len(collisions) == 0

        report = CollisionReport(
            is_valid=is_valid,
            collisions=collisions,
            first_collision_time=first_collision_time,
            n_waypoints_checked=n_checked
        )

        # Attach to trajectory
        trajectory.is_collision_free = is_valid
        trajectory.collision_report = report

        return report

    def validate_start_goal(self, start_conf: np.ndarray,
                            goal_conf: np.ndarray,
                            obstacle_list: list = None,
                            other_robot_list: list = None) -> tuple:
        """
        Quick check that start and goal configurations are collision-free.

        Returns
        -------
        (start_ok, goal_ok) : tuple of bool
            Whether each configuration is collision-free.
        """
        if obstacle_list is None:
            obstacle_list = []
        if other_robot_list is None:
            other_robot_list = []

        self.robot.backup_state()
        try:
            start_col = self.validate_configuration(
                start_conf, obstacle_list, other_robot_list)
            goal_col = self.validate_configuration(
                goal_conf, obstacle_list, other_robot_list)
        finally:
            self.robot.restore_state()

        return (not start_col, not goal_col)
