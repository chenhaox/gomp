"""
Trajectory data structure for GOMP.

Stores a discretized trajectory as a sequence of waypoints and velocities,
along with timing and grasp information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class CollisionReport:
    """
    Result of a trajectory collision validation.

    Attributes
    ----------
    is_valid : bool
        True if the entire trajectory is collision-free.
    collisions : list of tuple (int, object)
        List of (waypoint_index, collision_info) for each collision detected.
    first_collision_time : float or None
        Time (seconds) of the first collision along the trajectory.
    n_waypoints_checked : int
        Total number of waypoints checked.
    """
    is_valid: bool
    collisions: list = field(default_factory=list)
    first_collision_time: Optional[float] = None
    n_waypoints_checked: int = 0


@dataclass
class Trajectory:
    """
    A time-optimal trajectory computed by GOMP.

    Attributes
    ----------
    waypoints : np.ndarray, shape (H+1, n)
        Joint configurations at each waypoint.
    velocities : np.ndarray, shape (H+1, n)
        Joint velocities at each waypoint.
    t_step : float
        Time interval between consecutive waypoints (seconds).
    H : int
        Number of time steps (trajectory has H+1 waypoints).
    start_theta : float
        Optimized grasp rotation at the start (radians).
    goal_theta : float
        Optimized grasp rotation at the goal (radians).
    is_collision_free : bool or None
        Whether the trajectory has been validated as collision-free.
        None if not yet checked.
    collision_report : CollisionReport or None
        Detailed collision validation results.
    """
    waypoints: np.ndarray
    velocities: np.ndarray
    t_step: float
    H: int
    start_theta: float = 0.0
    goal_theta: float = 0.0
    is_collision_free: Optional[bool] = None
    collision_report: Optional[CollisionReport] = None

    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        return self.H * self.t_step

    @property
    def n_dof(self) -> int:
        """Number of joints."""
        return self.waypoints.shape[1]

    @property
    def n_waypoints(self) -> int:
        """Number of waypoints (H+1)."""
        return self.H + 1

    @property
    def accelerations(self) -> np.ndarray:
        """
        Compute accelerations at each waypoint from finite differences
        of velocities.

        Returns
        -------
        accel : np.ndarray, shape (H, n)
            Approximated accelerations (H values from H+1 velocities).
        """
        return np.diff(self.velocities, axis=0) / self.t_step

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the trajectory at an arbitrary time t via linear interpolation.

        Parameters
        ----------
        t : float
            Time in seconds (0 ≤ t ≤ duration).

        Returns
        -------
        q : np.ndarray, shape (n,)
            Interpolated joint configuration.
        v : np.ndarray, shape (n,)
            Interpolated joint velocity.
        """
        t = np.clip(t, 0, self.duration)
        # Fractional waypoint index
        idx_f = t / self.t_step
        idx = int(idx_f)
        frac = idx_f - idx

        if idx >= self.H:
            return self.waypoints[-1].copy(), self.velocities[-1].copy()

        q = (1 - frac) * self.waypoints[idx] + frac * self.waypoints[idx + 1]
        v = (1 - frac) * self.velocities[idx] + frac * self.velocities[idx + 1]
        return q, v

    def time_array(self) -> np.ndarray:
        """Return array of time values for each waypoint."""
        return np.arange(self.n_waypoints) * self.t_step

    def max_velocity(self) -> np.ndarray:
        """Maximum absolute velocity for each joint."""
        return np.max(np.abs(self.velocities), axis=0)

    def max_acceleration(self) -> np.ndarray:
        """Maximum absolute acceleration for each joint."""
        accel = self.accelerations
        return np.max(np.abs(accel), axis=0)

    def is_within_limits(self, q_min, q_max, v_max, a_max) -> dict:
        """
        Check if the trajectory respects all mechanical limits.

        Returns
        -------
        report : dict
            Keys: 'position', 'velocity', 'acceleration' — each True/False.
        """
        pos_ok = np.all(self.waypoints >= q_min) and np.all(self.waypoints <= q_max)
        vel_ok = np.all(np.abs(self.velocities) <= v_max + 1e-6)
        accel = self.accelerations
        acc_ok = np.all(np.abs(accel) <= a_max + 1e-6)
        return {
            'position': pos_ok,
            'velocity': vel_ok,
            'acceleration': acc_ok
        }

    def to_motion_data(self, robot):
        """
        Convert to WRS MotionData for compatibility with WRS workflows.

        Parameters
        ----------
        robot : ManipulatorInterface
            The WRS robot model (must match the DOF of this trajectory).

        Returns
        -------
        mot_data : MotionData
            WRS-compatible motion data with jv_list populated.
        """
        import wrs.motion.motion_data as motd

        mot_data = motd.MotionData(robot)
        jv_list = [self.waypoints[i] for i in range(self.n_waypoints)]
        # Populate jv_list directly instead of using extend()
        # since extend() requires get_ee_values which ManipulatorInterface lacks
        mot_data._jv_list = jv_list
        mot_data._ev_list = [None] * len(jv_list)
        mot_data._mesh_list = [None] * len(jv_list)
        mot_data._oiee_gl_pose_list = [None] * len(jv_list)
        return mot_data

