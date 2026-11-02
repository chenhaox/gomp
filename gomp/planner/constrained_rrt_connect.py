"""WRS RRT-connect with a state-validity callback and checked local edges.

The callback ``constraint_fn(robot, conf) -> bool`` returns True when a state
is allowed. Joint limits and WRS collision checking still apply. The robot's
FK has already been updated to ``conf`` when the callback is called.
"""

import math
import time
from typing import Callable, Optional

import networkx as nx
import numpy as np

from wrs.motion.motion_data import MotionData
from wrs.motion.probabilistic.rrt_connect import RRTConnect


class ConstrainedRRTConnect(RRTConnect):
    """Add rejection-based task constraints to the existing WRS planner.

    Parameters
    ----------
    robot
        A WRS RobotInterface supporting the existing RRTConnect API.
    constraint_fn
        Optional callable ``(robot, conf) -> bool``. True means valid. Read
        ``robot.gl_tcp_pos`` / ``robot.gl_tcp_rotmat`` for workspace constraints.
        The callback must be deterministic and must not modify robot state.
        Combine several constraints with ``all(...)``. Exceptions propagate.
    check_resolution
        Maximum Euclidean joint-space distance between validation samples,
        independent of ``plan(ext_dist=...)``. Revolute joints use radians.
        This is discrete checking, not a continuous collision guarantee.

    Notes
    -----
    This planner reuses WRS's bidirectional search, collision checks and
    shortcut smoothing. It also checks short start/goal links, tree bridges
    and shortcut interiors, which a state-only override would miss.
    Inequality constraints and finite tolerances work well. Exact equality
    manifolds generally require a projection-based planner instead.
    """

    def __init__(self, robot, constraint_fn: Optional[Callable] = None,
                 check_resolution: float = 0.02):
        super().__init__(robot)
        if constraint_fn is not None and not callable(constraint_fn):
            raise TypeError("constraint_fn must be callable or None")
        if not np.isfinite(check_resolution) or check_resolution <= 0:
            raise ValueError("check_resolution must be finite and positive")
        self.constraint_fn = constraint_fn
        self.check_resolution = float(check_resolution)
        self._obstacles = []
        self._other_robots = []
        self.constraint_checks = 0
        self.constraint_rejections = 0

    def _is_collided(self, conf, obstacle_list=None, other_robot_list=None,
                     toggle_contacts=False, toggle_dbg=False):
        # Retain physical joint limits, self collision and obstacle collision.
        info = super()._is_collided(
            conf, [] if obstacle_list is None else obstacle_list,
            [] if other_robot_list is None else other_robot_list,
            toggle_contacts=toggle_contacts, toggle_dbg=toggle_dbg)
        collided = info[0] if toggle_contacts else info
        if collided or self.constraint_fn is None:
            return info

        # A private, read-only copy prevents callbacks corrupting tree nodes.
        callback_conf = np.array(conf, dtype=float, copy=True)
        callback_conf.setflags(write=False)
        self.constraint_checks += 1
        valid = self.constraint_fn(self.robot, callback_conf)
        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError("constraint_fn must return bool (True means valid)")
        if not valid:
            self.constraint_rejections += 1
            if toggle_dbg:
                print("ConstrainedRRTConnect: task constraint rejected this state.")
            return (True, []) if toggle_contacts else True
        return info

    def _extend_conf(self, src_conf, end_conf, ext_dist, exact_end=True):
        """Bound every sample gap and include the exact endpoint, no overshoot.

        WRS uses this hook for both tree growth and shortcut smoothing. Even
        when its caller requests exact_end=False, returning the endpoint is
        necessary to validate the edge joining a shortcut to the old path.
        """
        src_conf = np.asarray(src_conf, dtype=float)
        end_conf = np.asarray(end_conf, dtype=float)
        distance = np.linalg.norm(end_conf - src_conf)
        if distance == 0:
            return [src_conf.copy()]
        step = min(ext_dist, self.check_resolution)
        count = max(1, math.ceil(distance / step))
        return list(np.linspace(src_conf, end_conf, count + 1))

    def _is_goal_reached(self, conf, goal_conf, threshold):
        if not super()._is_goal_reached(conf, goal_conf, threshold):
            return False
        # Called for the direct-path shortcut AND every inter-tree bridge.
        return all(not self._is_collided(q, self._obstacles, self._other_robots)
                   for q in self._extend_conf(conf, goal_conf,
                                              self.check_resolution))

    def plan(self, start_conf, goal_conf, obstacle_list=None,
             other_robot_list=None, ext_dist=0.2, max_n_iter=10000,
             max_time=15.0, smoothing_n_iter=500, animation=False,
             toggle_dbg=False, toggle_mesh=True):
        """Return WRS MotionData, or None if no valid path is found.

        Parameters match WRS RRTConnect.plan. All output segments use linear
        joint interpolation. Any later spline fitting must be revalidated.
        ``animation=True`` uses WRS's 2D search drawing; the examples provide
        richer constraint plots and robot playback separately.
        Robot state is restored even if a callback raises an exception, unless
        the caller explicitly sets the inherited ``toggle_keep=False``.
        ``toggle_mesh=False`` skips mesh generation for headless examples
        (use the default True when carrying a held object).
        """
        start = np.array(start_conf, dtype=float, copy=True)
        goal = np.array(goal_conf, dtype=float, copy=True)
        if (start.ndim != 1 or start.size == 0 or start.shape != goal.shape
                or not np.all(np.isfinite(start))
                or not np.all(np.isfinite(goal))):
            raise ValueError("start_conf and goal_conf must be finite vectors of equal size")
        if start.shape != np.asarray(self.robot.get_jnt_values()).shape:
            raise ValueError("configuration size must match the robot's joint count")
        if not np.isfinite(ext_dist) or ext_dist <= 0:
            raise ValueError("ext_dist must be finite and positive")
        if not np.isfinite(max_time):
            raise ValueError("max_time must be finite (<= 0 disables the time limit)")
        for name, value in (("max_n_iter", max_n_iter),
                            ("smoothing_n_iter", smoothing_n_iter)):
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

        self.constraint_checks = self.constraint_rejections = 0
        self._obstacles = [] if obstacle_list is None else list(obstacle_list)
        self._other_robots = [] if other_robot_list is None else list(other_robot_list)
        keep = self.toggle_keep
        if keep:
            self.robot.backup_state()
        try:
            self.roadmap.clear()
            self.roadmap_start.clear()
            self.roadmap_goal.clear()
            self.start_conf, self.goal_conf = start, goal
            for label, conf in (("start", start), ("goal", goal)):
                if self._is_collided(conf, self._obstacles, self._other_robots,
                                     toggle_dbg=toggle_dbg):
                    if toggle_dbg:
                        print(f"ConstrainedRRTConnect: invalid {label} configuration.")
                    return None
            if self._is_goal_reached(start, goal, ext_dist):
                path = self._extend_conf(start, goal, ext_dist)
            else:
                self.roadmap_start.add_node("start", conf=start)
                self.roadmap_goal.add_node("goal", conf=goal)
                tree_a, tree_b = self.roadmap_start, self.roadmap_goal
                target_a, target_b = "goal", "start"
                deadline = time.monotonic() + max_time
                for _ in range(max_n_iter):
                    if max_time > 0 and time.monotonic() >= deadline:
                        return None
                    last_a = self._extend_roadmap(
                        tree_a, self._sample_conf(100, None), ext_dist,
                        tree_b.nodes[target_a]["conf"], self._obstacles,
                        self._other_robots, animation)
                    # Either tree can reach the opposite tree. Finish on the
                    # first connection, before reusing WRS's sentinel node.
                    if last_a == "connection":
                        self.roadmap = nx.compose(tree_a, tree_b)
                        self.roadmap.add_edge("connection", target_a)
                        break
                    if last_a != -1:
                        target_b = last_a
                        last_b = self._extend_roadmap(
                            tree_b, tree_a.nodes[last_a]["conf"], ext_dist,
                            tree_a.nodes[last_a]["conf"], self._obstacles,
                            self._other_robots, animation)
                        if last_b == "connection":
                            self.roadmap = nx.compose(tree_a, tree_b)
                            self.roadmap.add_edge("connection", last_a)
                            break
                        if last_b != -1:
                            target_a = last_b
                    if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                        tree_a, tree_b = tree_b, tree_a
                        target_a, target_b = target_b, target_a
                else:
                    return None
                path = self._smooth_path(
                    self._path_from_roadmap(), self._obstacles,
                    self._other_robots, ext_dist, smoothing_n_iter, animation)
            motion = MotionData(self.robot)
            motion.extend(jv_list=path, toggle_mesh=toggle_mesh)
            return motion
        finally:
            if keep:
                self.robot.restore_state()
            self._obstacles = []
            self._other_robots = []
