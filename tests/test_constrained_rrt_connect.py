"""Safety regressions for callback-constrained RRT-connect.

Run with pytest, or without extra test dependencies:
    python -m unittest discover -s tests -p test_constrained_rrt_connect.py -v
"""

from pathlib import Path
import io
import random
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomp.planner.constrained_rrt_connect import ConstrainedRRTConnect


class Robot:
    """Small WRS-compatible joint-space robot, without GUI/mesh creation."""

    oiee_list = []

    def __init__(self):
        self.q = np.array([.4, .6])
        self.states = []
        self.seen_other_robots = []

    def backup_state(self):
        self.states.append(self.q.copy())

    def restore_state(self):
        self.q = self.states.pop()

    def get_jnt_values(self):
        return self.q.copy()

    def goto_given_conf(self, jnt_values):
        self.q = np.array(jnt_values, copy=True)

    def are_jnts_in_ranges(self, jnt_values):
        return bool(np.all(np.abs(jnt_values) <= 2))

    def rand_conf(self):
        return np.random.uniform(-2, 2, 2)

    def is_collided(self, obstacle_list, other_robot_list,
                    toggle_contacts=False, toggle_dbg=False):
        self.seen_other_robots = other_robot_list
        hit = any(np.linalg.norm(self.q - center) < radius
                  for center, radius in obstacle_list)
        return (hit, []) if toggle_contacts else hit

    def get_ee_values(self):
        return None

    def gen_meshmodel(self):
        return None


class ConstrainedRRTTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        self.robot = Robot()
        self.initial = self.robot.q.copy()

    def assert_restored(self):
        np.testing.assert_array_equal(self.robot.q, self.initial)
        self.assertEqual(self.robot.states, [])

    def test_callback_sees_fk_and_readonly_configuration(self):
        seen = []

        def callback(robot, q):
            np.testing.assert_array_equal(robot.q, q)
            self.assertFalse(q.flags.writeable)
            seen.append(q.copy())
            return np.bool_(True)

        planner = ConstrainedRRTConnect(self.robot, callback)
        motion = planner.plan([0, 0], [.1, 0], toggle_mesh=False)
        self.assertIsNotNone(motion)
        self.assertGreater(len(seen), 2)
        self.assert_restored()

    def test_invalid_start_and_goal_and_hardware_limits(self):
        planner = ConstrainedRRTConnect(self.robot, lambda r, q: bool(q[1] >= 0))
        for start, goal in (([0, -.1], [0, 1]), ([0, 1], [0, -.1]),
                            ([2.1, 0], [0, 0])):
            with self.subTest(start=start, goal=goal):
                self.assertIsNone(planner.plan(start, goal, toggle_mesh=False))
                self.assert_restored()

    def test_short_start_goal_edge_cannot_jump_thin_constraint(self):
        planner = ConstrainedRRTConnect(self.robot,
                                        lambda r, q: bool(abs(q[0]) >= .012), .005)
        self.assertIsNone(planner.plan([-.08, 0], [.08, 0], ext_dist=.2,
                                        max_n_iter=70, max_time=0, toggle_mesh=False))
        self.assert_restored()

    def test_tree_bridge_cannot_jump_thin_constraint(self):
        planner = ConstrainedRRTConnect(self.robot,
                                        lambda r, q: bool(abs(q[0]) >= .012), .005)
        self.assertIsNone(planner.plan([-1, 0], [1, 0], ext_dist=.5,
                                        max_n_iter=100, max_time=0, toggle_mesh=False))
        self.assert_restored()

    def test_short_edge_cannot_jump_physical_obstacle(self):
        planner = ConstrainedRRTConnect(self.robot, check_resolution=.005)
        self.assertIsNone(planner.plan([-.08, 0], [.08, 0], ext_dist=.2,
                                        obstacle_list=[(np.zeros(2), .03)],
                                        max_n_iter=0, toggle_mesh=False))
        self.assert_restored()

    def test_smoothing_preserves_detour(self):
        planner = ConstrainedRRTConnect(self.robot,
                                        lambda r, q: bool(np.linalg.norm(q) >= .3), .01)
        path = [np.array(q) for q in [[-1, 0], [-1, 1], [1, 1], [1, 0]]]
        with patch("random.randint", side_effect=[0, 3]):
            smoothed = planner._smooth_path(path, granularity=3, n_iter=1)
        np.testing.assert_array_equal(smoothed, path)

    def test_success_around_combined_task_and_physical_obstacles(self):
        callback = lambda r, q: bool(np.linalg.norm(q) >= .35)
        planner = ConstrainedRRTConnect(self.robot, callback, .01)
        obstacle = (np.array([0., .9]), .25)
        motion = planner.plan([-1, 0], [1, 0], obstacle_list=[obstacle],
                              ext_dist=.2, max_time=0, smoothing_n_iter=100,
                              toggle_mesh=False)
        self.assertIsNotNone(motion)
        np.testing.assert_array_equal(motion.jv_list[0], [-1, 0])
        np.testing.assert_array_equal(motion.jv_list[-1], [1, 0])
        for a, b in zip(motion.jv_list[:-1], motion.jv_list[1:]):
            for q in np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / .002) + 2)):
                self.assertTrue(callback(self.robot, q))
                self.assertGreaterEqual(np.linalg.norm(q - obstacle[0]), obstacle[1])
        self.assert_restored()

    def test_first_tree_connection_is_accepted_immediately(self):
        planner = ConstrainedRRTConnect(self.robot, check_resolution=.01)
        with patch.object(planner, "_sample_conf", return_value=np.array([1., 0.])):
            motion = planner.plan([-1, 0], [1, 0], max_n_iter=1,
                                  smoothing_n_iter=0, toggle_mesh=False)
        self.assertIsNotNone(motion)
        np.testing.assert_array_equal(motion.jv_list[0], [-1, 0])
        np.testing.assert_array_equal(motion.jv_list[-1], [1, 0])
        self.assert_restored()

    def test_callback_exception_restores_state_and_can_reuse_planner(self):
        def broken(robot, q):
            raise RuntimeError("callback failed")

        planner = ConstrainedRRTConnect(self.robot, broken)
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            planner.plan([0, 0], [1, 1], toggle_mesh=False)
        self.assert_restored()
        self.assertTrue(planner.toggle_keep)
        planner.constraint_fn = None
        self.assertIsNotNone(planner.plan([0, 0], [.1, 0], toggle_mesh=False))
        self.assert_restored()

    def test_callback_requires_scalar_boolean(self):
        for value in (None, 1, np.array([True, False])):
            planner = ConstrainedRRTConnect(self.robot, lambda r, q: value)
            with self.assertRaisesRegex(TypeError, "must return bool"):
                planner.plan([0, 0], [.1, 0], toggle_mesh=False)
            self.assert_restored()

    def test_contacts_and_other_robot_arguments(self):
        planner = ConstrainedRRTConnect(self.robot, lambda r, q: False)
        sentinel = object()
        self.assertEqual(planner._is_collided(np.zeros(2), other_robot_list=[sentinel],
                                              toggle_contacts=True), (True, []))
        self.assertEqual(self.robot.seen_other_robots, [sentinel])

    def test_zero_distance_and_repeated_plans(self):
        planner = ConstrainedRRTConnect(self.robot)
        for start in (np.zeros(2), np.ones(2)):
            motion = planner.plan(start, start, toggle_mesh=False)
            self.assertEqual(len(motion), 1)
            np.testing.assert_array_equal(motion.jv_list[0], start)
            self.assertEqual(planner.roadmap.number_of_nodes(), 0)
            self.assert_restored()

    def test_invalid_inputs(self):
        for resolution in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                ConstrainedRRTConnect(self.robot, check_resolution=resolution)
        planner = ConstrainedRRTConnect(self.robot)
        for start, goal in (([0], [1]), ([0, 0], [0]), ([np.nan, 0], [0, 0]),
                            ([[0, 0]], [[1, 1]])):
            with self.assertRaises(ValueError):
                planner.plan(start, goal)
        for kwargs in ({"ext_dist": 0}, {"max_time": np.inf}, {"max_n_iter": -1},
                       {"smoothing_n_iter": 1.2}):
            with self.assertRaises(ValueError):
                planner.plan([0, 0], [1, 1], **kwargs)
        self.assert_restored()

    def test_wrs_binary_stl_loader_accepts_current_numpy(self):
        # The 3D example must work with the committed WRS submodule, without
        # relying on a local patch to NumPy's length-one-array conversion.
        from wrs.basis.trimesh.io.stl import load_stl_binary
        triangle = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
        mesh = load_stl_binary(io.BytesIO(b"\0" * 80 + struct.pack("<I", 1) + triangle))
        np.testing.assert_array_equal(mesh["faces"], [[0, 1, 2]])
        np.testing.assert_allclose(mesh["vertices"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
