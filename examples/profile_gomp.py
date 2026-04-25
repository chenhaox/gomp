"""
Profile GOMP planner to identify performance bottlenecks.

Measures:
- Total planning time
- Per-phase breakdown: IK, constraint build, QP solve, warm start
- Per-SQP-iteration timing
- FK/Jacobian call frequency and cost
"""

import sys
import os
import time
import cProfile
import pstats
import io
import numpy as np

# Ensure gomp and wrs are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrs'))
import gomp  # noqa: F401

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.planner.gomp_planner import GOMPPlanner


def create_test_scenario():
    """Create a standard test scenario for profiling."""
    robot = RobotAdapter()

    pick_grasp = create_topdown_grasp(
        pos=np.array([0.4, 0.0, 0.15]),
        angle_z=0.0,
        theta_range=(-np.pi / 4, np.pi / 4),
        quality=0.9
    )
    place_grasp = create_topdown_grasp(
        pos=np.array([0.0, 0.4, 0.20]),
        angle_z=np.pi / 4,
        theta_range=(-np.pi / 4, np.pi / 4),
        quality=1.0
    )

    bin_obstacle = DepthMapObstacle.create_bin_obstacle(
        bin_center=np.array([0.4, 0.0, 0.0]),
        bin_size=(0.3, 0.4, 0.15),
        wall_height=0.15,
        resolution=0.01
    )

    return robot, pick_grasp, place_grasp, bin_obstacle


def run_with_timing(robot, pick_grasp, place_grasp, bin_obstacle):
    """Run GOMP with manual timing instrumentation."""
    planner = GOMPPlanner(
        robot=robot,
        t_step=0.008,
        initial_H=120,
        min_H=10,
        H_reduction='geometric',
        H_reduction_factor=0.85,
        sqp_kwargs={
            'max_iterations': 30,
            'initial_trust_region': 2.0,
        },
        verbose=True
    )

    t0 = time.perf_counter()
    trajectory = planner.plan(
        grasp_start=pick_grasp,
        grasp_goal=place_grasp,
        obstacles=bin_obstacle
    )
    total_time = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"TOTAL PLANNING TIME: {total_time*1000:.1f} ms")
    if trajectory is not None:
        print(f"Result: H={trajectory.H}, "
              f"duration={trajectory.duration*1000:.0f} ms, "
              f"waypoints={trajectory.n_waypoints}")
    else:
        print("Result: No feasible trajectory found")
    print(f"{'='*60}")
    return trajectory, total_time


def run_with_cprofile(robot, pick_grasp, place_grasp, bin_obstacle):
    """Run GOMP under cProfile to find hotspots."""
    planner = GOMPPlanner(
        robot=robot,
        t_step=0.008,
        initial_H=120,
        min_H=10,
        H_reduction='geometric',
        H_reduction_factor=0.85,
        sqp_kwargs={
            'max_iterations': 30,
            'initial_trust_region': 2.0,
        },
        verbose=False  # Suppress output for clean profiling
    )

    pr = cProfile.Profile()
    pr.enable()

    trajectory = planner.plan(
        grasp_start=pick_grasp,
        grasp_goal=place_grasp,
        obstacles=bin_obstacle
    )

    pr.disable()

    # Print top functions by cumulative time
    print(f"\n{'='*60}")
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print(f"{'='*60}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Print top functions by total time (self time)
    print(f"\n{'='*60}")
    print("TOP 30 FUNCTIONS BY SELF TIME")
    print(f"{'='*60}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    return trajectory


def run_component_timing(robot, pick_grasp, place_grasp, bin_obstacle):
    """Time individual components in isolation."""
    from gomp.grasp.grasp_set import GraspSet
    from gomp.obstacles.collision import ObstacleConstraint
    from gomp.optimization.qp_builder import QPBuilder
    from gomp.optimization.constraints import ConstraintBuilder
    from gomp.optimization.warm_start import spline_warm_start, interpolate_to_shorter

    n = robot.n_dof
    H = 60
    t_step = 0.008

    print(f"\n{'='*60}")
    print("COMPONENT-LEVEL TIMING")
    print(f"{'='*60}")

    # 1. IK
    t0 = time.perf_counter()
    start_gs = GraspSet(pick_grasp, robot)
    q_start = start_gs.get_topdown_ik()
    goal_gs = GraspSet(place_grasp, robot)
    q_goal = goal_gs.get_topdown_ik(seed_jnt_values=q_start)
    t_ik = time.perf_counter() - t0
    print(f"  IK (2 solves):          {t_ik*1000:8.2f} ms")

    if q_start is None or q_goal is None:
        print("  IK failed, cannot continue component timing.")
        return

    # 2. Warm start
    t0 = time.perf_counter()
    for _ in range(10):
        x = spline_warm_start(q_start, q_goal, H, n, t_step)
    t_warm = (time.perf_counter() - t0) / 10
    print(f"  Warm start (H={H}):     {t_warm*1000:8.2f} ms")

    # 3. QP Builder
    t0 = time.perf_counter()
    for _ in range(10):
        qp = QPBuilder(H, n)
        P = qp.build_P()
        p = qp.build_p()
    t_qp = (time.perf_counter() - t0) / 10
    print(f"  QP build (P matrix):    {t_qp*1000:8.2f} ms")

    # 4. Static constraints
    qp = QPBuilder(H, n)
    cb = ConstraintBuilder(qp, t_step, robot)
    t0 = time.perf_counter()
    for _ in range(10):
        A_s, l_s, u_s = cb.build_static_constraints()
    t_static = (time.perf_counter() - t0) / 10
    print(f"  Static constraints:     {t_static*1000:8.2f} ms")

    # 5. Dynamic constraints (with obstacle + grasp)
    waypoints = qp.extract_waypoints(x)
    obs_constraint = ObstacleConstraint(robot, bin_obstacle, safety_margin=0.01)

    t0 = time.perf_counter()
    for _ in range(10):
        A_d, l_d, u_d = cb.build_dynamic_constraints(
            waypoints=waypoints,
            obstacle_constraint=obs_constraint,
            start_grasp_set=start_gs,
            goal_grasp_set=goal_gs,
            trust_region=1.0
        )
    t_dyn = (time.perf_counter() - t0) / 10
    print(f"  Dynamic constraints:    {t_dyn*1000:8.2f} ms")

    # 5b. Break down dynamic constraints
    t0 = time.perf_counter()
    for _ in range(10):
        obs_constraint.compute_all_waypoints(waypoints)
    t_obs = (time.perf_counter() - t0) / 10
    print(f"    ├─ Obstacle (FK loop): {t_obs*1000:7.2f} ms")

    # 6. OSQP setup + solve
    import osqp
    import scipy.sparse as sp

    A = sp.vstack([A_s, A_d], format='csc')
    l = np.concatenate([l_s, l_d])
    u = np.concatenate([u_s, u_d])

    t0 = time.perf_counter()
    for _ in range(5):
        solver = osqp.OSQP()
        solver.setup(
            P=sp.triu(P, format='csc'), q=p, A=A, l=l, u=u,
            eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
            warm_start=True, verbose=False, polish=True
        )
        solver.warm_start(x=x)
        result = solver.solve()
    t_osqp = (time.perf_counter() - t0) / 5
    print(f"  OSQP (setup+solve):     {t_osqp*1000:8.2f} ms")

    # 6b. OSQP solve only (reuse solver)
    t0 = time.perf_counter()
    for _ in range(10):
        result = solver.solve()
    t_solve_only = (time.perf_counter() - t0) / 10
    print(f"    ├─ Solve only (reuse): {t_solve_only*1000:7.2f} ms")

    # 7. Interpolation to shorter
    t0 = time.perf_counter()
    for _ in range(10):
        x_short = interpolate_to_shorter(x, H, int(H * 0.85), n, t_step)
    t_interp = (time.perf_counter() - t0) / 10
    print(f"  Interpolate (H-reduce): {t_interp*1000:8.2f} ms")

    # Summary
    print(f"\n  --- Estimated per-SQP-iteration ---")
    per_iter = t_dyn + t_osqp
    print(f"  Dynamic constraints + OSQP: {per_iter*1000:.2f} ms")
    print(f"  With ~10 SQP iters/H × ~8 H values: "
          f"~{per_iter*10*8*1000:.0f} ms total")


def main():
    print("=" * 60)
    print("GOMP PROFILER")
    print("=" * 60)

    print("\nCreating test scenario...")
    robot, pick, place, obstacle = create_test_scenario()

    # Component timing (most informative)
    run_component_timing(robot, pick, place, obstacle)

    # Full run with timing
    print("\n\nRunning full planning with timing...")
    traj, total = run_with_timing(robot, pick, place, obstacle)

    # cProfile for detailed hotspot analysis
    print("\n\nRunning full planning with cProfile...")
    run_with_cprofile(robot, pick, place, obstacle)


if __name__ == '__main__':
    main()
