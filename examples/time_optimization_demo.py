"""
Time optimization demo.

Shows the H-reduction process visually: plots joint velocity/acceleration
profiles for decreasing H values, demonstrating how the optimizer
approaches the time-optimal solution.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import gomp  # noqa: F401

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp import Grasp
from gomp.grasp.grasp_set import GraspSet
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.collision import ObstacleConstraint
from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.sqp_solver import SQPSolver
from gomp.optimization.warm_start import spline_warm_start, interpolate_to_shorter
from gomp.planner.trajectory import Trajectory
from gomp.visualization.joint_plots import plot_time_optimization


def main():
    print("=" * 60)
    print("GOMP: Time Optimization Visualization")
    print("=" * 60)

    # Setup
    robot = RobotAdapter(enable_cc=True)
    n = robot.n_dof
    t_step = 0.008

    # Grasps
    pick = create_topdown_grasp(
        pos=np.array([0.4, 0.0, 0.15]),
        angle_z=0.0,
        theta_range=(-np.pi / 4, np.pi / 4)
    )
    place = create_topdown_grasp(
        pos=np.array([0.0, 0.4, 0.20]),
        angle_z=np.pi / 6,
        theta_range=(-np.pi / 4, np.pi / 4)
    )

    # Grasp sets
    start_gs = GraspSet(pick, robot)
    goal_gs = GraspSet(place, robot)

    # IK
    q_start = start_gs.get_topdown_ik()
    q_goal = goal_gs.get_topdown_ik(seed_jnt_values=q_start)

    if q_start is None or q_goal is None:
        print("IK failed!")
        return

    # Obstacle
    obs = DepthMapObstacle.create_bin_obstacle(
        bin_center=np.array([0.4, 0.0, 0.0]),
        bin_size=(0.3, 0.4, 0.15),
        wall_height=0.15
    )
    obs_constraint = ObstacleConstraint(robot, obs)

    # SQP solver
    sqp = SQPSolver(max_iterations=30, verbose=True)

    # Solve at multiple H values and collect trajectories
    trajectories = []
    H_values = [60, 40, 25, 15]

    x = None
    for H in H_values:
        print(f"\n{'='*40}")
        print(f"Solving H = {H} (duration = {H * t_step * 1000:.0f}ms)")
        print(f"{'='*40}")

        if x is None:
            x = spline_warm_start(q_start, q_goal, H, n)
        else:
            x = interpolate_to_shorter(x_prev, H_prev, H, n, t_step)

        result = sqp.solve(
            x_init=x, H=H, n=n, t_step=t_step,
            robot=robot,
            obstacle_constraint=obs_constraint,
            start_grasp_set=start_gs,
            goal_grasp_set=goal_gs
        )

        if not result.feasible:
            print(f"H={H}: INFEASIBLE — stopping")
            break

        qp = QPBuilder(H, n)
        waypoints = qp.extract_waypoints(result.x)
        velocities = qp.extract_velocities(result.x) / t_step

        traj = Trajectory(
            waypoints=waypoints,
            velocities=velocities,
            t_step=t_step,
            H=H
        )
        trajectories.append(traj)

        x_prev = result.x
        H_prev = H

        print(f"H={H}: cost={result.final_cost:.4f}, "
              f"max_v={np.max(np.abs(velocities)):.2f}, "
              f"duration={traj.duration*1000:.0f}ms")

    # Plot comparison
    if trajectories:
        print(f"\nPlotting {len(trajectories)} trajectories...")
        plot_time_optimization(
            trajectories,
            v_max=robot.v_max,
            a_max=robot.a_max
        )
    else:
        print("No feasible trajectories to plot!")


if __name__ == '__main__':
    main()
