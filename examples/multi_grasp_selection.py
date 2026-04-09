"""
Multi-grasp selection demo.

Demonstrates GOMP's ability to optimize over multiple candidate grasps
and select the one yielding the shortest trajectory.
"""

import numpy as np
import gomp  # noqa: F401
from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_sampler import create_topdown_grasp, sample_bin_grasps
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.planner.gomp_planner import GOMPPlanner
from gomp.visualization.joint_plots import plot_joint_profiles


def main():
    print("=" * 60)
    print("GOMP: Multi-Grasp Selection Demo")
    print("=" * 60)

    # Robot
    robot = RobotAdapter(enable_cc=True)

    # Generate multiple candidate pick grasps in a bin
    bin_center = np.array([0.4, 0.0, 0.0])
    pick_candidates = sample_bin_grasps(
        bin_center=bin_center,
        bin_size=(0.3, 0.4, 0.15),
        n_grasps=5,
        z_offset=0.15,
        seed=42
    )
    print(f"\nGenerated {len(pick_candidates)} candidate grasps")

    # Single place location
    place_grasp = create_topdown_grasp(
        pos=np.array([0.0, 0.4, 0.20]),
        angle_z=0.0,
        theta_range=(-np.pi / 4, np.pi / 4)
    )

    # Obstacle
    bin_obstacle = DepthMapObstacle.create_bin_obstacle(
        bin_center=bin_center,
        bin_size=(0.3, 0.4, 0.15),
        wall_height=0.15
    )

    # Plan with all candidates
    planner = GOMPPlanner(
        robot=robot,
        t_step=0.008,
        initial_H=50,
        min_H=5,
        verbose=True,
        sqp_kwargs={'max_iterations': 25}
    )

    best = planner.plan_multi_grasp(
        grasp_candidates=pick_candidates,
        grasp_goal=place_grasp,
        obstacles=bin_obstacle
    )

    if best is not None:
        print(f"\nBest trajectory: {best.duration * 1000:.0f}ms, H={best.H}")
        plot_joint_profiles(best, v_max=robot.v_max, a_max=robot.a_max,
                          title=f'Best of {len(pick_candidates)} Grasps '
                                f'({best.duration*1000:.0f}ms)')
    else:
        print("\nNo feasible trajectory found for any candidate!")


if __name__ == '__main__':
    main()
