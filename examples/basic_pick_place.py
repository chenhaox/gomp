"""
Basic pick-and-place demo using GOMP.

This example:
1. Creates a UR5e robot via WRS
2. Defines pick and place grasps with rotational DOF
3. Creates a bin obstacle
4. Runs GOMP to find an optimized trajectory
5. Plots joint profiles and optionally visualizes in 3D
"""

import sys
import os
import numpy as np

# Ensure gomp and wrs are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrs'))
import gomp  # noqa: F401

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.planner.gomp_planner import GOMPPlanner
from gomp.visualization.joint_plots import plot_joint_profiles


def main(visualize_3d=False):
    """Run basic pick-and-place GOMP demo."""
    print("=" * 60)
    print("GOMP: Basic Pick-and-Place Demo")
    print("=" * 60)

    # --- 1. Create robot ---
    print("\n1. Creating UR5e robot...")
    robot = RobotAdapter()
    print(f"   DOF: {robot.n_dof}")
    print(f"   q_min: {np.round(robot.q_min, 2)}")
    print(f"   q_max: {np.round(robot.q_max, 2)}")
    print(f"   v_max: {np.round(robot.v_max, 2)}")

    # --- 2. Define grasps ---
    print("\n2. Defining grasps...")

    # Pick grasp: above a bin center, top-down
    pick_grasp = create_topdown_grasp(
        pos=np.array([0.4, 0.0, 0.15]),  # Above bin
        angle_z=0.0,
        theta_range=(-np.pi / 4, np.pi / 4),  # ±45° rotation DOF
        quality=0.9
    )
    print(f"   Pick: pos={pick_grasp.pos}, axis={np.round(pick_grasp.axis, 2)}")

    # Place grasp: to the side, top-down
    place_grasp = create_topdown_grasp(
        pos=np.array([0.0, 0.4, 0.20]),  # Above place location
        angle_z=np.pi / 4,  # Rotated 45°
        theta_range=(-np.pi / 4, np.pi / 4),
        quality=1.0
    )
    print(f"   Place: pos={place_grasp.pos}, axis={np.round(place_grasp.axis, 2)}")

    # --- 3. Create obstacle ---
    print("\n3. Creating bin obstacle...")
    bin_obstacle = DepthMapObstacle.create_bin_obstacle(
        bin_center=np.array([0.4, 0.0, 0.0]),
        bin_size=(0.3, 0.4, 0.15),  # width, depth, height
        wall_height=0.15,
        resolution=0.01
    )
    print(f"   Grid size: {bin_obstacle.grid.shape}")
    print(f"   Wall height: {np.max(bin_obstacle.grid):.2f}m")

    # --- 4. Run GOMP ---
    print("\n4. Running GOMP planner...")
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

    trajectory = planner.plan(
        grasp_start=pick_grasp,
        grasp_goal=place_grasp,
        obstacles=bin_obstacle
    )

    # --- 5. Results ---
    if trajectory is None:
        print("\nNo feasible trajectory found!")
        return

    print(f"\n{'=' * 60}")
    print(f"RESULT:")
    print(f"  Duration: {trajectory.duration * 1000:.0f} ms")
    print(f"  Waypoints: {trajectory.n_waypoints}")
    print(f"  Max velocity: {np.round(trajectory.max_velocity(), 2)} rad/s")
    print(f"  Max acceleration: {np.round(trajectory.max_acceleration(), 2)} rad/s^2")

    limits = trajectory.is_within_limits(
        robot.q_min, robot.q_max, robot.v_max, robot.a_max
    )
    print(f"  Limits satisfied: {limits}")
    print(f"{'=' * 60}")

    # --- 6. Plot ---
    print("\n5. Plotting joint profiles...")
    plot_joint_profiles(
        trajectory,
        v_max=robot.v_max,
        a_max=robot.a_max,
        title=f'GOMP Pick-and-Place (H={trajectory.H}, '
              f'{trajectory.duration*1000:.0f}ms)'
    )

    # --- 7. 3D Visualization ---
    if visualize_3d:
        print("\n6. Launching 3D visualization...")
        from gomp.visualization.wrs_viz import visualize_trajectory
        visualize_trajectory(robot, trajectory, bin_obstacle, n_frames=8)


if __name__ == '__main__':
    visualize_3d = '--3d' in sys.argv
    main(visualize_3d=visualize_3d)
