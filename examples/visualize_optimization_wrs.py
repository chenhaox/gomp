"""
WRS 3D visualization of the GOMP time-optimization process.

Shows how the trajectory changes as the time horizon H is reduced:
1. Solves the pick-and-place problem at multiple H values
2. Displays all trajectories side-by-side in WRS as ghost EE paths
3. Animates through each H-stage sequentially, cycling in a loop

This lets you visually compare how the robot motion tightens as
the planner approaches the minimum feasible duration.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrs'))
import gomp  # noqa: F401

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.collision import ObstacleConstraint
from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.sqp_solver import SQPSolver
from gomp.optimization.warm_start import spline_warm_start, interpolate_to_shorter
from gomp.planner.trajectory import Trajectory


# Distinct colors for each H-stage trajectory
STAGE_COLORS = [
    np.array([0.2, 0.6, 1.0]),   # blue    (initial, long)
    np.array([0.2, 0.8, 0.4]),   # green
    np.array([1.0, 0.7, 0.0]),   # orange
    np.array([1.0, 0.2, 0.3]),   # red     (final, short)
    np.array([0.7, 0.3, 0.9]),   # purple
    np.array([0.0, 0.8, 0.8]),   # cyan
]


def draw_bin_obstacle(base, depth_map):
    """Draw the bin obstacle."""
    from wrs.modeling import geometric_model as gm

    nx, ny = depth_map.grid.shape
    step = max(1, min(nx, ny) // 20)

    for i in range(0, nx - 1, step):
        for j in range(0, ny - 1, step):
            z = depth_map.grid[i, j]
            if z > 0.001:
                x = depth_map.origin[0] + i * depth_map.resolution
                y = depth_map.origin[1] + j * depth_map.resolution
                pos = np.array([x, y, z / 2])
                size = np.array([depth_map.resolution * step,
                                 depth_map.resolution * step, z])
                gm.gen_box(xyz_lengths=size, pos=pos,
                           rgb=np.array([0.6, 0.6, 0.8]), alpha=0.5).attach_to(base)


def draw_ee_path_static(base, robot_adapter, trajectory, color, alpha=0.6, n_samples=50):
    """Draw the end-effector path of a trajectory as a static polyline."""
    from wrs.modeling import geometric_model as gm

    times = np.linspace(0, trajectory.duration, n_samples)
    ee_positions = []
    for t in times:
        q, _ = trajectory.evaluate(t)
        pos, _ = robot_adapter.forward_kinematics(q)
        ee_positions.append(pos.copy())

    for i in range(len(ee_positions) - 1):
        gm.gen_stick(spos=ee_positions[i], epos=ee_positions[i + 1],
                     radius=0.002, rgb=color, alpha=alpha).attach_to(base)

    # Small spheres at start and end
    gm.gen_sphere(pos=ee_positions[0], radius=0.008,
                  rgb=color, alpha=0.9).attach_to(base)
    gm.gen_sphere(pos=ee_positions[-1], radius=0.008,
                  rgb=color, alpha=0.9).attach_to(base)

    return ee_positions


def compute_trajectories(robot, t_step=0.008, H_values=None):
    """
    Run the SQP optimizer at multiple H values and collect trajectories.

    Returns
    -------
    trajectories : list of Trajectory
        Feasible trajectories at decreasing H values.
    """
    n = robot.n_dof

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

    start_gs = GraspSet(pick, robot)
    goal_gs = GraspSet(place, robot)

    q_start = start_gs.get_topdown_ik()
    q_goal = goal_gs.get_topdown_ik(seed_jnt_values=q_start)

    if q_start is None or q_goal is None:
        print("IK failed!")
        return [], None

    obs = DepthMapObstacle.create_bin_obstacle(
        bin_center=np.array([0.4, 0.0, 0.0]),
        bin_size=(0.3, 0.4, 0.15),
        wall_height=0.3
    )
    obs_constraint = ObstacleConstraint(robot, obs)

    sqp = SQPSolver(max_iterations=30, verbose=True)

    if H_values is None:
        H_values = [120, 100, 90, 85]

    trajectories = []
    x = None
    for H in H_values:
        print(f"\n{'=' * 40}")
        print(f"Solving H = {H} (duration = {H * t_step * 1000:.0f}ms)")
        print(f"{'=' * 40}")

        if x is None:
            x = spline_warm_start(q_start, q_goal, H, n, t_step)
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
        velocities = qp.extract_velocities(result.x)

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
              f"duration={traj.duration * 1000:.0f}ms")

    return trajectories, obs


def visualize_optimization_process(robot_adapter, trajectories, obstacle=None):
    """
    Visualize the H-reduction optimization process in WRS.

    Draws:
    - All EE paths overlaid (color-coded per stage)
    - Animates the robot through each trajectory sequentially,
      pausing briefly between stages
    """
    import wrs.visualization.panda.world as wd
    from wrs.modeling import geometric_model as gm
    from direct.task.TaskManagerGlobal import taskMgr

    base = wd.World(cam_pos=[1.5, 1.0, 1.2], lookat_pos=[0.2, 0.1, 0.2])
    base.setBackgroundColor(0.95, 0.95, 0.97, 1)

    # World frame
    gm.gen_frame(ax_length=0.15).attach_to(base)

    # Ground plane
    gm.gen_box(xyz_lengths=np.array([2.0, 2.0, 0.001]),
               pos=np.array([0, 0, -0.0005]),
               rgb=np.array([0.85, 0.85, 0.88]), alpha=0.3).attach_to(base)

    # Draw obstacle
    if obstacle is not None:
        draw_bin_obstacle(base, obstacle)

    # Draw all EE paths overlaid with distinct colors
    print("\nDrawing EE paths for all stages...")
    for idx, traj in enumerate(trajectories):
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        draw_ee_path_static(base, robot_adapter, traj, color=color, alpha=0.5)
        print(f"  Stage {idx + 1}: H={traj.H}, "
              f"duration={traj.duration * 1000:.0f}ms  (color: {color})")

    # Draw start/end frames from the first trajectory
    if trajectories:
        pos_s, rot_s = robot_adapter.forward_kinematics(trajectories[0].waypoints[0])
        gm.gen_frame(pos=pos_s, rotmat=rot_s, ax_length=0.08).attach_to(base)
        pos_e, rot_e = robot_adapter.forward_kinematics(trajectories[0].waypoints[-1])
        gm.gen_frame(pos=pos_e, rotmat=rot_e, ax_length=0.08).attach_to(base)

    # --- Animation: cycle through each trajectory stage ---
    # Between stages, hold the final pose for a pause period

    PAUSE_FRAMES = 30  # number of idle frames between stages

    class AnimData:
        def __init__(self):
            self.stage = 0           # current trajectory index
            self.frame = 0           # current waypoint within trajectory
            self.pause_counter = 0   # pause counter between stages
            self.in_pause = False
            self.mesh = None
            self.marker = None
            self.label = None

    anim = AnimData()

    n_stages = len(trajectories)
    print(f"\nAnimating {n_stages} stages in a loop...")
    print("  Close the window to exit.")

    def update(ad, task):
        # Clean up previous frame
        if ad.mesh is not None:
            ad.mesh.detach()
        if ad.marker is not None:
            ad.marker.detach()
        if ad.label is not None:
            ad.label.detach()

        traj = trajectories[ad.stage]
        color = STAGE_COLORS[ad.stage % len(STAGE_COLORS)]

        # --- Pause between stages ---
        if ad.in_pause:
            ad.pause_counter += 1
            if ad.pause_counter >= PAUSE_FRAMES:
                ad.in_pause = False
                ad.pause_counter = 0
                ad.stage = (ad.stage + 1) % n_stages
                ad.frame = 0
            else:
                # Show the robot frozen at the last waypoint during pause
                q = traj.waypoints[-1]
                robot_adapter.goto_given_conf(q)
                mesh = robot_adapter.robot.gen_meshmodel(alpha=0.6,
                                                          toggle_tcp_frame=False)
                mesh.attach_to(base)
                ad.mesh = mesh
            return task.again

        # --- Normal playback ---
        q = traj.waypoints[ad.frame]
        robot_adapter.goto_given_conf(q)

        mesh = robot_adapter.robot.gen_meshmodel(alpha=1.0,
                                                  toggle_tcp_frame=True)
        mesh.attach_to(base)
        ad.mesh = mesh

        # EE marker in stage color
        pos, _ = robot_adapter.forward_kinematics(q)
        marker = gm.gen_sphere(pos=pos, radius=0.010,
                               rgb=color, alpha=0.9)
        marker.attach_to(base)
        ad.marker = marker

        # Stage label as a small colored sphere cluster at a fixed position
        # (simple visual indicator of which stage is playing)
        label_pos = np.array([0.0, -0.3, 0.5 + ad.stage * 0.05])
        label = gm.gen_sphere(pos=label_pos, radius=0.015,
                              rgb=color, alpha=0.9)
        label.attach_to(base)
        ad.label = label

        ad.frame += 1
        if ad.frame >= traj.n_waypoints:
            ad.in_pause = True
            ad.frame = traj.n_waypoints - 1

        return task.again

    # Use a fixed update interval (not tied to any single trajectory's t_step)
    # to keep animation speed consistent across all stages
    update_interval = 0.016  # ~60fps visual update, each frame = one waypoint
    taskMgr.doMethodLater(update_interval, update, "optimization_animation",
                          extraArgs=[anim],
                          appendTask=True)

    base.run()


def main():
    print("=" * 60)
    print("GOMP: WRS Optimization Process Visualization")
    print("=" * 60)

    # --- 1. Create robot ---
    print("\n1. Creating UR5e robot...")
    robot = RobotAdapter()

    # --- 2. Compute trajectories at multiple H values ---
    print("\n2. Running SQP at multiple H values...")
    trajectories, obs = compute_trajectories(
        robot,
        t_step=0.008,
        H_values=[120, 100, 85, 70]
    )

    if not trajectories:
        print("\nNo feasible trajectories found!")
        return

    # --- 3. Summary ---
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(trajectories)} feasible trajectories")
    for i, t in enumerate(trajectories):
        print(f"  Stage {i + 1}: H={t.H:3d}  duration={t.duration * 1000:6.0f}ms  "
              f"max_v={np.max(t.max_velocity()):5.2f} rad/s")
    print(f"{'=' * 60}")

    # --- 4. Visualize ---
    print("\n3. Launching WRS visualization...")
    visualize_optimization_process(robot, trajectories, obstacle=obs)


if __name__ == '__main__':
    main()
