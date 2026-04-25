"""
WRS 3D Trajectory Visualization Demo.

This example:
1. Runs GOMP to compute a pick-and-place trajectory
2. Visualizes the trajectory in WRS Panda3D with:
   - Static ghost poses showing the robot at keyframes (multi-alpha)
   - End-effector path curve
   - Real-time animation of the robot tracing the trajectory
   - Start/goal grasp frames and obstacle rendering
3. Press SPACE to toggle between ghost-pose view and animation playback
"""

import sys
import os
import numpy as np

# Ensure gomp and wrs are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrs'))
import gomp  # noqa: F401

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp import Grasp
from gomp.grasp.grasp_sampler import create_topdown_grasp
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.planner.gomp_planner import GOMPPlanner
from gomp.planner.trajectory import Trajectory


def draw_bin_obstacle(base, depth_map: DepthMapObstacle):
    """Draw a simple representation of the bin obstacle."""
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


def draw_ee_path(base, robot_adapter, trajectory, n_samples=60):
    """Draw a smooth end-effector path as a polyline."""
    from wrs.modeling import geometric_model as gm

    # Sample the trajectory at fine intervals
    times = np.linspace(0, trajectory.duration, n_samples)
    ee_positions = []
    for t in times:
        q, _ = trajectory.evaluate(t)
        pos, _ = robot_adapter.forward_kinematics(q)
        ee_positions.append(pos.copy())

    # Draw path segments with a smooth color gradient (orange -> cyan)
    for i in range(len(ee_positions) - 1):
        frac = i / max(1, len(ee_positions) - 2)
        r = 1.0 - 0.7 * frac
        g = 0.3 + 0.5 * frac
        b = 0.0 + 0.9 * frac
        gm.gen_stick(spos=ee_positions[i], epos=ee_positions[i + 1],
                     radius=0.003, rgb=np.array([r, g, b]), alpha=0.9).attach_to(base)

    return ee_positions


def draw_ghost_poses(base, robot_adapter, trajectory, n_frames=10):
    """Draw multiple semi-transparent robot poses along the trajectory."""
    from wrs.modeling import geometric_model as gm

    if n_frames > trajectory.n_waypoints:
        n_frames = trajectory.n_waypoints
    indices = np.linspace(0, trajectory.H, n_frames, dtype=int)

    ghost_models = []
    for idx, frame_idx in enumerate(indices):
        q = trajectory.waypoints[frame_idx]
        robot_adapter.goto_given_conf(q)

        # Vary alpha for temporal progression (faint -> solid)
        alpha = 0.15 + 0.85 * (idx / max(1, len(indices) - 1))
        mesh = robot_adapter.robot.gen_meshmodel(alpha=alpha,
                                                  toggle_tcp_frame=False)
        mesh.attach_to(base)
        ghost_models.append(mesh)

    # Draw start and goal frames
    pos_start, rot_start = robot_adapter.forward_kinematics(trajectory.waypoints[0])
    gm.gen_frame(pos=pos_start, rotmat=rot_start, ax_length=0.1).attach_to(base)

    pos_end, rot_end = robot_adapter.forward_kinematics(trajectory.waypoints[-1])
    gm.gen_frame(pos=pos_end, rotmat=rot_end, ax_length=0.1).attach_to(base)

    return ghost_models


def draw_grasp_set(base, grasp: Grasp, n_samples: int = 7,
                   label: str = "grasp",
                   color_start: np.ndarray = np.array([1.0, 0.3, 0.0]),
                   color_end: np.ndarray = np.array([0.0, 0.8, 0.3])):
    """
    Visualize the rotational DOF of a grasp as a fan of coordinate frames.

    Samples poses across [theta_min, theta_max] and draws small frames at each,
    plus a dashed arc showing the rotation axis/range.

    Parameters
    ----------
    base : wd.World
        WRS world to attach models to.
    grasp : Grasp
        The grasp with rotational DOF to visualize.
    n_samples : int
        Number of sampled poses to draw.
    label : str
        Label for console output.
    color_start : np.ndarray
        RGB color for the first sample (theta_min).
    color_end : np.ndarray
        RGB color for the last sample (theta_max).
    """
    from wrs.modeling import geometric_model as gm

    poses = grasp.sample_poses(n_samples)

    for idx, (pos, rotmat, theta) in enumerate(poses):
        frac = idx / max(1, len(poses) - 1)
        # Interpolate color
        rgb = (1.0 - frac) * color_start + frac * color_end
        alpha_val = 0.4 + 0.5 * (1.0 - abs(2.0 * frac - 1.0))  # brighter in the middle

        # Draw a small frame at this grasp pose
        gm.gen_frame(pos=pos, rotmat=rotmat,
                     ax_length=0.04, ax_radius=0.0015,
                     alpha=alpha_val).attach_to(base)

        # Draw a small sphere at the grasp center for each sample
        gm.gen_sphere(pos=pos, radius=0.005,
                      rgb=rgb, alpha=alpha_val).attach_to(base)

    # Draw the rotation axis as a dashed line through the grasp center
    axis_half_len = 0.06
    axis_start = grasp.pos - grasp.axis * axis_half_len
    axis_end = grasp.pos + grasp.axis * axis_half_len
    gm.gen_dashed_stick(spos=axis_start, epos=axis_end,
                        radius=0.001, rgb=np.array([0.5, 0.5, 0.5]),
                        alpha=0.6).attach_to(base)

    # Connect the sampled EE positions with thin sticks to show the arc
    for i in range(len(poses) - 1):
        frac = i / max(1, len(poses) - 2)
        rgb = (1.0 - frac) * color_start + frac * color_end
        p0 = poses[i][0]
        p1 = poses[i + 1][0]
        # Only draw if positions differ (they share the same pos for top-down grasps,
        # but the frames rotate — still useful to draw the axis indicator)
        dist = np.linalg.norm(p1 - p0)
        if dist > 1e-6:
            gm.gen_stick(spos=p0, epos=p1, radius=0.001,
                         rgb=rgb, alpha=0.5).attach_to(base)

    print(f"  {label}: {n_samples} sampled poses "
          f"over θ ∈ [{np.degrees(grasp.theta_min):.0f}°, "
          f"{np.degrees(grasp.theta_max):.0f}°]")


def visualize_trajectory_animated(robot_adapter: RobotAdapter,
                                  trajectory: Trajectory,
                                  obstacles: DepthMapObstacle = None,
                                  pick_grasp: Grasp = None,
                                  place_grasp: Grasp = None,
                                  n_ghost_frames: int = 8,
                                  playback_speed: float = 1.0):
    """
    Visualize a GOMP trajectory with real-time animation in WRS.

    Features:
    - Ghost poses at keyframes show the full trajectory at a glance
    - End-effector path as a smooth colored curve
    - Real-time animated playback of the robot executing the trajectory
    - Plays back in a loop at configurable speed

    Parameters
    ----------
    robot_adapter : RobotAdapter
        Robot model.
    trajectory : Trajectory
        Computed trajectory to visualize.
    obstacles : DepthMapObstacle or None
        Obstacle to render.
    n_ghost_frames : int
        Number of ghost keyframe poses to display.
    playback_speed : float
        Speed multiplier for animation (1.0 = real-time, 0.5 = half speed).
    """
    import wrs.visualization.panda.world as wd
    from wrs.modeling import geometric_model as gm

    base = wd.World(cam_pos=[1.5, 1.0, 1.2], lookat_pos=[0.2, 0.1, 0.2])
    base.setBackgroundColor(0.95, 0.95, 0.97, 1)

    # World frame
    gm.gen_frame(ax_length=0.15).attach_to(base)

    # Ground plane (subtle grid effect)
    gm.gen_box(xyz_lengths=np.array([2.0, 2.0, 0.001]),
               pos=np.array([0, 0, -0.0005]),
               rgb=np.array([0.85, 0.85, 0.88]), alpha=0.3).attach_to(base)

    # Draw obstacles
    if obstacles is not None:
        draw_bin_obstacle(base, obstacles)

    # Draw grasp sets (rotational DOF visualization)
    # The fan shows the full range; the trajectory endpoints show what the
    # optimizer actually selected within that range.
    if pick_grasp is not None:
        print("  Drawing pick grasp set...")
        draw_grasp_set(base, pick_grasp, n_samples=7, label="Pick grasp set",
                       color_start=np.array([1.0, 0.3, 0.0]),
                       color_end=np.array([0.0, 0.8, 0.3]))
        # Draw the actual optimized pick pose (from trajectory start)
        pos_pick, rot_pick = robot_adapter.forward_kinematics(trajectory.waypoints[0])
        gm.gen_frame(pos=pos_pick, rotmat=rot_pick,
                     ax_length=0.07, ax_radius=0.003).attach_to(base)
        gm.gen_sphere(pos=pos_pick, radius=0.010,
                      rgb=np.array([1.0, 0.0, 0.0]), alpha=0.9).attach_to(base)
        print(f"    → Optimized pick EE at {np.round(pos_pick, 4)}")

    if place_grasp is not None:
        print("  Drawing place grasp set...")
        draw_grasp_set(base, place_grasp, n_samples=7, label="Place grasp set",
                       color_start=np.array([0.2, 0.3, 1.0]),
                       color_end=np.array([0.8, 0.2, 0.8]))
        # Draw the actual optimized place pose (from trajectory end)
        pos_place, rot_place = robot_adapter.forward_kinematics(trajectory.waypoints[-1])
        gm.gen_frame(pos=pos_place, rotmat=rot_place,
                     ax_length=0.07, ax_radius=0.003).attach_to(base)
        gm.gen_sphere(pos=pos_place, radius=0.010,
                      rgb=np.array([0.0, 0.0, 1.0]), alpha=0.9).attach_to(base)
        print(f"    → Optimized place EE at {np.round(pos_place, 4)}")

    # Draw ghost poses
    print(f"  Drawing {n_ghost_frames} ghost keyframes...")
    draw_ghost_poses(base, robot_adapter, trajectory, n_frames=n_ghost_frames)

    # Draw EE path
    print(f"  Drawing end-effector path...")
    draw_ee_path(base, robot_adapter, trajectory)

    # --- Animation data ---
    # Subsample waypoints for smooth animation at ~30fps visual update
    # The trajectory has H+1 waypoints at t_step intervals
    # We'll step through all of them with doMethodLater
    update_interval = trajectory.t_step / playback_speed

    class AnimationData:
        def __init__(self):
            self.counter = 0
            self.current_mesh = None
            self.ee_marker = None
            self.paused = False

    anim = AnimationData()

    # Title text
    duration_ms = trajectory.duration * 1000
    print(f"\n  Trajectory: H={trajectory.H}, "
          f"duration={trajectory.duration:.3f}s ({duration_ms:.0f}ms)")
    print(f"  Animation speed: {playback_speed}x "
          f"(update interval: {update_interval * 1000:.1f}ms)")
    print(f"  Close the window to exit.")

    def update(anim_data, task):
        """Panda3D task: animate the robot through trajectory waypoints."""
        # Remove the previous animation mesh
        if anim_data.current_mesh is not None:
            anim_data.current_mesh.detach()
        if anim_data.ee_marker is not None:
            anim_data.ee_marker.detach()

        # Loop back to start
        if anim_data.counter >= trajectory.n_waypoints:
            anim_data.counter = 0

        # Set robot to current waypoint
        q = trajectory.waypoints[anim_data.counter]
        robot_adapter.goto_given_conf(q)

        # Render the animated robot in a distinct color (opaque, highlighted)
        mesh = robot_adapter.robot.gen_meshmodel(alpha=1.0,
                                                  toggle_tcp_frame=True)
        mesh.attach_to(base)
        anim_data.current_mesh = mesh

        # Draw a small sphere at the current EE position
        pos, _ = robot_adapter.forward_kinematics(q)
        marker = gm.gen_sphere(pos=pos, radius=0.012,
                               rgb=np.array([1.0, 0.2, 0.2]), alpha=0.9)
        marker.attach_to(base)
        anim_data.ee_marker = marker

        anim_data.counter += 1
        return task.again

    # Register the animation task
    # Use doMethodLater so we can control the playback rate
    from direct.task.TaskManagerGlobal import taskMgr
    taskMgr.doMethodLater(update_interval, update, "gomp_trajectory_animation",
                          extraArgs=[anim],
                          appendTask=True)

    base.run()


def main():
    """Run the WRS trajectory visualization demo."""
    print("=" * 60)
    print("GOMP: WRS 3D Trajectory Visualization ")
    print("=" * 60)

    # --- 1. Create robot ---
    print("\n1. Creating UR5e robot...")
    robot = RobotAdapter()

    # --- 2. Define grasps ---
    print("2. Defining grasps...")
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

    # --- 3. Create obstacle ---
    print("3. Creating bin obstacle...")
    bin_obstacle = DepthMapObstacle.create_bin_obstacle(
        bin_center=np.array([0.4, 0.0, 0.0]),
        bin_size=(0.3, 0.4, 0.15),
        wall_height=0.4,
        resolution=0.01
    )

    # --- 4. Run GOMP ---
    print("4. Running GOMP planner...")
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

    if trajectory is None:
        print("\nNo feasible trajectory found!")
        return

    # --- 5. Results summary ---
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

    # --- 6. Launch WRS 3D animated visualization ---
    print("\n5. Launching WRS 3D animated visualization...")
    visualize_trajectory_animated(
        robot_adapter=robot,
        trajectory=trajectory,
        obstacles=bin_obstacle,
        pick_grasp=pick_grasp,
        place_grasp=place_grasp,
        n_ghost_frames=8,
        playback_speed=0.5  # Half speed for better visibility
    )


if __name__ == '__main__':
    main()
