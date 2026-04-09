"""
WRS-based 3D visualization for GOMP trajectories.

Uses Panda3D (via WRS) to render the robot executing a trajectory,
with obstacles and grasp frames shown in the scene.
"""

import numpy as np
from typing import Optional, List

from gomp.robot_adapter import RobotAdapter
from gomp.planner.trajectory import Trajectory
from gomp.obstacles.depth_map import DepthMapObstacle

import gomp  # noqa: F401
from wrs import modeling as gm


def visualize_trajectory(robot_adapter: RobotAdapter,
                         trajectory: Trajectory,
                         obstacles: Optional[DepthMapObstacle] = None,
                         n_frames: int = 10,
                         cam_pos: list = None,
                         lookat_pos: list = None):
    """
    Visualize a GOMP trajectory using WRS Panda3D rendering.

    Shows the robot at evenly-spaced keyframes along the trajectory,
    with a line connecting the end-effector positions.

    Parameters
    ----------
    robot_adapter : RobotAdapter
        Robot model.
    trajectory : Trajectory
        Computed trajectory to visualize.
    obstacles : DepthMapObstacle or None
        Obstacle to render.
    n_frames : int
        Number of robot poses to display along the trajectory.
    cam_pos : list or None
        Camera position [x, y, z].
    lookat_pos : list or None
        Camera look-at position [x, y, z].
    """
    import wrs.visualization.panda.world as wd

    if cam_pos is None:
        cam_pos = [2, 0, 1.5]
    if lookat_pos is None:
        lookat_pos = [0, 0, 0.3]

    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos)

    # World frame
    gm.gen_frame(ax_length=0.2).attach_to(base)

    # Select keyframe indices
    if n_frames > trajectory.n_waypoints:
        n_frames = trajectory.n_waypoints
    indices = np.linspace(0, trajectory.H, n_frames, dtype=int)

    # Render robot at each keyframe
    ee_positions = []
    for idx, frame_idx in enumerate(indices):
        q = trajectory.waypoints[frame_idx]
        robot_adapter.goto_given_conf(q)

        # Vary alpha for temporal progression
        alpha = 0.2 + 0.8 * (idx / max(1, len(indices) - 1))

        # Generate mesh model
        mesh = robot_adapter.robot.gen_meshmodel(alpha=alpha,
                                                  toggle_tcp_frame=False)
        mesh.attach_to(base)

        # Collect EE positions for path line
        pos, _ = robot_adapter.forward_kinematics(q)
        ee_positions.append(pos.copy())

    # Draw end-effector path
    for i in range(len(ee_positions) - 1):
        gm.gen_stick(spos=ee_positions[i], epos=ee_positions[i + 1],
                     radius=0.003, rgba=[1, 0.3, 0, 1]).attach_to(base)

    # Draw start/end frames
    if len(ee_positions) > 0:
        pos_start, rot_start = robot_adapter.forward_kinematics(trajectory.waypoints[0])
        gm.gen_frame(pos=pos_start, rotmat=rot_start,
                     ax_length=0.08).attach_to(base)

        pos_end, rot_end = robot_adapter.forward_kinematics(trajectory.waypoints[-1])
        gm.gen_frame(pos=pos_end, rotmat=rot_end,
                     ax_length=0.08).attach_to(base)

    # Draw obstacle (bin) if provided
    if obstacles is not None:
        _draw_bin_obstacle(base, obstacles)

    # Title
    duration_ms = trajectory.duration * 1000
    base.setBackgroundColor(0.9, 0.9, 0.9, 1)

    print(f"Trajectory visualization: H={trajectory.H}, "
          f"duration={trajectory.duration:.3f}s ({duration_ms:.0f}ms)")
    print("Close the window to continue...")

    base.run()


def _draw_bin_obstacle(base, depth_map: DepthMapObstacle):
    """Draw a simple representation of the bin obstacle."""
    # Sample the depth map to create a surface visualization
    nx, ny = depth_map.grid.shape
    step = max(1, min(nx, ny) // 20)  # Subsample for performance

    for i in range(0, nx - 1, step):
        for j in range(0, ny - 1, step):
            z = depth_map.grid[i, j]
            if z > 0.001:  # Only draw walls
                x = depth_map.origin[0] + i * depth_map.resolution
                y = depth_map.origin[1] + j * depth_map.resolution
                pos = np.array([x, y, z / 2])
                size = np.array([depth_map.resolution * step,
                                 depth_map.resolution * step, z])
                gm.gen_box(extent=size, pos=pos,
                          rgba=[0.6, 0.6, 0.8, 0.5]).attach_to(base)
