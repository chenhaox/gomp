"""
WRS-compatible GOMP interface with 3D trajectory visualization.

Uses GOMPMotionPlanner (WRS-compatible plan() interface) and
visualizes the resulting trajectory with animated ghost poses.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrs'))
import numpy as np
import gomp

from wrs.robot_sim.manipulators.ur5e.ur5e import UR5E
from gomp.planner.gomp_motion_planner import GOMPMotionPlanner
from gomp.robot_adapter import RobotAdapter

# Visualization functions from visualize_trajectory_wrs.py
from visualize_trajectory_wrs import (
    draw_ghost_poses,
    draw_ee_path,
)


def main():
    print('=' * 60)
    print('GOMP: WRS Interface + 3D Visualization')
    print('=' * 60)

    # --- 1. Setup robot and planner ---
    print('\n1. Creating UR5e robot and GOMP planner...')
    robot = UR5E(enable_cc=True)
    planner = GOMPMotionPlanner(robot)
    adapter = RobotAdapter.from_wrs_robot(robot)

    # --- 2. Define start/goal configs ---
    q_start = np.array([0.0, -1.0, 0.5, 0.0, 0.5, 0.0])
    q_goal = np.array([1.0, -0.5, 0.3, 0.5, 0.8, -0.5])
    print(f'   start: {q_start}')
    print(f'   goal:  {q_goal}')

    # --- 3. Plan trajectory ---
    print('\n2. Running GOMP planner...')
    trajectory = planner.plan_trajectory(q_start, q_goal, toggle_dbg=True)

    if trajectory is None:
        print('\nNo feasible trajectory found!')
        return

    # --- 4. Results summary ---
    print(f'\n{"=" * 60}')
    print(f'RESULT:')
    print(f'  Duration: {trajectory.duration * 1000:.0f} ms')
    print(f'  Waypoints: {trajectory.n_waypoints}')
    print(f'  Max velocity: {np.round(trajectory.max_velocity(), 2)} rad/s')
    print(f'  Max acceleration: {np.round(trajectory.max_acceleration(), 2)} rad/s^2')

    limits = trajectory.is_within_limits(
        adapter.q_min, adapter.q_max, adapter.v_max, adapter.a_max
    )
    print(f'  Limits satisfied: {limits}')
    print(f'{"=" * 60}')

    # --- 5. WRS MotionData conversion ---
    print('\n3. Converting to WRS MotionData...')
    mot_data = trajectory.to_motion_data(robot)
    print(f'   MotionData: {len(mot_data)} waypoints, type={type(mot_data).__name__}')

    # --- 6. Launch WRS 3D animated visualization ---
    print('\n4. Launching WRS 3D visualization...')
    print('   Ghost poses + EE path + real-time animation')
    print('   Close window to exit.')

    import wrs.visualization.panda.world as wd
    from wrs.modeling import geometric_model as gm
    from direct.task.TaskManagerGlobal import taskMgr

    base = wd.World(cam_pos=[1.5, 1.0, 1.2], lookat_pos=[0.2, 0.1, 0.2])
    base.setBackgroundColor(0.95, 0.95, 0.97, 1)

    # World frame + ground plane
    gm.gen_frame(ax_length=0.15).attach_to(base)
    gm.gen_box(xyz_lengths=np.array([2.0, 2.0, 0.001]),
               pos=np.array([0, 0, -0.0005]),
               rgb=np.array([0.85, 0.85, 0.88]), alpha=0.3).attach_to(base)

    # Draw ghost poses along the trajectory
    print('   Drawing ghost keyframes...')
    draw_ghost_poses(base, adapter, trajectory, n_frames=8)

    # Draw EE path
    print('   Drawing end-effector path...')
    draw_ee_path(base, adapter, trajectory)

    # Draw start/goal markers
    pos_start, rot_start = adapter.forward_kinematics(q_start)
    gm.gen_sphere(pos=pos_start, radius=0.012,
                  rgb=np.array([0.0, 0.8, 0.2]), alpha=0.9).attach_to(base)

    pos_goal, rot_goal = adapter.forward_kinematics(q_goal)
    gm.gen_sphere(pos=pos_goal, radius=0.012,
                  rgb=np.array([1.0, 0.2, 0.2]), alpha=0.9).attach_to(base)

    # --- Animated playback ---
    playback_speed = 0.5
    update_interval = trajectory.t_step / playback_speed

    class AnimState:
        counter = 0
        current_mesh = None
        ee_marker = None

    anim = AnimState()

    def update(anim_data, task):
        if anim_data.current_mesh is not None:
            anim_data.current_mesh.detach()
        if anim_data.ee_marker is not None:
            anim_data.ee_marker.detach()

        if anim_data.counter >= trajectory.n_waypoints:
            anim_data.counter = 0

        q = trajectory.waypoints[anim_data.counter]
        adapter.goto_given_conf(q)

        mesh = adapter.robot.gen_meshmodel(alpha=1.0, toggle_tcp_frame=True)
        mesh.attach_to(base)
        anim_data.current_mesh = mesh

        pos, _ = adapter.forward_kinematics(q)
        marker = gm.gen_sphere(pos=pos, radius=0.012,
                               rgb=np.array([1.0, 0.2, 0.2]), alpha=0.9)
        marker.attach_to(base)
        anim_data.ee_marker = marker

        anim_data.counter += 1
        return task.again

    taskMgr.doMethodLater(update_interval, update, 'gomp_anim',
                          extraArgs=[anim], appendTask=True)

    print(f'\n   Animation: {playback_speed}x speed '
          f'(update every {update_interval * 1000:.1f}ms)')
    base.run()


if __name__ == '__main__':
    main()
