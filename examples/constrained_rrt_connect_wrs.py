"""UR3e 约束 RRT-connect 示例，仅使用 WRS Panda3D 可视化。

运行：python examples/constrained_rrt_connect_wrs.py
空格暂停/继续，鼠标调整视角。
"""

from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomp.planner import ConstrainedRRTConnect
from wrs import wd, mgm
from wrs.robot_sim.robots.ur3e_dual.ur3e_rtqhe import UR3e_RtqHE
from direct.gui.OnscreenText import OnscreenText


def check_link_poses(robot):
    """发现旧 native 包时立即报错，避免用错位的网格和碰撞体规划。"""
    for joint in robot.manipulator.jlc.jnts:
        expected_pos = joint.gl_pos_q + joint.gl_rotmat_q @ joint.lnk.loc_pos
        expected_rot = joint.gl_rotmat_q @ joint.lnk.loc_rotmat
        if (not np.allclose(joint.lnk.gl_pos, expected_pos, rtol=0, atol=1e-9)
                or not np.allclose(joint.lnk.gl_rotmat, expected_rot, rtol=0, atol=1e-9)):
            backend_file = sys.modules[type(robot.manipulator.jlc).__module__].__file__
            raise RuntimeError(
                "连杆坐标错误：请更新当前解释器中的 native 后端。\n"
                f"Python: {sys.executable}\nBackend: {backend_file}")


def main():
    random.seed(7)
    np.random.seed(7)
    robot = UR3e_RtqHE(enable_cc=True)
    middle = np.array([0., -.8, 1.2, -.4, 1.2, 0.])
    robot.goto_given_conf(middle)
    check_link_poses(robot)
    start, goal = middle.copy(), middle.copy()
    start[0], goal[0] = -.5, .5

    # 1. 定义任务约束：TCP 禁入球、高度范围、工具 z 轴方向容差。
    center = robot.gl_tcp_pos.copy()
    reference_axis = robot.gl_tcp_rotmat[:, 2].copy()
    radius, z_min, z_max = .08, -.02, .45
    max_tilt = np.radians(50)

    def task_constraint(robot, conf):
        # 调用前已完成 conf 的 FK；True 表示允许。关节限位和碰撞由规划器检查。
        pos = robot.gl_tcp_pos
        outside_sphere = np.linalg.norm(pos - center) >= radius
        within_height = z_min <= pos[2] <= z_max
        within_tilt = np.dot(robot.gl_tcp_rotmat[:, 2], reference_axis) >= np.cos(max_tilt)
        return bool(outside_sphere and within_height and within_tilt)

    # 2. 在原 WRS RRT-connect 基础上接入 callback。
    planner = ConstrainedRRTConnect(robot, task_constraint, check_resolution=.02)
    motion = planner.plan(start, goal, ext_dist=.2, max_time=30,
                          smoothing_n_iter=150, toggle_mesh=False)
    if motion is None:
        raise RuntimeError("未找到满足约束的路径，请调整约束范围或随机种子。")

    # 线性加密用于回放，并以 0.005 rad 步长复核所有路径段。
    path = [motion.jv_list[0]]
    for a, b in zip(motion.jv_list[:-1], motion.jv_list[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / .005)))
        path.extend(np.linspace(a, b, count + 1)[1:])
    positions = []
    for q in path:
        robot.goto_given_conf(q)
        if (not robot.are_jnts_in_ranges(q) or robot.is_collided()
                or not task_constraint(robot, q)):
            raise RuntimeError("路径密集复核未通过。")
        positions.append(robot.gl_tcp_pos.copy())
    print(f"规划成功，{len(path)} 个插值采样点通过约束及碰撞复核。")

    # 3. 显示约束区域和 TCP 路径；只回放一个机器人，避免残影干扰。
    base = wd.World(cam_pos=[-1.7, -1.8, 1.1], lookat_pos=[-.3, -.1, .18])
    base.setBackgroundColor(.95, .97, .99, 1)
    mgm.gen_frame(ax_length=.15).attach_to(base)
    mgm.gen_sphere(pos=center, radius=radius, rgb=[.85, .3, .25], alpha=.3).attach_to(base)
    for z in (z_min, z_max):
        mgm.gen_box(xyz_lengths=np.array([.95, 1.05, .002]),
                    pos=np.array([-.325, -.125, z]),
                    rgb=[.4, .65, .85], alpha=.12).attach_to(base)
    mgm.gen_arrow(spos=center, epos=center + .18 * reference_axis,
                  rgb=[.65, .45, .1]).attach_to(base)
    for a, b in zip(positions[:-1], positions[1:]):
        mgm.gen_stick(spos=a, epos=b, radius=.003, rgb=[0., .5, .55]).attach_to(base)
    for pos, color in ((positions[0], [.1, .7, .3]), (positions[-1], [.2, .3, .8])):
        mgm.gen_sphere(pos=pos, radius=.012, rgb=color).attach_to(base)
    OnscreenText(text="Constrained RRT-connect | Space: pause/resume\n"
                      f"Red: TCP keep-out sphere (r={radius:.2f} m)\n"
                      f"Blue: height bounds [{z_min:.2f}, {z_max:.2f}] m\n"
                      f"Gold: reference tool axis (tilt <= {np.degrees(max_tilt):.0f} deg)",
                  pos=(-1.25, .90), align=0, scale=.04, fg=(.12, .2, .3, 1))

    # 4. 循环回放；每帧仅保留一个当前机器人网格。
    index, mesh, paused = 0, None, False

    def toggle_pause():
        nonlocal paused
        paused = not paused

    def update(task):
        nonlocal index, mesh
        if not paused:
            if mesh is not None:
                mesh.detach()
            robot.goto_given_conf(path[index])
            mesh = robot.gen_meshmodel(toggle_tcp_frame=True)
            mesh.attach_to(base)
            index = (index + 1) % len(path)
        return task.again

    base.accept("space", toggle_pause)
    base.taskMgr.doMethodLater(.025, update, "constrained_rrt_playback")
    base.run()


if __name__ == "__main__":
    main()
