"""UR3e + gripper: TCP keep-out sphere, height band and tool-axis tolerance.

    python examples/constrained_rrt_connect_wrs.py --headless
    python examples/constrained_rrt_connect_wrs.py --gif
    python examples/constrained_rrt_connect_wrs.py --panda3d

Uses WRS full-body/self collision checking in addition to the task callback.
PNG/NPZ are always saved; GIF and the WRS mesh viewer are optional.
"""

import argparse
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomp.planner.constrained_rrt_connect import ConstrainedRRTConnect
from wrs.robot_sim.robots.ur3e_dual.ur3e_rtqhe import UR3e_RtqHE
from constrained_rrt_connect import sample_path


class CarryConstraints:
    """A callable object can keep task parameters alongside its callback."""

    z_min, z_max = -0.02, 0.45
    radius = 0.08
    max_tilt = np.radians(50)

    def __init__(self, center, reference_axis):
        self.center = np.array(center, copy=True)
        self.reference_axis = np.array(reference_axis, copy=True)

    def margins(self, robot):
        pos = robot.gl_tcp_pos
        tilt = np.arccos(np.clip(np.dot(robot.gl_tcp_rotmat[:, 2],
                                       self.reference_axis), -1, 1))
        return np.array([np.linalg.norm(pos - self.center) - self.radius,
                         min(pos[2] - self.z_min, self.z_max - pos[2]),
                         self.max_tilt - tilt])

    def __call__(self, robot, conf):
        # FK is already at conf. True = valid. These are task constraints;
        # WRS joint limits and full-body collision are checked by the planner.
        return bool(np.all(self.margins(robot) >= 0))


def record_path(robot, path, constraints):
    positions, axes, links, margins = [], [], [], []
    robot.backup_state()
    try:
        for q in path:
            robot.goto_given_conf(q)
            if (not robot.are_jnts_in_ranges(q) or robot.is_collided() or
                    not constraints(robot, q)):
                raise RuntimeError("Independent dense validation failed")
            positions.append(robot.gl_tcp_pos.copy())
            axes.append(robot.gl_tcp_rotmat[:, 2].copy())
            links.append(np.vstack([robot.pos,
                                    *[joint.gl_pos_q for joint in robot.manipulator.jlc.jnts],
                                    robot.gl_tcp_pos]))
            margins.append(constraints.margins(robot))
    finally:
        robot.restore_state()
    return tuple(np.asarray(values) for values in (positions, axes, links, margins))


def plot_result(path, positions, axes, links, margins, constraints, output_dir,
                save_gif, interactive):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "figure.facecolor": "#f6f8fb",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    workspace = fig.add_subplot(grid[:, 0], projection="3d")
    fig.suptitle("UR3e / Carry through a constrained workspace", fontsize=19, fontweight="bold")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 20))
    sphere = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    sphere = constraints.center[:, None, None] + constraints.radius * sphere
    workspace.plot_surface(*sphere, color="#d45c54", alpha=.35, linewidth=0)
    xx, yy = np.meshgrid([-.8, .15], [-.65, .4])
    for z in (constraints.z_min, constraints.z_max):
        workspace.plot_surface(xx, yy, np.full_like(xx, z), color="#93b5d6", alpha=.10)
    workspace.plot(*positions.T, color="#087f8c", lw=3)
    for i in np.linspace(0, len(path) - 1, 6, dtype=int):
        workspace.plot(*links[i].T, "o-", color="#467994", lw=2, ms=3, alpha=.22)
        workspace.quiver(*positions[i], *axes[i], length=.09, color="#a27731")
    workspace.quiver(*constraints.center, *constraints.reference_axis, length=.2,
                     color="#9d7527", linewidth=2)
    workspace.scatter(*positions[0], color="#2c955e", s=70)
    workspace.scatter(*positions[-1], color="#23364d", marker="*", s=100)
    workspace.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                  xlim=(-.85, .2), ylim=(-.7, .45), zlim=(-.1, .6),
                  title="TCP exclusion sphere + allowed height slab")
    workspace.set_box_aspect([1.05, 1.15, .7])
    workspace.view_init(elev=24, azim=135)
    workspace.legend(handles=[
        Line2D([], [], color="#087f8c", lw=3, label="Validated TCP path"),
        Line2D([], [], color="#d45c54", lw=6, alpha=.5, label="TCP-only exclusion sphere"),
        Line2D([], [], color="#a27731", lw=2, label="Tool z-axis / reference direction"),
    ], loc="lower left", fontsize=9)
    arm, = workspace.plot([], [], [], "o-", color="#23364d", lw=3, ms=4)
    marker, = workspace.plot([], [], [], "o", color="#087f8c", ms=8)

    progress = np.linspace(0, 1, len(path))
    labels = ["TCP clearance (m)", "Height margin (m)", "Tool-axis margin (deg)"]
    cursors = []
    for col, label in enumerate(labels):
        ax = fig.add_subplot(grid[col, 1])
        values = np.degrees(margins[:, col]) if col == 2 else margins[:, col]
        ax.plot(progress, values, lw=2, color="#087f8c")
        ax.axhline(0, color="#d45c54", ls="--")
        ax.set(xlim=(0, 1), ylabel=label, title=f"Minimum: {values.min():.4f}")
        ax.grid(alpha=.15)
        cursors.append(ax.axvline(0, color="#23364d", alpha=.5))
        if col == 2:
            ax.set_xlabel("Path progress (not time)")
    fig.supxlabel(f"PASS / {len(path)} samples checked at 0.005 rad, including WRS collision checks. "
                  f"Tool tilt limit: {np.degrees(constraints.max_tilt):.0f} deg.", fontsize=10)
    fig.savefig(output_dir / "ur3e_constraints.png", dpi=150)
    indices = np.linspace(0, len(path) - 1, 90, dtype=int)

    def update(frame):
        i = indices[frame]
        arm.set_data_3d(*links[i].T)
        marker.set_data_3d(*positions[i, :, None])
        for cursor in cursors:
            cursor.set_xdata([progress[i], progress[i]])
        return arm, marker, *cursors

    animation = None
    if interactive or save_gif:
        animation = FuncAnimation(fig, update, frames=len(indices), interval=45)
        if save_gif:
            animation.save(output_dir / "ur3e_constraints.gif", writer=PillowWriter(fps=22), dpi=85)
    return fig, animation


def show_panda3d(robot, path, positions, constraints):
    """Actual WRS meshes; Space pauses/resumes, mouse rotates the camera."""
    from wrs import wd, mgm
    from direct.gui.OnscreenText import OnscreenText

    base = wd.World(cam_pos=[-1.7, -1.8, 1.1], lookat_pos=[-.3, -.1, .18])
    base.setBackgroundColor(.95, .97, .99, 1)
    mgm.gen_frame(ax_length=.15).attach_to(base)
    mgm.gen_sphere(pos=constraints.center, radius=constraints.radius,
                   rgb=np.array([.85, .3, .25]), alpha=.3).attach_to(base)
    for z in (constraints.z_min, constraints.z_max):
        mgm.gen_box(xyz_lengths=np.array([.95, 1.05, .002]),
                    pos=np.array([-.325, -.125, z]),
                    rgb=np.array([.4, .65, .85]), alpha=.12).attach_to(base)
    trail = positions[np.unique(np.linspace(0, len(positions) - 1,
                                            min(len(positions), 100), dtype=int))]
    for a, b in zip(trail[:-1], trail[1:]):
        mgm.gen_stick(spos=a, epos=b, radius=.003,
                      rgb=np.array([0., .5, .55])).attach_to(base)
    mgm.gen_arrow(spos=constraints.center,
                  epos=constraints.center + .18 * constraints.reference_axis,
                  rgb=np.array([.65, .45, .1])).attach_to(base)
    for i in np.linspace(0, len(path) - 1, 5, dtype=int):
        robot.goto_given_conf(path[i])
        robot.gen_meshmodel(alpha=.14).attach_to(base)
    status = OnscreenText(text="", pos=(-1.25, .90), align=0, scale=.045,
                          fg=(.12, .2, .3, 1), mayChange=True)
    state = {"index": 0, "mesh": None, "paused": False}

    def toggle_pause():
        state["paused"] = not state["paused"]

    def update(task):
        if state["mesh"] is not None:
            state["mesh"].detach()
        q = path[state["index"]]
        robot.goto_given_conf(q)
        state["mesh"] = robot.gen_meshmodel(toggle_tcp_frame=True)
        state["mesh"].attach_to(base)
        clearance, height, tilt = constraints.margins(robot)
        status.setText("Constrained RRT-connect | Space: pause\n"
                       f"TCP clearance: {clearance:.3f} m | Height margin: {height:.3f} m\n"
                       f"Tool-axis margin: {np.degrees(tilt):.1f} deg / 50 deg tolerance")
        if not state["paused"]:
            state["index"] = (state["index"] + 3) % len(path)
        return task.again

    base.accept("space", toggle_pause)
    base.taskMgr.doMethodLater(.04, update, "constrained_rrt_playback")
    base.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--panda3d", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/constrained_rrt"))
    args = parser.parse_args()
    if args.headless and args.panda3d:
        parser.error("--headless and --panda3d are mutually exclusive")
    import matplotlib
    if args.headless or args.panda3d:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = UR3e_RtqHE(enable_cc=True)
    middle = np.array([0., -.8, 1.2, -.4, 1.2, 0.])
    robot.goto_given_conf(middle)
    constraints = CarryConstraints(robot.gl_tcp_pos, robot.gl_tcp_rotmat[:, 2])
    start, goal = middle.copy(), middle.copy()
    start[0], goal[0] = -.5, .5

    planner = ConstrainedRRTConnect(robot, constraints, check_resolution=.02)
    motion = planner.plan(start, goal, ext_dist=.2, max_time=30,
                          smoothing_n_iter=150, toggle_mesh=False)
    if motion is None:
        raise RuntimeError("No path found. Try another --seed or a wider task tolerance.")
    path = sample_path(np.array(motion.jv_list), resolution=.005)
    positions, axes, links, margins = record_path(robot, path, constraints)
    np.savez(args.output_dir / "ur3e_constraints.npz", path=path, tcp=positions,
             margins=margins, sphere_center=constraints.center,
             sphere_radius=constraints.radius, reference_axis=constraints.reference_axis,
             max_tilt=constraints.max_tilt, height_band=[constraints.z_min, constraints.z_max])
    fig, animation = plot_result(path, positions, axes, links, margins, constraints,
                                 args.output_dir, args.gif,
                                 interactive=not (args.headless or args.panda3d))
    print(f"UR3e: PASS, {len(path)} dense samples; min margins (m, m, rad): {margins.min(axis=0)}")
    print(f"Results: {args.output_dir.resolve()}")
    if args.panda3d:
        plt.close(fig)
        show_panda3d(robot, path, positions, constraints)
    elif args.headless:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
