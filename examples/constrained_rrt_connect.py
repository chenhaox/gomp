"""Three callback constraints on a two-link arm, with constraint/path plots.

    python examples/constrained_rrt_connect.py
    python examples/constrained_rrt_connect.py --headless --gif

The small robot below implements the WRS RRT robot protocol. The planner is
the same ConstrainedRRTConnect used with the real WRS robot in the 3D example.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomp.planner.constrained_rrt_connect import ConstrainedRRTConnect


class PlanarArm:
    """Two revolute joints; lengths in meters, angles in radians."""

    limits = np.array([[-2.8, 2.8], [-2.8, 2.8]])
    oiee_list = []

    def __init__(self):
        self.q = np.zeros(2)
        self._states = []

    @staticmethod
    def points(q):
        q = np.asarray(q)
        elbow = 0.8 * np.array([np.cos(q[0]), np.sin(q[0])])
        tip = elbow + 0.6 * np.array([np.cos(q.sum()), np.sin(q.sum())])
        return np.array([[0, 0], elbow, tip])

    @property
    def gl_tcp_pos(self):
        return np.append(self.points(self.q)[-1], 0.0)

    @property
    def gl_tcp_rotmat(self):
        c, s = np.cos(self.q.sum()), np.sin(self.q.sum())
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def goto_given_conf(self, jnt_values):
        self.q = np.array(jnt_values, copy=True)

    def get_jnt_values(self):
        return self.q.copy()

    def backup_state(self):
        self._states.append(self.q.copy())

    def restore_state(self):
        self.q = self._states.pop()

    def rand_conf(self):
        return np.random.uniform(self.limits[:, 0], self.limits[:, 1])

    def are_jnts_in_ranges(self, jnt_values):
        return np.all((jnt_values >= self.limits[:, 0]) &
                      (jnt_values <= self.limits[:, 1]))

    def is_collided(self, obstacle_list, other_robot_list,
                    toggle_contacts=False, toggle_dbg=False):
        if other_robot_list:
            raise NotImplementedError("The teaching arm only supports circle obstacles")
        points = self.points(self.q)
        hit = False
        for center, radius in obstacle_list:
            for a, b in zip(points[:-1], points[1:]):
                t = np.clip(np.dot(np.asarray(center) - a, b - a) /
                            np.dot(b - a, b - a), 0, 1)
                hit |= np.linalg.norm(a + t * (b - a) - center) <= radius
        return (bool(hit), []) if toggle_contacts else bool(hit)

    def get_ee_values(self):
        return None

    def gen_meshmodel(self):
        return None  # Matplotlib draws this teaching robot.


def joint_margin(q):
    lower, upper = np.array([-1.6, 0.15]), np.array([1.6, 2.2])
    return float(min(np.min(q - lower), np.min(upper - q)))


def joint_window(robot, q):
    """Example 1: task-specific joint limits, inside hardware joint limits."""
    return joint_margin(q) >= 0


KEEP_OUT_CENTER = np.array([1.3, 0.0])
KEEP_OUT_RADIUS = 0.25


def tcp_keep_out(robot, q):
    """Example 2: a virtual workspace zone that the TCP must never enter."""
    return bool(np.linalg.norm(robot.gl_tcp_pos[:2] - KEEP_OUT_CENTER) >=
                KEEP_OUT_RADIUS)


ORIENTATION_TOLERANCE = 0.65
CARRY_CENTER = np.array([1.36, 0.0])
CARRY_RADIUS = 0.10


def orientation_error(q):
    return np.arctan2(np.sin(q.sum()), np.cos(q.sum()))


def tool_orientation(robot, q):
    """Example 3a: keep the tool's x axis within a finite angular tolerance."""
    angle = np.arctan2(robot.gl_tcp_rotmat[1, 0], robot.gl_tcp_rotmat[0, 0])
    return bool(abs(angle) <= ORIENTATION_TOLERANCE)


def carry_keep_out(robot, q):
    """Example 3b: compose orientation and TCP clearance in one callback."""
    return bool(np.linalg.norm(robot.gl_tcp_pos[:2] - CARRY_CENTER) >= CARRY_RADIUS)


def carry_constraint(robot, q):
    return all(check(robot, q) for check in (tool_orientation, carry_keep_out))


@dataclass
class Scenario:
    name: str
    title: str
    start: np.ndarray
    goal: np.ndarray
    callback: object
    metrics: tuple
    disk: object = None


def scenarios():
    return [
        Scenario("joint_window", "01 / Joint window", np.array([-1.2, 1.1]),
                 np.array([1.2, 1.1]), joint_window,
                 (("Joint margin (rad)", joint_margin),)),
        Scenario("tcp_keepout", "02 / TCP keep-out zone", np.array([-0.95, 0.8]),
                 np.array([0.8, -0.5]), tcp_keep_out,
                 (("TCP clearance (m)", lambda q: np.linalg.norm(
                     PlanarArm.points(q)[-1] - KEEP_OUT_CENTER) - KEEP_OUT_RADIUS),),
                 (KEEP_OUT_CENTER, KEEP_OUT_RADIUS)),
        Scenario("orientation", "03 / Carry with orientation tolerance",
                 np.array([-1., 1.]), np.array([1., -1.]), carry_constraint,
                 (("Orientation margin (rad)", lambda q:
                   ORIENTATION_TOLERANCE - abs(orientation_error(q))),
                  ("TCP clearance (m)", lambda q: np.linalg.norm(
                      PlanarArm.points(q)[-1] - CARRY_CENTER) - CARRY_RADIUS)),
                 (CARRY_CENTER, CARRY_RADIUS)),
    ]


def sample_path(path, resolution=0.005):
    """Linear interpolation, preserving all corners of the planned path."""
    samples = [np.array(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
        samples.extend(np.linspace(a, b, count + 1)[1:])
    return np.asarray(samples)


def draw_result(scenario, robot, planner, path, output_dir, save_gif, animate=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Wedge

    teal, coral, ink = "#087f8c", "#d45c54", "#23364d"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": ink,
                         "text.color": ink, "figure.facecolor": "#f6f8fb"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    fig.suptitle(scenario.title + "   |   Constrained RRT-connect", fontsize=17,
                 fontweight="bold")
    ax, workspace, margins = axes
    grid = np.linspace(-2.8, 2.8, 141)
    mask = np.zeros((len(grid), len(grid)))
    robot.backup_state()
    try:
        for y, q2 in enumerate(grid):
            for x, q1 in enumerate(grid):
                q = np.array([q1, q2])
                robot.goto_given_conf(q)
                mask[y, x] = not scenario.callback(robot, q)
    finally:
        robot.restore_state()
    ax.imshow(mask, origin="lower", extent=[-2.8, 2.8, -2.8, 2.8],
              cmap=ListedColormap(["#e4f2ee", "#f6dcd8"]), vmin=0, vmax=1)
    for tree, color in ((planner.roadmap_start, "#97bfca"),
                        (planner.roadmap_goal, "#b7adc9")):
        segments = [[tree.nodes[a]["conf"], tree.nodes[b]["conf"]]
                    for a, b in tree.edges]
        ax.add_collection(LineCollection(segments, colors=color, linewidths=.5, alpha=.5))
    direct = np.linspace(scenario.start, scenario.goal, 301)
    ax.plot(*direct.T, "--", color=coral, lw=1.6, label="Straight interpolation")
    ax.plot(*path.T, color=teal, lw=2.5, label="Validated path")
    ax.scatter(*scenario.start, color="#2c955e", s=60, zorder=6, label="Start")
    ax.scatter(*scenario.goal, color=ink, marker="*", s=110, zorder=6, label="Goal")
    moving_q, = ax.plot([], [], "o", color=teal, ms=9)
    ax.set(xlabel="Joint 1 (rad)", ylabel="Joint 2 (rad)",
           title="Configuration space / red = rejected", xlim=(-2.8, 2.8), ylim=(-2.8, 2.8))
    ax.legend(loc="lower left", fontsize=8, framealpha=.95)

    dense = sample_path(path)
    tcp = np.array([robot.points(q)[-1] for q in dense])
    direct_tcp = np.array([robot.points(q)[-1] for q in direct])
    workspace.plot(*direct_tcp.T, "--", color=coral, lw=1.6)
    workspace.plot(*tcp.T, color=teal, lw=2.5, label="TCP path")
    if scenario.disk is not None:
        center, radius = scenario.disk
        workspace.add_patch(Circle(center, radius, facecolor="#f6dcd8",
                                   edgecolor=coral, lw=1.5, hatch="///"))
        workspace.text(-1.45, -1.42, "Hatched disk: TCP-only task constraint", fontsize=8)
    for index in np.linspace(0, len(dense) - 1, 7, dtype=int):
        workspace.plot(*robot.points(dense[index]).T, "o-", color=teal,
                       alpha=.13 + .20 * index / max(1, len(dense) - 1), lw=2, ms=3)
    if scenario.name == "orientation":
        tolerance = np.degrees(ORIENTATION_TOLERANCE)
        workspace.add_patch(Wedge((-1.0, 1.05), .4, -tolerance, tolerance,
                                  color="#a9d5bf", alpha=.8))
        workspace.text(-1.45, 1.4, f"Tool x-axis: 0 +/- {tolerance:.1f} deg", fontsize=9)
        for index in np.linspace(0, len(dense) - 1, 7, dtype=int):
            angle = dense[index].sum()
            workspace.arrow(*tcp[index], .16 * np.cos(angle), .16 * np.sin(angle),
                            head_width=.035, color="#927149", length_includes_head=True)
    moving_arm, = workspace.plot([], [], "o-", color=ink, lw=3, ms=5)
    workspace.scatter(*tcp[0], color="#2c955e", s=60, zorder=6)
    workspace.scatter(*tcp[-1], color=ink, marker="*", s=110, zorder=6)
    workspace.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), aspect="equal",
                  xlabel="x (m)", ylabel="y (m)", title="Workspace / arm poses + TCP path")
    workspace.grid(alpha=.15)

    progress = np.linspace(0, 1, len(dense))
    minimums = []
    for (label, metric), color in zip(scenario.metrics, [teal, "#a27731"]):
        values = np.array([metric(q) for q in dense])
        minimums.append(values.min())
        margins.plot(progress, values, color=color, lw=2,
                     label=f"{label}; min={values.min():.3f}")
    margins.axhline(0, color=coral, ls="--", lw=1.5, label="Validity boundary")
    ymin, ymax = margins.get_ylim()
    margins.axhspan(min(ymin, -.08), 0, color="#f6dcd8", alpha=.65)
    cursor = margins.axvline(0, color=ink, alpha=.5, lw=1)
    margins.set(xlim=(0, 1), xlabel="Path progress (not time)", ylabel="Constraint margin",
                title="Validation / every margin must be >= 0")
    margins.grid(alpha=.15)
    margins.legend(loc="upper right", fontsize=8)
    margins.text(.03, .04, f"PASS  |  {len(dense)} independently checked samples\n"
                 f"Search callback rejections: {planner.constraint_rejections}",
                 transform=margins.transAxes, fontsize=9,
                 bbox={"facecolor": "#e4f2ee", "edgecolor": "none", "pad": 8})
    fig.savefig(output_dir / f"{scenario.name}.png", dpi=150)
    np.savez(output_dir / f"{scenario.name}.npz", path=path,
             validation_path=dense, min_margins=np.array(minimums))

    indices = np.linspace(0, len(dense) - 1, 90, dtype=int)

    def update(frame):
        i = indices[frame]
        moving_q.set_data([dense[i, 0]], [dense[i, 1]])
        moving_arm.set_data(*robot.points(dense[i]).T)
        cursor.set_xdata([progress[i], progress[i]])
        return moving_q, moving_arm, cursor

    animation = (FuncAnimation(fig, update, frames=len(indices), interval=45, blit=False)
                 if animate or save_gif else None)
    if save_gif:
        animation.save(output_dir / f"{scenario.name}.gif", writer=PillowWriter(fps=22), dpi=85)
    return fig, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", "joint_window", "tcp_keepout", "orientation"], default="all")
    parser.add_argument("--headless", action="store_true", help="Save results without opening a window")
    parser.add_argument("--gif", action="store_true", help="Also export animated robot playback")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/constrained_rrt"))
    args = parser.parse_args()
    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    animations = []  # Keep live animations alive until windows close.
    for scenario in scenarios():
        if args.scenario not in ("all", scenario.name):
            continue
        random.seed(args.seed)
        np.random.seed(args.seed)
        robot = PlanarArm()
        planner = ConstrainedRRTConnect(robot, scenario.callback, check_resolution=.025)
        motion = planner.plan(scenario.start, scenario.goal, ext_dist=.18,
                              max_time=30, smoothing_n_iter=150, toggle_mesh=False)
        if motion is None:
            raise RuntimeError(f"No path found for {scenario.name}; try another --seed")
        path = np.array(motion.jv_list)
        # Independently verify more finely than the planner, including edges.
        dense = sample_path(path, resolution=.005)
        robot.backup_state()
        try:
            for q in dense:
                robot.goto_given_conf(q)
                if (not robot.are_jnts_in_ranges(q) or
                        robot.is_collided([], []) or not scenario.callback(robot, q)):
                    raise RuntimeError(f"Dense validation failed for {scenario.name}")
        finally:
            robot.restore_state()
        fig, animation = draw_result(scenario, robot, planner, path, args.output_dir,
                                     args.gif, animate=not args.headless)
        animations.append(animation)
        print(f"{scenario.name}: PASS, {len(path)} waypoints, {len(dense)} validation samples")
        if args.headless:
            plt.close(fig)
    print(f"Results: {args.output_dir.resolve()}")
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
