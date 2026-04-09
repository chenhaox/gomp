"""
Joint-space plots for GOMP trajectories.

Produces plots matching Fig. 5 from the paper:
- Joint velocities vs time
- Joint accelerations vs time
- Overlaid limit lines
- Side-by-side comparison for different H values
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from gomp.planner.trajectory import Trajectory


# Joint colors matching the paper's style
JOINT_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#984ea3',  # purple
    '#ff7f00',  # orange
    '#a65628',  # brown
]

JOINT_LABELS = [f'Joint {i+1}' for i in range(6)]


def plot_joint_profiles(trajectory: Trajectory,
                        v_max: np.ndarray = None,
                        a_max: np.ndarray = None,
                        title: str = None,
                        save_path: str = None):
    """
    Plot joint velocities and accelerations for a trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    v_max : np.ndarray or None
        Velocity limits to overlay as dashed lines.
    a_max : np.ndarray or None
        Acceleration limits to overlay as dashed lines.
    title : str or None
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    t = trajectory.time_array()
    n = trajectory.n_dof
    accel = trajectory.accelerations
    t_accel = t[:-1] + trajectory.t_step / 2  # centered between waypoints

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Velocities ---
    for j in range(min(n, len(JOINT_COLORS))):
        color = JOINT_COLORS[j % len(JOINT_COLORS)]
        ax1.plot(t * 1000, trajectory.velocities[:, j],
                color=color, linewidth=1.5, label=JOINT_LABELS[j])

    if v_max is not None:
        for j in range(min(n, len(JOINT_COLORS))):
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            ax1.axhline(y=v_max[j], color=color, linestyle='--',
                       alpha=0.3, linewidth=0.8)
            ax1.axhline(y=-v_max[j], color=color, linestyle='--',
                       alpha=0.3, linewidth=0.8)

    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Joint Velocities')

    # --- Accelerations ---
    for j in range(min(n, len(JOINT_COLORS))):
        color = JOINT_COLORS[j % len(JOINT_COLORS)]
        ax2.plot(t_accel * 1000, accel[:, j],
                color=color, linewidth=1.5, label=JOINT_LABELS[j])

    if a_max is not None:
        for j in range(min(n, len(JOINT_COLORS))):
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            ax2.axhline(y=a_max[j], color=color, linestyle='--',
                       alpha=0.3, linewidth=0.8)
            ax2.axhline(y=-a_max[j], color=color, linestyle='--',
                       alpha=0.3, linewidth=0.8)

    ax2.set_ylabel('Acceleration (rad/s²)')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Joint Accelerations')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_time_optimization(trajectories: List[Trajectory],
                           v_max: np.ndarray = None,
                           a_max: np.ndarray = None,
                           save_path: str = None):
    """
    Plot multiple trajectories side-by-side showing the time optimization process.

    Matches Fig. 5 from the paper: shows how reducing H produces shorter
    trajectories until the SQP becomes infeasible.

    Parameters
    ----------
    trajectories : list of Trajectory
        Trajectories at different H values (sorted by decreasing H).
    v_max : np.ndarray or None
        Velocity limits.
    a_max : np.ndarray or None
        Acceleration limits.
    save_path : str or None
        If provided, save the figure.
    """
    n_traj = len(trajectories)
    if n_traj == 0:
        return

    fig, axes = plt.subplots(2, n_traj, figsize=(4 * n_traj, 6),
                              squeeze=False)

    for col, traj in enumerate(trajectories):
        t = traj.time_array()
        n = traj.n_dof
        accel = traj.accelerations
        t_accel = t[:-1] + traj.t_step / 2

        # Velocities
        ax1 = axes[0, col]
        for j in range(min(n, len(JOINT_COLORS))):
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            ax1.plot(t * 1000, traj.velocities[:, j],
                    color=color, linewidth=1.2)

        if v_max is not None:
            for j in range(min(n, len(JOINT_COLORS))):
                ax1.axhline(y=v_max[j], color=JOINT_COLORS[j],
                           linestyle='--', alpha=0.2, linewidth=0.6)
                ax1.axhline(y=-v_max[j], color=JOINT_COLORS[j],
                           linestyle='--', alpha=0.2, linewidth=0.6)

        ax1.set_title(f'H = {traj.H}\n({traj.duration*1000:.0f} ms)',
                     fontsize=10)
        ax1.grid(True, alpha=0.3)
        if col == 0:
            ax1.set_ylabel('Velocity (rad/s)')

        # Accelerations
        ax2 = axes[1, col]
        for j in range(min(n, len(JOINT_COLORS))):
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            ax2.plot(t_accel * 1000, accel[:, j],
                    color=color, linewidth=1.2)

        if a_max is not None:
            for j in range(min(n, len(JOINT_COLORS))):
                ax2.axhline(y=a_max[j], color=JOINT_COLORS[j],
                           linestyle='--', alpha=0.2, linewidth=0.6)
                ax2.axhline(y=-a_max[j], color=JOINT_COLORS[j],
                           linestyle='--', alpha=0.2, linewidth=0.6)

        ax2.set_xlabel('Time (ms)')
        ax2.grid(True, alpha=0.3)
        if col == 0:
            ax2.set_ylabel('Acceleration (rad/s²)')

    fig.suptitle('Time Optimization: Reducing H', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
