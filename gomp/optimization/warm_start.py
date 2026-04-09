"""
Warm-starting utilities for GOMP.

From GOMP Section IV-F:
1. Initial warm start: cubic spline interpolating q_0 to q_H with zero velocity
2. Time reduction warm start: interpolate previous solution to shorter trajectory
"""

import numpy as np
from scipy.interpolate import CubicSpline


def spline_warm_start(q_start: np.ndarray, q_goal: np.ndarray,
                      H: int, n: int) -> np.ndarray:
    """
    Create initial warm start via cubic spline interpolation.

    Generates a smooth trajectory from q_start to q_goal with zero
    velocity at both endpoints, without considering obstacles.

    Parameters
    ----------
    q_start : np.ndarray, shape (n,)
        Start joint configuration.
    q_goal : np.ndarray, shape (n,)
        Goal joint configuration.
    H : int
        Number of time steps.
    n : int
        Number of joints.

    Returns
    -------
    x : np.ndarray, shape (2*(H+1)*n,)
        Decision variable with spline-interpolated waypoints and velocities.
    """
    m = H + 1  # number of waypoints
    t = np.linspace(0, 1, m)

    # Cubic spline with zero-derivative boundary conditions
    # For each joint, interpolate between start and goal
    waypoints = np.zeros((m, n))
    velocities = np.zeros((m, n))

    for j in range(n):
        # Create cubic spline with clamped boundaries (zero velocity)
        cs = CubicSpline(
            [0, 1],
            [q_start[j], q_goal[j]],
            bc_type='clamped'  # zero derivative at both ends
        )
        waypoints[:, j] = cs(t)
        # Velocity is the derivative, but scaled by t_step
        # The spline parameter goes from 0 to 1 over H steps
        # Real velocity = dq/dt_real = (dq/ds) * (ds/dt_real) = (dq/ds) / (H * t_step)
        # But since we set t_step implicitly, we store the finite difference here
        velocities[:, j] = cs(t, 1) / H  # Scale derivative to per-step velocity

    # Pack into x = [q0, q1, ..., qH, v0, v1, ..., vH]
    x = np.concatenate([waypoints.ravel(), velocities.ravel()])

    # Force zero velocity at endpoints
    v_start_idx = m * n
    x[v_start_idx:v_start_idx + n] = 0.0  # v_0 = 0
    x[-n:] = 0.0  # v_H = 0

    return x


def interpolate_to_shorter(x_prev: np.ndarray, H_prev: int,
                           H_new: int, n: int,
                           t_step: float) -> np.ndarray:
    """
    Interpolate a trajectory solution to a shorter time horizon.

    When reducing H, the previous solution's waypoints don't align with
    the new discretization. This function interpolates the previous
    solution to fit the shorter trajectory.

    Parameters
    ----------
    x_prev : np.ndarray
        Previous solution decision variable.
    H_prev : int
        Previous number of time steps.
    H_new : int
        New (shorter) number of time steps.
    n : int
        Number of joints.
    t_step : float
        Time step (same for both trajectories).

    Returns
    -------
    x_new : np.ndarray
        Interpolated decision variable for the shorter trajectory.
    """
    m_prev = H_prev + 1
    m_new = H_new + 1

    # Extract previous waypoints and velocities
    q_prev = x_prev[:m_prev * n].reshape(m_prev, n)
    v_prev = x_prev[m_prev * n:].reshape(m_prev, n)

    # Time arrays
    t_prev = np.linspace(0, H_prev * t_step, m_prev)
    t_new = np.linspace(0, H_new * t_step, m_new)

    # Interpolate each joint using cubic spline
    q_new = np.zeros((m_new, n))
    v_new = np.zeros((m_new, n))

    for j in range(n):
        # Interpolate positions
        cs = CubicSpline(t_prev, q_prev[:, j])
        q_new[:, j] = cs(t_new)
        v_new[:, j] = cs(t_new, 1) * t_step  # Convert to per-step velocity

    # Enforce zero velocity at endpoints
    v_new[0, :] = 0.0
    v_new[-1, :] = 0.0

    return np.concatenate([q_new.ravel(), v_new.ravel()])
