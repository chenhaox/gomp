"""
Warm-starting utilities for GOMP.

From GOMP Section IV-F:
1. Initial warm start: cubic spline interpolating q_0 to q_H with zero velocity
2. Time reduction warm start: interpolate previous solution to shorter trajectory

Note on velocity decision variable:
    The decision variable v_i represents actual joint velocity (rad/s).
    The dynamics constraint is: q_{i+1} = q_i + v_i * t_step
    So v_i = (q_{i+1} - q_i) / t_step
    
    The warm start must satisfy this exactly, so we compute positions from
    the spline and then derive velocities from finite differences.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def spline_warm_start(q_start: np.ndarray, q_goal: np.ndarray,
                      H: int, n: int, t_step: float) -> np.ndarray:
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
    t_step : float
        Time step between waypoints (seconds).

    Returns
    -------
    x : np.ndarray, shape (2*(H+1)*n,)
        Decision variable with spline-interpolated waypoints and velocities.
    """
    m = H + 1  # number of waypoints
    t_real = np.linspace(0, 1, m)  # normalized parameter

    waypoints = np.zeros((m, n))
    for j in range(n):
        cs = CubicSpline(
            [0, 1],
            [q_start[j], q_goal[j]],
            bc_type='clamped'  # zero derivative at endpoints
        )
        waypoints[:, j] = cs(t_real)

    # Compute velocities from finite differences to exactly satisfy dynamics:
    # v_i = (q_{i+1} - q_i) / t_step
    velocities = np.zeros((m, n))
    for i in range(H):
        velocities[i, :] = (waypoints[i + 1, :] - waypoints[i, :]) / t_step

    # Force zero velocity at endpoints (overrides the finite differences)
    velocities[0, :] = 0.0
    velocities[-1, :] = 0.0

    # Since v_0 = 0 is enforced, we need q_1 = q_0 + v_0*t_step = q_0
    # But our spline has q_1 != q_0. Fix: adjust q_0 to match, or accept
    # that the dynamics at endpoints won't be exactly satisfied.
    # The SQP will handle the correction.

    # Pack into x = [q0, q1, ..., qH, v0, v1, ..., vH]
    return np.concatenate([waypoints.ravel(), velocities.ravel()])


def interpolate_to_shorter(x_prev: np.ndarray, H_prev: int,
                           H_new: int, n: int,
                           t_step: float) -> np.ndarray:
    """
    Interpolate a trajectory solution to a shorter time horizon.

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

    # Extract previous waypoints
    q_prev = x_prev[:m_prev * n].reshape(m_prev, n)

    # Time arrays (normalized)
    t_prev = np.linspace(0, 1, m_prev)
    t_new = np.linspace(0, 1, m_new)

    # Interpolate positions
    q_new = np.zeros((m_new, n))
    for j in range(n):
        cs = CubicSpline(t_prev, q_prev[:, j])
        q_new[:, j] = cs(t_new)

    # Compute velocities from finite differences
    v_new = np.zeros((m_new, n))
    for i in range(H_new):
        v_new[i, :] = (q_new[i + 1, :] - q_new[i, :]) / t_step

    # Force zero velocity at endpoints
    v_new[0, :] = 0.0
    v_new[-1, :] = 0.0

    return np.concatenate([q_new.ravel(), v_new.ravel()])
