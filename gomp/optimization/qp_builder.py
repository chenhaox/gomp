"""
QP Builder: Construct the objective matrix P and decision variable layout.

From GOMP Section IV:
- Decision variable: x = [q_0, q_1, ..., q_H, v_0, v_1, ..., v_H]^T
- Size: 2*(H+1)*n where n = number of joints
- Objective: min ½ x^T P x (encourages smooth trajectories via sum-of-squared accelerations)

From Section IV-A:
    P = [[0, 0], [0, P_v]]
where P_v is block-tridiagonal with pattern [2, -1, 0, ...] (Eq. 8),
Kronecker product with I_n for n joints.
"""

import numpy as np
import scipy.sparse as sp


class QPBuilder:
    """
    Builds the QP objective matrix and manages the decision variable layout.

    The decision variable x has layout:
        x = [q_0, q_1, ..., q_H, v_0, v_1, ..., v_H]

    where each q_i and v_i is an n-dimensional vector (n = n_dof).

    Total size of x: 2 * (H+1) * n

    Parameters
    ----------
    H : int
        Number of time steps (trajectory has H+1 waypoints).
    n : int
        Number of degrees of freedom.
    """

    def __init__(self, H: int, n: int):
        self.H = H
        self.n = n
        self.n_waypoints = H + 1
        self.n_vars = 2 * self.n_waypoints * n  # total size of x

    # ----- Index helpers -----

    def q_idx(self, i: int) -> slice:
        """Slice for the q_i portion of x (waypoint i configuration)."""
        start = i * self.n
        return slice(start, start + self.n)

    def v_idx(self, i: int) -> slice:
        """Slice for the v_i portion of x (waypoint i velocity)."""
        start = self.n_waypoints * self.n + i * self.n
        return slice(start, start + self.n)

    def q_start_idx(self) -> int:
        """Starting index of the q block in x."""
        return 0

    def v_start_idx(self) -> int:
        """Starting index of the v block in x."""
        return self.n_waypoints * self.n

    # ----- Extract from x -----

    def extract_waypoints(self, x: np.ndarray) -> np.ndarray:
        """Extract all waypoint configurations from x. Shape: (H+1, n)."""
        q_block = x[:self.n_waypoints * self.n]
        return q_block.reshape(self.n_waypoints, self.n)

    def extract_velocities(self, x: np.ndarray) -> np.ndarray:
        """Extract all velocities from x. Shape: (H+1, n)."""
        v_block = x[self.n_waypoints * self.n:]
        return v_block.reshape(self.n_waypoints, self.n)

    def pack_x(self, waypoints: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Pack waypoints and velocities into decision variable x."""
        return np.concatenate([waypoints.ravel(), velocities.ravel()])

    # ----- Objective matrix -----

    def build_P(self) -> sp.csc_matrix:
        """
        Build the smoothness objective matrix P.

        From Section IV-A, Eq. (8):
            P = [[0, 0], [0, P_v]]

        P_v penalizes acceleration: min Σ ||v_{i+1} - v_i||²
        P_v is a block-tridiagonal matrix where the scalar pattern is:
            diag = 2, off-diag = -1
        And each block is n×n identity scaled by the tridiagonal value.

        Returns
        -------
        P : scipy.sparse.csc_matrix, shape (n_vars, n_vars)
            Positive semi-definite objective matrix.
        """
        n = self.n
        m = self.n_waypoints  # H + 1

        # Build the scalar tridiagonal pattern for P_v
        # This is (H+1) x (H+1) with 2 on diagonal, -1 on off-diagonals
        # But for acceleration penalty: ||v_{i+1} - v_i||^2
        # Expanding: Σ (v_{i+1} - v_i)^T (v_{i+1} - v_i)
        # = Σ v_{i+1}^T v_{i+1} - 2 v_{i+1}^T v_i + v_i^T v_i
        # In matrix form: v^T D^T D v where D is the first-difference matrix
        # D^T D has pattern: 1, -1 on sub/super diagonal → D^T D has 2 on diag, -1 off-diag
        # But boundary rows have 1 instead of 2

        # First-difference matrix D: (H) x (H+1), D[i] = v_{i+1} - v_i
        # D^T D is (H+1) x (H+1)
        diag_main = np.zeros(m)
        diag_off = np.zeros(m - 1)

        # Interior points get 2, boundary points get 1
        for i in range(m):
            if i == 0 or i == m - 1:
                diag_main[i] = 1.0
            else:
                diag_main[i] = 2.0
        diag_off[:] = -1.0

        # Build P_v as Kronecker product of scalar tridiagonal with I_n
        # P_v_scalar is (H+1) x (H+1)
        P_v_scalar = sp.diags([diag_off, diag_main, diag_off],
                              [-1, 0, 1], shape=(m, m), format='csc')

        # Kronecker product: P_v = P_v_scalar ⊗ I_n
        I_n = sp.eye(n, format='csc')
        P_v = sp.kron(P_v_scalar, I_n, format='csc')

        # Full P matrix: [[0, 0], [0, P_v]]
        q_size = m * n
        zero_qq = sp.csc_matrix((q_size, q_size))
        zero_qv = sp.csc_matrix((q_size, q_size))

        P = sp.bmat([
            [zero_qq, zero_qv],
            [zero_qv.T, P_v]
        ], format='csc')

        return P

    def build_p(self) -> np.ndarray:
        """
        Build the linear objective vector p.

        From the paper: p is set to 0 (no linear term in the objective).

        Returns
        -------
        p : np.ndarray, shape (n_vars,)
        """
        return np.zeros(self.n_vars)
