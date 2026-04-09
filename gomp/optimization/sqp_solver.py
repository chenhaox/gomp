"""
Sequential Quadratic Programming (SQP) solver for GOMP.

From GOMP Section IV: The SQP method solves a sequence of QP subproblems,
iteratively updating trust regions and linearized constraints until
convergence or infeasibility.

Uses OSQP as the underlying QP solver for its warm-starting capabilities
and infeasibility detection.
"""

import numpy as np
import osqp
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional

from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.constraints import ConstraintBuilder
from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.collision import ObstacleConstraint


@dataclass
class SQPResult:
    """
    Result of the SQP optimization.

    Attributes
    ----------
    x : np.ndarray or None
        Solution decision variable, or None if infeasible.
    feasible : bool
        Whether a feasible solution was found.
    n_iterations : int
        Number of SQP iterations performed.
    final_cost : float
        Final objective value.
    constraint_violation : float
        Maximum constraint violation at convergence.
    """
    x: Optional[np.ndarray]
    feasible: bool
    n_iterations: int
    final_cost: float
    constraint_violation: float


class SQPSolver:
    """
    SQP solver for the GOMP trajectory optimization.

    Iteratively solves QP subproblems with updated linearizations
    of the obstacle and grasp constraints, and adaptively sized
    trust regions.

    Parameters
    ----------
    max_iterations : int
        Maximum number of SQP iterations.
    constraint_tol : float
        Tolerance for constraint satisfaction.
    trust_region_tol : float
        Tolerance for trust region convergence.
    initial_trust_region : float
        Initial trust region radius.
    trust_region_expand : float
        Factor to expand trust region on improvement.
    trust_region_shrink : float
        Factor to shrink trust region on violation.
    verbose : bool
        Print iteration info.
    """

    def __init__(self,
                 max_iterations: int = 50,
                 constraint_tol: float = 1e-4,
                 trust_region_tol: float = 1e-3,
                 initial_trust_region: float = 1.0,
                 trust_region_expand: float = 1.5,
                 trust_region_shrink: float = 0.5,
                 verbose: bool = False):
        self.max_iterations = max_iterations
        self.constraint_tol = constraint_tol
        self.trust_region_tol = trust_region_tol
        self.initial_trust_region = initial_trust_region
        self.trust_region_expand = trust_region_expand
        self.trust_region_shrink = trust_region_shrink
        self.verbose = verbose

    def solve(self, x_init: np.ndarray,
              H: int, n: int, t_step: float,
              robot: RobotAdapter,
              obstacle_constraint: Optional[ObstacleConstraint] = None,
              start_grasp_set: Optional[GraspSet] = None,
              goal_grasp_set: Optional[GraspSet] = None) -> SQPResult:
        """
        Run the SQP loop for a fixed horizon H.

        Parameters
        ----------
        x_init : np.ndarray
            Initial guess for the decision variable.
        H : int
            Time horizon (number of steps).
        n : int
            Number of DOFs.
        t_step : float
            Time step between waypoints.
        robot : RobotAdapter
            Robot model.
        obstacle_constraint : ObstacleConstraint or None
            Obstacle model.
        start_grasp_set : GraspSet or None
            Start grasp set with DOF.
        goal_grasp_set : GraspSet or None
            Goal grasp set with DOF.

        Returns
        -------
        SQPResult
            The result of the optimization.
        """
        qp = QPBuilder(H, n)
        cb = ConstraintBuilder(qp, t_step, robot)

        # Build objective (doesn't change)
        P = qp.build_P()
        p = qp.build_p()

        # Build static constraints (don't change between SQP iterations)
        A_static, l_static, u_static = cb.build_static_constraints()

        # Initialize
        x_current = x_init.copy()
        trust_radius = self.initial_trust_region
        osqp_solver = None

        for iteration in range(self.max_iterations):
            # Extract current waypoints for linearization
            waypoints = qp.extract_waypoints(x_current)

            # Build dynamic constraints (updated each iteration)
            A_dynamic, l_dynamic, u_dynamic = cb.build_dynamic_constraints(
                waypoints=waypoints,
                obstacle_constraint=obstacle_constraint,
                start_grasp_set=start_grasp_set,
                goal_grasp_set=goal_grasp_set,
                trust_region=trust_radius
            )

            # Stack all constraints
            A = sp.vstack([A_static, A_dynamic], format='csc')
            l = np.concatenate([l_static, l_dynamic])
            u = np.concatenate([u_static, u_dynamic])

            # Solve QP
            if osqp_solver is None:
                # First iteration: create solver
                osqp_solver = osqp.OSQP()
                osqp_solver.setup(
                    P=sp.triu(P, format='csc'),
                    q=p,
                    A=A,
                    l=l,
                    u=u,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    max_iter=4000,
                    warm_start=True,
                    verbose=False,
                    polish=True
                )
            else:
                # Subsequent iterations: update and warm start
                # Resize if constraint matrix changed shape
                try:
                    osqp_solver.update(
                        q=p,
                        l=l,
                        u=u,
                        Ax=A.data
                    )
                except Exception:
                    # If update fails (e.g., sparsity pattern changed),
                    # recreate the solver
                    osqp_solver = osqp.OSQP()
                    osqp_solver.setup(
                        P=sp.triu(P, format='csc'),
                        q=p,
                        A=A,
                        l=l,
                        u=u,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        max_iter=4000,
                        warm_start=True,
                        verbose=False,
                        polish=True
                    )
                osqp_solver.warm_start(x=x_current)

            result = osqp_solver.solve()

            # Check feasibility
            if result.info.status == 'primal infeasible' or \
               result.info.status == 'primal infeasible (inaccurate)':
                if self.verbose:
                    print(f"  SQP iter {iteration}: INFEASIBLE")
                return SQPResult(
                    x=None,
                    feasible=False,
                    n_iterations=iteration + 1,
                    final_cost=np.inf,
                    constraint_violation=np.inf
                )

            if result.info.status not in ('solved', 'solved inaccurate'):
                if self.verbose:
                    print(f"  SQP iter {iteration}: status={result.info.status}")
                # Try shrinking trust region
                trust_radius *= self.trust_region_shrink
                if trust_radius < 1e-6:
                    return SQPResult(
                        x=None, feasible=False,
                        n_iterations=iteration + 1,
                        final_cost=np.inf,
                        constraint_violation=np.inf
                    )
                continue

            x_new = result.x

            # Compute change
            delta_x = np.max(np.abs(x_new - x_current))

            # Check convergence
            converged = delta_x < self.trust_region_tol

            if self.verbose:
                cost = 0.5 * x_new @ P @ x_new
                print(f"  SQP iter {iteration}: cost={cost:.6f}, "
                      f"delta_x={delta_x:.6f}, trust_r={trust_radius:.4f}")

            # Update solution
            x_current = x_new

            if converged:
                break

            # Adaptive trust region
            if delta_x < trust_radius * 0.25:
                trust_radius *= self.trust_region_expand
            elif delta_x > trust_radius * 0.75:
                trust_radius *= self.trust_region_shrink

            trust_radius = np.clip(trust_radius, 0.01, 5.0)

        # Compute final cost and constraint violation
        final_cost = 0.5 * x_current @ P @ x_current
        constraint_violation = self._compute_constraint_violation(
            A, l, u, x_current)

        return SQPResult(
            x=x_current,
            feasible=True,
            n_iterations=iteration + 1,
            final_cost=final_cost,
            constraint_violation=constraint_violation
        )

    @staticmethod
    def _compute_constraint_violation(A, l, u, x) -> float:
        """Compute maximum constraint violation."""
        Ax = A @ x
        lower_viol = np.maximum(0, l - Ax)
        upper_viol = np.maximum(0, Ax - u)
        return max(np.max(lower_viol), np.max(upper_viol))
