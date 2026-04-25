"""
GOMP Planner: Grasp-Optimized Motion Planning (Algorithm 1).

This implements the full GOMP algorithm from the paper:
1. Compute IK for initial top-down grasps
2. Initialize trajectory via spline interpolation
3. For H from initial_H down to min_H:
   a. Run SQP solver
   b. If infeasible → return previous best trajectory
   c. Warm-start shorter trajectory from current solution
   d. Enforce v_H = 0
4. Return shortest feasible trajectory

For multiple grasp candidates, run GOMP on each and select the fastest.
"""

import numpy as np
from typing import List, Optional
import time as time_module

from gomp.robot_adapter import RobotAdapter
from gomp.grasp.grasp import Grasp
from gomp.grasp.grasp_set import GraspSet
from gomp.obstacles.depth_map import DepthMapObstacle
from gomp.obstacles.collision import ObstacleConstraint
from gomp.optimization.qp_builder import QPBuilder
from gomp.optimization.sqp_solver import SQPSolver
from gomp.optimization.warm_start import spline_warm_start, interpolate_to_shorter
from gomp.optimization.fast_kinematics import FastKinematics
from gomp.planner.trajectory import Trajectory


class GOMPPlanner:
    """
    Grasp-Optimized Motion Planner.

    Computes minimum-time trajectories that avoid obstacles while
    constraining start/end configurations to grasp sets with
    rotational degrees of freedom.

    Parameters
    ----------
    robot : RobotAdapter
        Robot model.
    t_step : float
        Time step between waypoints (seconds).
        Default 0.008 matches UR5e 125Hz control loop.
    initial_H : int
        Initial time horizon (number of steps).
    min_H : int
        Minimum time horizon to try.
    H_reduction : str
        Strategy for reducing H: 'linear' (subtract 1) or
        'geometric' (multiply by factor).
    H_reduction_factor : float
        Factor for geometric reduction. E.g., 0.8 means H_new = int(H * 0.8).
    sqp_kwargs : dict
        Additional arguments for the SQP solver.
    verbose : bool
        Print progress information.
    """

    def __init__(self,
                 robot: RobotAdapter,
                 t_step: float = 0.008,
                 initial_H: int = 60,
                 min_H: int = 2,
                 H_reduction: str = 'geometric',
                 H_reduction_factor: float = 0.8,
                 sqp_kwargs: dict = None,
                 verbose: bool = True):
        self.robot = robot
        self.t_step = t_step
        self.initial_H = initial_H
        self.min_H = min_H
        self.H_reduction = H_reduction
        self.H_reduction_factor = H_reduction_factor
        self.verbose = verbose

        # SQP solver configuration
        self.sqp_kwargs = sqp_kwargs or {}
        if 'verbose' not in self.sqp_kwargs:
            self.sqp_kwargs['verbose'] = verbose
        self.sqp_solver = SQPSolver(**self.sqp_kwargs)

        # Pre-compile fast kinematics engine
        self.fast_kin = FastKinematics(robot)

    def plan(self,
             grasp_start: Grasp,
             grasp_goal: Grasp,
             obstacles: Optional[DepthMapObstacle] = None,
             ik_seed_start: Optional[np.ndarray] = None,
             ik_seed_goal: Optional[np.ndarray] = None) -> Optional[Trajectory]:
        """
        Plan a minimum-time trajectory from grasp_start to grasp_goal.

        This is the main GOMP algorithm (Algorithm 1):
        1. Compute IK for top-down grasps
        2. Spline warm start
        3. Iteratively reduce H until infeasible
        4. Return shortest feasible trajectory

        Parameters
        ----------
        grasp_start : Grasp
            Start grasp with rotational DOF.
        grasp_goal : Grasp
            Goal grasp with rotational DOF.
        obstacles : DepthMapObstacle or None
            Obstacle depth map.
        ik_seed_start : np.ndarray or None
            Known IK solution for start grasp (bypasses IK solver).
        ik_seed_goal : np.ndarray or None
            Known IK solution for goal grasp (bypasses IK solver).

        Returns
        -------
        trajectory : Trajectory or None
            Optimized trajectory, or None if no feasible solution found.
        """
        n = self.robot.n_dof
        t0 = time_module.time()

        if self.verbose:
            print(f"GOMP: Planning with t_step={self.t_step}s, "
                  f"initial_H={self.initial_H}, n_dof={n}")

        # --- Step 1: Compute initial IK solutions ---
        start_grasp_set = GraspSet(grasp_start, self.robot)
        goal_grasp_set = GraspSet(grasp_goal, self.robot)

        # Use provided IK seeds or solve IK
        if ik_seed_start is not None:
            # Pre-seed the cache so get_topdown_ik returns the known config
            start_grasp_set._ik_cache[0.0] = ik_seed_start
            q_start = ik_seed_start
        else:
            q_start = start_grasp_set.get_topdown_ik()

        if q_start is None:
            if self.verbose:
                print("GOMP: Failed to find IK for start grasp")
            return None

        if ik_seed_goal is not None:
            goal_grasp_set._ik_cache[0.0] = ik_seed_goal
            q_goal = ik_seed_goal
        else:
            q_goal = goal_grasp_set.get_topdown_ik(seed_jnt_values=q_start)

        if q_goal is None:
            if self.verbose:
                print("GOMP: Failed to find IK for goal grasp")
            return None

        if self.verbose:
            print(f"GOMP: IK solutions found")
            print(f"  q_start = {np.round(q_start, 3)}")
            print(f"  q_goal  = {np.round(q_goal, 3)}")

        # --- Step 2: Create obstacle constraint ---
        obs_constraint = None
        if obstacles is not None:
            obs_constraint = ObstacleConstraint(
                self.robot, obstacles, safety_margin=0.01,
                fast_kin=self.fast_kin
            )

        # --- Step 3: Initialize trajectory with spline ---
        H = self.initial_H
        x = spline_warm_start(q_start, q_goal, H, n, self.t_step)

        if self.verbose:
            print(f"GOMP: Spline warm start initialized (H={H})")

        # --- Step 4: Time minimization loop ---
        best_trajectory = None

        while H >= self.min_H:
            if self.verbose:
                print(f"\nGOMP: Solving for H={H} "
                      f"(duration={H * self.t_step:.3f}s)...")

            # Run SQP
            result = self.sqp_solver.solve(
                x_init=x,
                H=H,
                n=n,
                t_step=self.t_step,
                robot=self.robot,
                obstacle_constraint=obs_constraint,
                start_grasp_set=start_grasp_set,
                goal_grasp_set=goal_grasp_set,
                fast_kin=self.fast_kin
            )

            if not result.feasible:
                if self.verbose:
                    print(f"GOMP: H={H} is INFEASIBLE after "
                          f"{result.n_iterations} iterations")
                break  # Return the previous best

            # Build trajectory from solution
            qp = QPBuilder(H, n)
            waypoints = qp.extract_waypoints(result.x)
            velocities = qp.extract_velocities(result.x)

            # Velocities are already in rad/s from the SQP
            best_trajectory = Trajectory(
                waypoints=waypoints,
                velocities=velocities,
                t_step=self.t_step,
                H=H
            )

            if self.verbose:
                print(f"GOMP: H={H} FEASIBLE (cost={result.final_cost:.4f}, "
                      f"iters={result.n_iterations}, "
                      f"duration={best_trajectory.duration:.3f}s)")

            # --- Step 5: Reduce H and warm start ---
            H_prev = H
            if self.H_reduction == 'geometric':
                H = max(int(H * self.H_reduction_factor), H - 1)
            else:
                H -= 1

            if H < self.min_H:
                break

            # Warm start with interpolated previous solution
            x = interpolate_to_shorter(result.x, H_prev, H, n, self.t_step)

        elapsed = time_module.time() - t0
        if self.verbose:
            if best_trajectory is not None:
                print(f"\nGOMP: Done in {elapsed:.3f}s. "
                      f"Best trajectory: H={best_trajectory.H}, "
                      f"duration={best_trajectory.duration:.3f}s")
            else:
                print(f"\nGOMP: Failed to find feasible trajectory "
                      f"({elapsed:.3f}s)")

        return best_trajectory

    def plan_multi_grasp(self,
                         grasp_candidates: List[Grasp],
                         grasp_goal: Grasp,
                         obstacles: Optional[DepthMapObstacle] = None) -> Optional[Trajectory]:
        """
        Run GOMP for multiple candidate grasps and return the fastest trajectory.

        From Section V: "For each object, we compute trajectories to all of its
        candidate grasp points in parallel, and then select the grasp that results
        in the shortest execution time."

        Parameters
        ----------
        grasp_candidates : list of Grasp
            Multiple candidate start grasps.
        grasp_goal : Grasp
            Goal grasp.
        obstacles : DepthMapObstacle or None
            Obstacle depth map.

        Returns
        -------
        trajectory : Trajectory or None
            Fastest trajectory among all candidates.
        """
        if self.verbose:
            print(f"GOMP Multi-Grasp: {len(grasp_candidates)} candidates")

        best_trajectory = None
        best_duration = np.inf

        for i, grasp_start in enumerate(grasp_candidates):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Candidate {i+1}/{len(grasp_candidates)} "
                      f"(quality={grasp_start.quality:.3f})")
                print(f"{'='*60}")

            traj = self.plan(grasp_start, grasp_goal, obstacles)

            if traj is not None and traj.duration < best_duration:
                best_trajectory = traj
                best_duration = traj.duration
                if self.verbose:
                    print(f"  → New best! duration={best_duration:.3f}s")

        if self.verbose:
            if best_trajectory is not None:
                print(f"\nGOMP Multi-Grasp: Best duration = "
                      f"{best_trajectory.duration:.3f}s")
            else:
                print(f"\nGOMP Multi-Grasp: No feasible trajectory found")

        return best_trajectory
