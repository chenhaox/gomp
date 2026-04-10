# GOMP: Grasp-Optimized Motion Planning for Bin Picking

> **⚠️ Unofficial Implementation** — This is an unofficial Python implementation of the GOMP algorithm.
> For the original work, see the [official project page](https://berkeleyautomation.github.io/GOMP/).

[![Paper](https://img.shields.io/badge/arXiv-2003.02401-b31b1b.svg)](https://arxiv.org/abs/2003.02401)

## Overview

This repository implements the **GOMP (Grasp-Optimized Motion Planning)** algorithm described in:

> J. Ichnowski, M. Danielczuk, J. Xu, V. Satish, and K. Goldberg,
> **"GOMP: Grasp-Optimized Motion Planning for Bin Picking"**,
> *IEEE International Conference on Robotics and Automation (ICRA)*, 2020.
> [[Paper]](https://arxiv.org/abs/2003.02401) [[Official Page]](https://berkeleyautomation.github.io/GOMP/)

GOMP speeds up bin-picking robot operations by incorporating robot dynamics and a set of candidate grasps into an optimizing motion planner. It uses **Sequential Quadratic Programming (SQP)** with **OSQP** as the QP solver to compute minimum-time trajectories that:

- Avoid obstacles (via a depth-map model)
- Respect joint position, velocity, and acceleration limits
- Allow rotational degrees of freedom at pick and place poses
- Minimize trajectory execution time by iteratively reducing the time horizon

## Features

- **SQP-based trajectory optimization** using OSQP with warm starting
- **Grasp rotational DOF** — optimizer selects the best grasp angle from a continuous set
- **Depth-map obstacle model** — simplified collision avoidance for bin-picking
- **Time minimization** — iteratively shortens trajectory until infeasible
- **Multi-grasp selection** — evaluates multiple grasp candidates and picks the fastest
- **WRS robotics backend** — uses [WRS](https://github.com/wanweiwei07/wrs) for FK, IK, Jacobian, and 3D visualization
- **3D animated visualization** — real-time trajectory playback and optimization process visualization in Panda3D

## Project Structure

```
gomp/
├── gomp/                           # Core library
│   ├── __init__.py
│   ├── robot_adapter.py            # WRS manipulator wrapper (FK/IK/Jacobian)
│   ├── grasp/
│   │   ├── grasp.py                # Grasp representation with rotational DOF
│   │   ├── grasp_set.py            # IK-resolved grasp sets & constraint matrices
│   │   └── grasp_sampler.py        # Synthetic grasp generation
│   ├── obstacles/
│   │   ├── depth_map.py            # 2D depth-map obstacle model
│   │   └── collision.py            # Linearized obstacle avoidance constraints
│   ├── optimization/
│   │   ├── qp_builder.py           # QP objective matrix P and variable layout
│   │   ├── constraints.py          # All constraint types (dynamics, limits, etc.)
│   │   ├── sqp_solver.py           # SQP solver with trust regions
│   │   └── warm_start.py           # Spline warm start & trajectory interpolation
│   ├── planner/
│   │   ├── gomp_planner.py         # Main GOMP algorithm (Algorithm 1)
│   │   └── trajectory.py           # Trajectory data structure
│   └── visualization/
│       ├── joint_plots.py          # Joint velocity/acceleration plots (Fig. 5)
│       └── wrs_viz.py              # 3D trajectory visualization via Panda3D
├── examples/
│   ├── basic_pick_place.py         # Basic pick-and-place demo
│   ├── multi_grasp_selection.py    # Multi-grasp candidate evaluation
│   ├── time_optimization_demo.py   # H-reduction visualization (matplotlib)
│   ├── visualize_trajectory_wrs.py # 3D animated trajectory playback (Panda3D)
│   └── visualize_optimization_wrs.py # H-reduction process in 3D (Panda3D)
├── wrs/                            # WRS robotics framework (git submodule)
├── pyproject.toml
├── IMPLEMENTATION.md               # Detailed implementation status vs. paper
└── README.md
```

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.22
- scipy ≥ 1.10
- osqp ≥ 0.6
- matplotlib ≥ 3.7
- panda3d ≥ 1.10.7 (for 3D visualization)
- [WRS](https://github.com/wanweiwei07/wrs) (included as git submodule)

## Installation

```bash
git clone --recursive https://github.com/chenhaox/gomp.git
cd gomp
pip install -e .
```

## Quick Start

### Basic Pick-and-Place
```bash
python examples/basic_pick_place.py
```

### 3D Animated Trajectory Visualization
```bash
python examples/visualize_trajectory_wrs.py
```
This opens a Panda3D window showing:
- Semi-transparent ghost poses along the trajectory
- End-effector path as a gradient-colored curve
- Real-time animated robot playback (loops)
- Pick/place grasp set visualization (rotational DOF fan)

### Optimization Process Visualization
```bash
python examples/visualize_optimization_wrs.py
```
Shows how the trajectory evolves as the time horizon H shrinks, with all EE paths overlaid and sequential animated playback of each stage.

## Examples

| Example | Description |
|---|---|
| `basic_pick_place.py` | Simple pick-and-place with GOMP planner |
| `multi_grasp_selection.py` | Evaluates multiple grasp candidates, selects fastest |
| `time_optimization_demo.py` | Plots joint velocity/acceleration profiles as H decreases (matplotlib) |
| `visualize_trajectory_wrs.py` | 3D animated trajectory playback with ghost poses and grasp sets (Panda3D) |
| `visualize_optimization_wrs.py` | 3D visualization of the H-reduction process (Panda3D) |

## Implementation Status

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for a detailed mapping between the paper and this codebase.

**Summary:**

| Component | Paper Section | Status |
|---|---|---|
| QP objective (smooth trajectories) | IV-A | ✅ |
| Dynamics & mechanical limit constraints | IV-B | ✅ |
| Depth-map obstacle model | IV-C | ✅ (EE only) |
| SQP solver with trust regions | IV | ✅ |
| Grasp rotational DOF constraints | IV-D | ✅ |
| Start/goal grasp ε tolerance | IV-D | ✅ |
| Time minimization (H reduction) | IV-E | ✅ |
| Warm starting | IV-F | ✅ |
| Multi-grasp selection | V | ✅ |
| 3D trajectory visualization | — | ✅ |
| Multi-link obstacle checking | IV-C | ❌ |
| Dex-Net integration | V | ❌ |

### Known Limitations

- **EE-only obstacle checking:** The depth-map obstacle constraint only checks the end-effector z-position. Other links (forearm, elbow) are not checked. This is sufficient for shallow bins but may allow collisions in deep-bin scenarios.
- **Orientation linearization:** The grasp constraint uses a log-map (axis-angle) orientation error for the SQP linearization. This is the standard approach (same as TrajOpt) and is accurate when the trust region keeps step sizes small. A robust near-π fallback handles the `sin(angle)→0` singularity.
- **Warm-start dynamics mismatch:** The spline warm-start forces zero velocity at endpoints, which technically violates the dynamics constraint at `q₁`. The SQP corrects this immediately.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ichnowski2020gomp,
  title={GOMP: Grasp-Optimized Motion Planning for Bin Picking},
  author={Ichnowski, Jeffrey and Danielczuk, Michael and Xu, Jingyi and Satish, Vishal and Goldberg, Ken},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020},
  organization={IEEE}
}
```

## Acknowledgments

- **GOMP** by the [UC Berkeley AUTOLAB](https://autolab.berkeley.edu/)
- **WRS** robotics framework by [wanweiwei07](https://github.com/wanweiwei07/wrs)
- **OSQP** solver by [Stellato et al.](https://osqp.org/)
