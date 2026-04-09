"""
GOMP: Grasp-Optimized Motion Planning for Bin Picking

Implementation of the GOMP algorithm from:
  Ichnowski et al., "GOMP: Grasp-Optimized Motion Planning for Bin Picking", 2020.

Uses WRS (wanweiwei07/wrs) as the robotics backend for FK/IK/visualization.
"""

import sys
import os

# Add WRS to sys.path so that `import wrs` works
_wrs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wrs')
if os.path.isdir(_wrs_path) and _wrs_path not in sys.path:
    sys.path.insert(0, _wrs_path)

__version__ = "0.1.0"
