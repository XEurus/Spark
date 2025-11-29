"""Spark G1 EgoVLA task suite.

This package will host manager-based RL environments using the G1 Inspire robot
with task logic inspired by the Ego Humanoid Manipulation Benchmark.
"""

from omni.isaac.lab_tasks.utils import import_packages

# blacklist subpackages if needed (e.g., utils)
_BLACKLIST_PKGS = []

# Import all configs in this package so that gym.register in submodules runs
import_packages(__name__, _BLACKLIST_PKGS)
