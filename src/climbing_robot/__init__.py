"""
Climbing Robot — core package.

Modules:
    robot       — RobotModel abstraction and configuration
    kinematics  — FK, IK, SE3 transforms
    wall        — Wall geometry, hold system, reachability
    stability   — COM computation and support polygon
    planner     — Autonomous climbing planner
    simulators  — Physics backend (MuJoCo + abstract interface)
    visualization — Debug overlays and trajectory plots
    utils       — Logging, config loading, math helpers
"""

__version__ = "0.1.0"
__author__ = "Rock Climbing Robot Team"
