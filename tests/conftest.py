"""
pytest fixtures shared across all test modules.
"""
import numpy as np
import pytest

from climbing_robot.kinematics.forward import FKSolver, DHLink, make_limb_fk
from climbing_robot.kinematics.inverse import IKSolver, IKConfig
from climbing_robot.stability.com_checker import COMStabilityChecker
from climbing_robot.wall.hold_system import HoldSystem, WallConfig


@pytest.fixture
def simple_fk() -> FKSolver:
    """A simple 2-link FK chain for testing."""
    links = [
        DHLink(a=0.0, alpha=np.pi / 2, d=0.0),
        DHLink(a=0.25, alpha=0.0, d=0.0),
        DHLink(a=0.20, alpha=0.0, d=0.0),
    ]
    return FKSolver(links)


@pytest.fixture
def limb_fk() -> FKSolver:
    """Full climbing-robot limb FK."""
    return make_limb_fk(upper_arm_length=0.25, lower_arm_length=0.20)


@pytest.fixture
def ik_solver(limb_fk) -> IKSolver:
    cfg = IKConfig(
        max_iterations=200,
        position_tolerance=1e-3,
        joint_limits_lower=[-1.57, -0.78, -2.0, -2.0],
        joint_limits_upper=[ 1.57,  0.78,  0.1,  0.1],
    )
    return IKSolver(limb_fk, config=cfg)


@pytest.fixture
def hold_system() -> HoldSystem:
    """Small 3-row, 3-col hold system for testing."""
    return HoldSystem(WallConfig(n_rows=3, n_cols=3, row_spacing=0.35, base_height=0.30))


@pytest.fixture
def stability_checker() -> COMStabilityChecker:
    return COMStabilityChecker()


@pytest.fixture
def three_contact_positions() -> np.ndarray:
    """Triangle of contact points on the wall (XZ plane)."""
    return np.array([
        [-0.30, -0.03, 0.30],
        [ 0.30, -0.03, 0.30],
        [ 0.00, -0.03, 0.65],
    ])
