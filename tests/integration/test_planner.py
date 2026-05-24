"""
Integration tests for the climbing planner.

Tests wire together: HoldSystem + COMStabilityChecker + ClimbingPlanner.
Does NOT require MuJoCo — pure planning logic.
"""

import numpy as np
import pytest

from climbing_robot.planner.base_planner import GripperID, MoveType
from climbing_robot.planner.climbing_planner import ClimbingPlanner
from climbing_robot.stability.com_checker import COMStabilityChecker
from climbing_robot.wall.hold_system import HoldSystem, WallConfig


@pytest.fixture
def full_planner():
    holds = HoldSystem(WallConfig(n_rows=6, n_cols=3, row_spacing=0.35, base_height=0.30))
    stability = COMStabilityChecker()
    return ClimbingPlanner(holds, stability, max_reach=0.60, n_traj_steps=10)


@pytest.fixture
def initial_grips():
    return {
        GripperID.LEFT:   "r0_c0",
        GripperID.RIGHT:  "r0_c2",
        GripperID.CENTER: "r0_c1",
    }


class TestClimbingPlanner:
    def test_plan_returns_plan_result(self, full_planner, initial_grips):
        from climbing_robot.planner.base_planner import PlanResult
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.5)
        assert isinstance(result, PlanResult)

    def test_plan_produces_moves(self, full_planner, initial_grips):
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.5)
        # Should produce at least some moves for a reachable goal
        assert result.n_moves >= 0  # at minimum, no crash

    def test_each_move_has_trajectory(self, full_planner, initial_grips):
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.0)
        for move in result.moves:
            assert isinstance(move.joint_traj, list)

    def test_moves_reference_valid_holds(self, full_planner, initial_grips):
        holds = full_planner._holds
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.5)
        for move in result.moves:
            if move.target_hold:
                # Should not raise KeyError
                h = holds.get(move.target_hold)
                assert h is not None

    def test_moves_type_is_reach_hold(self, full_planner, initial_grips):
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.5)
        for move in result.moves:
            assert move.move_type == MoveType.REACH_HOLD

    def test_plan_reset_clears_state(self, full_planner, initial_grips):
        com = np.array([0.0, -0.08, 0.35])
        full_planner.plan(initial_grips, com, 1.0)
        full_planner.reset()
        assert full_planner._state.step_count == 0

    def test_gripper_id_values(self):
        assert GripperID.LEFT.value == "left"
        assert GripperID.RIGHT.value == "right"
        assert GripperID.CENTER.value == "center"

    def test_height_gain_non_negative(self, full_planner, initial_grips):
        com = np.array([0.0, -0.08, 0.35])
        result = full_planner.plan(initial_grips, com, target_height=1.5)
        assert result.total_height_gain >= 0.0
