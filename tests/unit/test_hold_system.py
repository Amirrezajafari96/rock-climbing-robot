"""
Unit tests for the HoldSystem and Hold classes.

Tests cover:
  - Grid generation
  - Hold lookup (by ID, row, col)
  - Nearest hold query
  - Reachability query
  - Occupancy management
"""

import numpy as np
import pytest

from climbing_robot.wall.hold_system import Hold, HoldSystem, HoldType, WallConfig


@pytest.fixture
def small_system():
    return HoldSystem(WallConfig(n_rows=4, n_cols=3, row_spacing=0.35, base_height=0.30))


class TestHold:
    def test_hold_position_is_np_array(self):
        h = Hold("test", [0.1, -0.03, 0.5], row=1, col=0)
        assert isinstance(h.position, np.ndarray)

    def test_hold_xz_properties(self):
        h = Hold("test", [0.2, -0.03, 0.7], row=2, col=1)
        assert h.x == pytest.approx(0.2)
        assert h.z == pytest.approx(0.7)

    def test_hold_distance_to_self_is_zero(self):
        h = Hold("a", [0.0, 0.0, 0.0], row=0, col=0)
        assert h.distance_to(h) == pytest.approx(0.0)

    def test_hold_distance_between_known_points(self):
        a = Hold("a", [0.0, 0.0, 0.0], row=0, col=0)
        b = Hold("b", [0.3, 0.0, 0.4], row=1, col=1)
        assert a.distance_to(b) == pytest.approx(0.5, rel=1e-5)


class TestHoldSystem:
    def test_grid_generates_correct_count(self, small_system):
        assert len(small_system) == 12  # 4 rows * 3 cols

    def test_hold_ids_follow_naming_convention(self, small_system):
        h = small_system.get("r0_c0")
        assert h.hold_id == "r0_c0"
        assert h.row == 0
        assert h.col == 0

    def test_get_row_returns_correct_count(self, small_system):
        row = small_system.get_row(0)
        assert len(row) == 3

    def test_get_col_returns_correct_count(self, small_system):
        col = small_system.get_col(1)
        assert len(col) == 4

    def test_goal_holds_in_last_row(self, small_system):
        goals = small_system.goal_holds()
        assert len(goals) == 3
        assert all(h.row == 3 for h in goals)
        assert all(h.hold_type == HoldType.GOAL for h in goals)

    def test_get_nonexistent_hold_raises(self, small_system):
        with pytest.raises(KeyError):
            small_system.get("r99_c99")

    def test_nearest_hold_returns_closest(self, small_system):
        # Query near r0_c1 (center bottom)
        h = small_system.get("r0_c1")
        nearest = small_system.nearest_hold(h.position + np.array([0.01, 0, 0]))
        assert nearest is not None
        assert nearest.hold_id == "r0_c1"

    def test_nearest_hold_excludes_ids(self, small_system):
        h = small_system.get("r0_c1")
        nearest = small_system.nearest_hold(
            h.position, exclude_ids={"r0_c1"}
        )
        assert nearest is None or nearest.hold_id != "r0_c1"

    def test_holds_in_radius(self, small_system):
        center = np.array([0.0, -0.03, 0.30])
        nearby = small_system.holds_in_radius(center, radius=0.4)
        assert len(nearby) >= 1

    def test_reachable_from_finds_higher_holds(self, small_system):
        h = small_system.get("r0_c1")
        reachable = small_system.reachable_from(h, max_reach=0.6, min_height_gain=0.1)
        assert all(r.z > h.z for r in reachable)

    def test_occupancy_management(self, small_system):
        small_system.set_active("r0_c0", True)
        assert small_system.get("r0_c0").active
        active = small_system.active_holds()
        assert any(h.hold_id == "r0_c0" for h in active)

        small_system.set_active("r0_c0", False)
        assert not small_system.get("r0_c0").active

    def test_support_positions_only_active(self, small_system):
        small_system.set_active("r0_c0", True)
        small_system.set_active("r1_c1", True)
        positions = small_system.support_positions()
        assert positions.shape == (2, 3)

    def test_all_holds_iteration(self, small_system):
        all_holds = small_system.all_holds()
        assert len(all_holds) == 12
        assert all(isinstance(h, Hold) for h in all_holds)
