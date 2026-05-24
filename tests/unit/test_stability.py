"""
Unit tests for COM stability checker.

Tests cover:
  - Stable configuration (COM inside support polygon)
  - Unstable configuration (COM outside polygon)
  - Degenerate cases (2 contacts, 1 contact, 0 contacts)
  - COM computation from body masses
  - Support polygon shape
"""

import numpy as np
import pytest

from climbing_robot.stability.com_checker import COMStabilityChecker, StabilityResult


class TestCOMStabilityChecker:
    def test_com_inside_triangle_is_stable(
        self, stability_checker, three_contact_positions
    ):
        """COM at centroid of three contacts should be stable."""
        centroid = three_contact_positions[:, [0, 2]].mean(axis=0)
        com = np.array([centroid[0], -0.08, centroid[1]])
        result = stability_checker.check(com, three_contact_positions)
        assert result.is_stable
        assert result.margin > 0

    def test_com_outside_polygon_is_unstable(
        self, stability_checker, three_contact_positions
    ):
        """COM far to the side should be unstable."""
        com = np.array([2.0, -0.08, 0.40])  # way outside
        result = stability_checker.check(com, three_contact_positions)
        assert not result.is_stable
        assert result.margin < 0

    def test_two_contacts_degenerate(self, stability_checker):
        """With 2 contacts, stability degenerates to segment containment."""
        contacts = np.array([
            [-0.3, -0.03, 0.30],
            [ 0.3, -0.03, 0.30],
        ])
        com_on = np.array([0.0, -0.05, 0.30])
        result = stability_checker.check(com_on, contacts)
        assert isinstance(result, StabilityResult)
        assert result.n_contacts == 2

    def test_zero_contacts_is_unstable(self, stability_checker):
        contacts = np.empty((0, 3))
        com = np.array([0.0, -0.08, 0.40])
        result = stability_checker.check(com, contacts)
        assert not result.is_stable

    def test_result_has_correct_n_contacts(
        self, stability_checker, three_contact_positions
    ):
        com = np.array([0.0, -0.08, 0.42])
        result = stability_checker.check(com, three_contact_positions)
        assert result.n_contacts == 3

    def test_com_position_preserved_in_result(
        self, stability_checker, three_contact_positions
    ):
        com = np.array([0.05, -0.08, 0.42])
        result = stability_checker.check(com, three_contact_positions)
        np.testing.assert_array_equal(result.com_position, com)

    def test_compute_com_weighted_average(self, stability_checker):
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ]
        masses = [1.0, 1.0]
        com = stability_checker.compute_com(positions, masses)
        np.testing.assert_allclose(com, [0.5, 0.0, 0.0])

    def test_compute_com_single_heavy_body(self, stability_checker):
        positions = [np.array([2.0, 1.0, 3.0])]
        masses = [5.0]
        com = stability_checker.compute_com(positions, masses)
        np.testing.assert_allclose(com, [2.0, 1.0, 3.0])

    def test_compute_com_zero_mass_raises(self, stability_checker):
        with pytest.raises(ValueError):
            stability_checker.compute_com([np.zeros(3)], [0.0])

    def test_margin_positive_when_stable(
        self, stability_checker, three_contact_positions
    ):
        # COM exactly at centroid — deeply inside
        centroid_xz = three_contact_positions[:, [0, 2]].mean(axis=0)
        com = np.array([centroid_xz[0], -0.08, centroid_xz[1]])
        result = stability_checker.check(com, three_contact_positions)
        assert result.margin > 0

    def test_support_polygon_returned(
        self, stability_checker, three_contact_positions
    ):
        com = np.array([0.0, -0.08, 0.42])
        result = stability_checker.check(com, three_contact_positions)
        assert result.support_polygon.ndim == 2
        assert result.support_polygon.shape[1] == 2

    @pytest.mark.parametrize("n_contacts", [4, 5, 6])
    def test_stability_with_many_contacts(self, stability_checker, n_contacts):
        """Large support polygon should be stable for central COM."""
        angles = np.linspace(0, 2 * np.pi, n_contacts, endpoint=False)
        radius = 0.3
        contacts = np.column_stack([
            radius * np.cos(angles),
            np.full(n_contacts, -0.03),
            radius * np.sin(angles) + 0.5,
        ])
        com = np.array([0.0, -0.05, 0.5])
        result = stability_checker.check(com, contacts)
        assert result.is_stable
