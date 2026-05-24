"""
Center-of-Mass (COM) stability checker.

Computes the robot's COM from body masses, projects it onto the wall plane,
and checks whether it falls within the convex support polygon formed by
active gripper contacts.

Theory:
  For a wall-climbing robot, stability requires the projected COM to lie
  within the convex hull of the contact points. We compute a signed
  stability margin (negative = unstable).

Reference:
  Bretl, T. (2006). "Motion Planning of Multi-Limbed Robots Subject to
  Equilibrium Constraints." International Journal of Robotics Research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """
    Result of a stability check.

    Attributes:
        is_stable:       True if COM projection is inside support polygon.
        com_position:    World-frame 3D COM position.
        com_projected:   COM projected onto the wall plane (X, Z).
        support_polygon: (N, 2) array of support polygon vertices in (X, Z).
        margin:          Signed distance from COM to polygon boundary [m].
                         Positive = inside (stable), Negative = outside.
        n_contacts:      Number of active contact points.
    """

    is_stable: bool
    com_position: np.ndarray
    com_projected: np.ndarray
    support_polygon: np.ndarray
    margin: float
    n_contacts: int


class COMStabilityChecker:
    """
    Checks static stability of the climbing robot on a vertical wall.

    The wall is assumed to lie in the XZ plane (Y = wall normal).
    We project the COM onto this plane and check containment within
    the convex hull of active gripper contact points.

    Args:
        gravity_direction: Unit vector of gravity (default [0,0,-1]).
        min_contacts:      Minimum grippers needed for any stability (default 2).

    Example::

        checker = COMStabilityChecker()
        gripper_positions = np.array([
            [-0.3, -0.03, 0.30],
            [ 0.3, -0.03, 0.30],
            [ 0.0, -0.03, 0.65],
        ])
        com = np.array([0.0, -0.08, 0.45])
        result = checker.check(com, gripper_positions)
        print(f"Stable: {result.is_stable}, margin: {result.margin:.3f}m")
    """

    def __init__(
        self,
        gravity_direction: np.ndarray | None = None,
        min_contacts: int = 2,
    ) -> None:
        self._g = np.array(gravity_direction or [0.0, 0.0, -1.0])
        self._g /= np.linalg.norm(self._g)
        self._min_contacts = min_contacts

    def check(
        self,
        com_position: np.ndarray,
        contact_positions: np.ndarray,
    ) -> StabilityResult:
        """
        Check stability given COM and active gripper contact points.

        Args:
            com_position:     3D world COM position [x, y, z].
            contact_positions: (N, 3) array of contact point positions.

        Returns:
            StabilityResult with stability flag and margin.
        """
        com = np.asarray(com_position, dtype=float)
        contacts = np.asarray(contact_positions, dtype=float)
        n_contacts = len(contacts)

        if n_contacts < self._min_contacts:
            logger.warning(
                "Only %d contact(s), need at least %d for stability",
                n_contacts, self._min_contacts,
            )
            return StabilityResult(
                is_stable=False,
                com_position=com,
                com_projected=com[[0, 2]],
                support_polygon=np.empty((0, 2)),
                margin=-float("inf"),
                n_contacts=n_contacts,
            )

        # Project all points onto the wall plane (X, Z)
        # For a vertical wall in the XZ plane, we ignore Y
        com_xz = com[[0, 2]]
        contacts_xz = contacts[:, [0, 2]]

        # Build support polygon
        if n_contacts < 3:
            # Degenerate case: line segment support
            is_stable, margin = self._check_on_segment(com_xz, contacts_xz)
            polygon = contacts_xz
        else:
            is_stable, margin, polygon = self._check_in_convex_hull(
                com_xz, contacts_xz
            )

        return StabilityResult(
            is_stable=is_stable,
            com_position=com.copy(),
            com_projected=com_xz.copy(),
            support_polygon=polygon,
            margin=margin,
            n_contacts=n_contacts,
        )

    def check_from_simulator(
        self,
        simulator: "BaseSimulator",  # type: ignore[name-defined]
        body_names: list[str],
        body_masses: list[float],
        contact_site_names: list[str],
    ) -> StabilityResult:
        """
        Convenience method: compute COM from simulator body states.

        Args:
            simulator:          Active simulator instance.
            body_names:         List of body names to include in COM.
            body_masses:        Corresponding masses [kg].
            contact_site_names: Site names for active gripper contacts.

        Returns:
            StabilityResult.
        """
        total_mass = sum(body_masses)
        com = np.zeros(3)
        for name, mass in zip(body_names, body_masses):
            pos = simulator.get_body_position(name)
            com += mass * pos
        com /= total_mass

        contact_positions = np.array([
            simulator.get_site_position(sname)
            for sname in contact_site_names
        ])

        return self.check(com, contact_positions)

    # ── Private geometry helpers ──────────────────────────────────────────

    @staticmethod
    def _check_on_segment(
        point: np.ndarray, endpoints: np.ndarray
    ) -> tuple[bool, float]:
        """
        Check if a point is on or between two endpoints (degenerate polygon).
        Returns (is_on, signed_distance_to_midpoint).
        """
        a, b = endpoints[0], endpoints[-1]
        ab = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-9:
            dist = float(np.linalg.norm(point - a))
            return dist < 1e-3, -dist
        t = np.dot(point - a, ab) / (ab_len ** 2)
        t_clamped = np.clip(t, 0, 1)
        closest = a + t_clamped * ab
        dist = float(np.linalg.norm(point - closest))
        # "margin" = how far inside the midpoint the COM is
        mid = (a + b) / 2
        half_len = ab_len / 2
        margin = half_len - float(np.linalg.norm(point - mid))
        return dist < 1e-3, margin

    @staticmethod
    def _check_in_convex_hull(
        point: np.ndarray, pts: np.ndarray
    ) -> tuple[bool, float, np.ndarray]:
        """
        Check if `point` is inside the convex hull of `pts`.

        Returns (inside, signed_margin, hull_vertices).

        Signed margin:
          - Positive = distance to nearest edge from inside
          - Negative = distance to nearest edge from outside
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Collinear points, fall back to segment check
            is_on, margin = COMStabilityChecker._check_on_segment(point, pts)
            return is_on, margin, pts

        # Check containment: point is inside if it satisfies all halfspace eqs
        # Hull equations: A @ x + b <= 0 for interior points
        eqs = hull.equations          # (n_facets, 3) for 2D → (n_facets, 3)
        dists = eqs[:, :2] @ point + eqs[:, 2]
        # Positive dist → outside that halfspace
        margin = float(-np.max(dists))   # negative of max violation
        inside = bool(np.all(dists <= 1e-9))

        hull_verts = pts[hull.vertices]
        return inside, margin, hull_verts

    def compute_com(
        self,
        positions: list[np.ndarray],
        masses: list[float],
    ) -> np.ndarray:
        """
        Compute the weighted COM from a list of body positions and masses.

        Args:
            positions: List of 3D body positions.
            masses:    Corresponding masses [kg].

        Returns:
            COM position [x, y, z].
        """
        total = sum(masses)
        if total <= 0:
            raise ValueError("Total mass must be positive")
        com = np.zeros(3)
        for pos, mass in zip(positions, masses):
            com += mass * np.asarray(pos)
        return com / total
