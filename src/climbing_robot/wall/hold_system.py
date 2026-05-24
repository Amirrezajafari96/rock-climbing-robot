"""
Wall and hold management system.

Models the climbing wall as a 2D grid (X = horizontal, Z = vertical) with
cylindrical holds at discrete positions. Provides spatial queries,
reachability checks, and hold graph construction for planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


class HoldType(Enum):
    """Classification of hold shape/difficulty."""
    JUG = auto()        # large, easy to grip
    CRIMP = auto()      # small edge
    SLOPER = auto()     # sloped, friction-dependent
    PINCH = auto()      # pinched between thumb and fingers
    GOAL = auto()       # top-out target hold


@dataclass
class Hold:
    """
    A single climbing hold on the wall.

    Attributes:
        hold_id:   Unique string identifier, e.g. "r2_c1"
        position:  3D world position [x, y, z]. y is the wall-normal depth
                   (holds protrude slightly from the wall surface).
        row:       Integer row index (0 = bottom).
        col:       Integer column index.
        hold_type: HoldType enum.
        radius:    Grip radius [m].
        active:    Whether this hold is currently occupied by a gripper.
    """

    hold_id: str
    position: np.ndarray
    row: int
    col: int
    hold_type: HoldType = HoldType.JUG
    radius: float = 0.04
    active: bool = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])

    def distance_to(self, other: "Hold") -> float:
        """Euclidean distance to another hold."""
        return float(np.linalg.norm(self.position - other.position))

    def horizontal_distance_to(self, other: "Hold") -> float:
        """Distance in the XZ plane (ignoring wall depth)."""
        diff = self.position[[0, 2]] - other.position[[0, 2]]
        return float(np.linalg.norm(diff))

    def __repr__(self) -> str:
        return (
            f"Hold({self.hold_id!r}, pos={self.position.round(3).tolist()}, "
            f"type={self.hold_type.name})"
        )


@dataclass
class WallConfig:
    """Configuration for the climbing wall geometry."""

    width: float = 1.2         # [m] total wall width
    height: float = 2.5        # [m] total wall height
    depth: float = 0.05        # [m] wall panel thickness
    hold_depth: float = 0.03   # [m] hold protrusion from wall face
    n_rows: int = 6            # number of hold rows
    n_cols: int = 3            # number of holds per row
    row_spacing: float = 0.35  # [m] vertical spacing between rows
    base_height: float = 0.30  # [m] height of first hold row
    wall_y: float = 0.0        # [m] world Y position of wall surface


class HoldSystem:
    """
    Manages all holds on the climbing wall.

    Provides:
      - Hold lookup by ID, row/col, or nearest position
      - Reachability queries (is hold X reachable from hold Y?)
      - Support polygon computation from active grippers
      - Hold graph for planning

    Args:
        config:  WallConfig defining wall geometry.
        holds:   Optional manual hold list. If None, generates a grid.

    Example::

        system = HoldSystem(WallConfig(n_rows=6, n_cols=3))
        target = system.nearest_hold(np.array([0.1, -0.03, 1.0]))
        reachable = system.reachable_from(current_hold, max_reach=0.5)
    """

    def __init__(
        self,
        config: WallConfig | None = None,
        holds: list[Hold] | None = None,
    ) -> None:
        self._cfg = config or WallConfig()
        self._holds: dict[str, Hold] = {}

        if holds is not None:
            for h in holds:
                self._holds[h.hold_id] = h
        else:
            self._generate_grid()

        logger.info("HoldSystem initialized with %d holds", len(self._holds))

    # ── Grid generation ───────────────────────────────────────────────────

    def _generate_grid(self) -> None:
        """
        Generate a regular grid of holds matching the MJCF model layout.
        Adds slight randomness to X positions to make climbing more interesting.
        """
        cfg = self._cfg
        rng = np.random.default_rng(seed=42)

        # Column X positions
        x_positions = np.linspace(
            -cfg.width / 2 * 0.85,
             cfg.width / 2 * 0.85,
             cfg.n_cols,
        )

        for row in range(cfg.n_rows):
            z = cfg.base_height + row * cfg.row_spacing
            # Slightly offset each row for realism
            x_offset = rng.uniform(-0.05, 0.05)

            for col in range(cfg.n_cols):
                x = float(x_positions[col]) + x_offset
                y = -(cfg.wall_y + cfg.hold_depth)  # protrudes into -Y

                hold_type = HoldType.GOAL if row == cfg.n_rows - 1 else HoldType.JUG
                hid = f"r{row}_c{col}"
                self._holds[hid] = Hold(
                    hold_id=hid,
                    position=np.array([x, y, z]),
                    row=row,
                    col=col,
                    hold_type=hold_type,
                )

    # ── Lookup ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._holds)

    def __iter__(self) -> Iterator[Hold]:
        return iter(self._holds.values())

    def get(self, hold_id: str) -> Hold:
        if hold_id not in self._holds:
            raise KeyError(f"Hold '{hold_id}' not found")
        return self._holds[hold_id]

    def get_row(self, row: int) -> list[Hold]:
        return [h for h in self._holds.values() if h.row == row]

    def get_col(self, col: int) -> list[Hold]:
        return [h for h in self._holds.values() if h.col == col]

    def all_holds(self) -> list[Hold]:
        return list(self._holds.values())

    def goal_holds(self) -> list[Hold]:
        return [h for h in self._holds.values() if h.hold_type == HoldType.GOAL]

    # ── Spatial queries ───────────────────────────────────────────────────

    def nearest_hold(
        self,
        position: np.ndarray,
        *,
        exclude_ids: set[str] | None = None,
        max_distance: float = float("inf"),
    ) -> Hold | None:
        """
        Find the closest hold to a given world position.

        Args:
            position:    Query position [x, y, z].
            exclude_ids: Hold IDs to skip (e.g. already occupied).
            max_distance: Return None if nearest is farther than this.

        Returns:
            Nearest Hold, or None if none within max_distance.
        """
        pos = np.asarray(position)
        exclude = exclude_ids or set()
        best: Hold | None = None
        best_dist = max_distance

        for h in self._holds.values():
            if h.hold_id in exclude:
                continue
            d = float(np.linalg.norm(h.position - pos))
            if d < best_dist:
                best_dist = d
                best = h

        return best

    def holds_in_radius(
        self,
        center: np.ndarray,
        radius: float,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[Hold]:
        """All holds within `radius` meters of `center`."""
        pos = np.asarray(center)
        exclude = exclude_ids or set()
        return [
            h for h in self._holds.values()
            if h.hold_id not in exclude
            and float(np.linalg.norm(h.position - pos)) <= radius
        ]

    def reachable_from(
        self,
        hold: Hold,
        max_reach: float = 0.55,
        min_height_gain: float = 0.0,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[Hold]:
        """
        Returns holds reachable from a given hold.

        A hold is reachable if:
          1. Its horizontal distance is <= max_reach
          2. Its height is >= hold.z + min_height_gain (prefer upward motion)

        Args:
            hold:             Source hold.
            max_reach:        Maximum limb reach radius [m].
            min_height_gain:  Minimum upward height gain [m] (0 = same level ok).
            exclude_ids:      Hold IDs to skip.

        Returns:
            Sorted list of reachable holds, nearest first.
        """
        exclude = (exclude_ids or set()) | {hold.hold_id}
        candidates = []

        for h in self._holds.values():
            if h.hold_id in exclude:
                continue
            horiz_dist = hold.horizontal_distance_to(h)
            height_gain = h.z - hold.z
            if horiz_dist <= max_reach and height_gain >= min_height_gain:
                candidates.append((horiz_dist, h))

        candidates.sort(key=lambda x: x[0])
        return [h for _, h in candidates]

    # ── Occupancy management ──────────────────────────────────────────────

    def set_active(self, hold_id: str, active: bool = True) -> None:
        self._holds[hold_id].active = active

    def active_holds(self) -> list[Hold]:
        return [h for h in self._holds.values() if h.active]

    def free_holds(self) -> list[Hold]:
        return [h for h in self._holds.values() if not h.active]

    # ── Geometry ──────────────────────────────────────────────────────────

    def support_positions(self) -> np.ndarray:
        """
        Returns (N, 3) array of active gripper positions on the wall.
        Used by COM stability checker.
        """
        active = self.active_holds()
        if not active:
            return np.empty((0, 3))
        return np.stack([h.position for h in active])

    def __repr__(self) -> str:
        return (
            f"HoldSystem(n_holds={len(self._holds)}, "
            f"n_active={sum(1 for h in self._holds.values() if h.active)})"
        )
