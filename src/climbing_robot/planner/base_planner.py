"""
Abstract base class for climbing planners.

Any planning algorithm (A*, RRT, RL policy) must implement BasePlanner.
This keeps the simulation loop and robot controller planner-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class GripperID(Enum):
    """Identifies which limb/gripper is acting."""
    LEFT   = "left"
    RIGHT  = "right"
    CENTER = "center"


class MoveType(Enum):
    """Type of planned motion primitive."""
    REACH_HOLD   = auto()   # move free limb to a new hold
    RELEASE_HOLD = auto()   # release current hold
    ADJUST_TORSO = auto()   # reposition body without changing grips
    STABILIZE    = auto()   # small corrective motion for COM stability


@dataclass
class ClimbMove:
    """
    One atomic motion primitive produced by the planner.

    Attributes:
        gripper:      Which limb executes this move.
        move_type:    What kind of motion.
        target_hold:  Hold ID to reach (for REACH_HOLD).
        joint_traj:   Sequence of joint configurations [n_steps, n_joints].
        estimated_duration: Approximate time [s].
        stability_margin:   Predicted COM margin after this move [m].
    """

    gripper: GripperID
    move_type: MoveType
    target_hold: str | None = None
    joint_traj: list[np.ndarray] = field(default_factory=list)
    estimated_duration: float = 1.0
    stability_margin: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    """
    Result of a planning call.

    Attributes:
        success:     Whether a valid plan was found.
        moves:       Ordered list of ClimbMove primitives.
        total_height_gain: Expected height gained [m].
        n_moves:     Number of moves in the plan.
        message:     Human-readable status.
    """

    success: bool
    moves: list[ClimbMove]
    total_height_gain: float = 0.0
    n_moves: int = 0
    message: str = ""

    def __post_init__(self) -> None:
        self.n_moves = len(self.moves)


class BasePlanner(abc.ABC):
    """
    Abstract climbing planner.

    Subclasses: ClimbingPlanner (A*), RLPlanner (future).

    The planner receives the current robot state (which holds are gripped,
    COM position) and returns a sequence of motion primitives to advance
    the robot upward.
    """

    @abc.abstractmethod
    def plan(
        self,
        active_grips: dict[GripperID, str | None],
        com_position: np.ndarray,
        target_height: float,
    ) -> PlanResult:
        """
        Compute a motion plan.

        Args:
            active_grips:   Maps each gripper to its current hold ID (or None).
            com_position:   Current COM world position [x, y, z].
            target_height:  Desired final Z height [m].

        Returns:
            PlanResult with ordered moves.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset planner internal state (call before each new climb)."""
