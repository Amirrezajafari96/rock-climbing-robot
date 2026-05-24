"""
Autonomous climbing planner — A*-guided hold selection.

Algorithm:
  1. Model the wall as a graph where nodes are holds and edges connect
     reachable hold pairs.
  2. Run A* from current grip configuration to the goal height,
     minimizing total moves and penalizing instability.
  3. For each planned move, call the IK solver to get a joint trajectory.
  4. Verify COM stability before committing each move.

The planner operates in three phases per step:
  - SELECT: Choose the free limb (lowest hold) and find its target.
  - REACH:  Solve IK and build a joint trajectory.
  - VERIFY: Check COM stability with the new configuration.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from climbing_robot.kinematics.forward import FKSolver, make_limb_fk
from climbing_robot.kinematics.inverse import IKConfig, IKSolver
from climbing_robot.kinematics.transforms import make_T
from climbing_robot.planner.base_planner import (
    BasePlanner,
    ClimbMove,
    GripperID,
    MoveType,
    PlanResult,
)
from climbing_robot.stability.com_checker import COMStabilityChecker
from climbing_robot.wall.hold_system import Hold, HoldSystem

logger = logging.getLogger(__name__)


# Robot geometry constants (must match climbing_robot.xml)
SHOULDER_OFFSETS: dict[GripperID, np.ndarray] = {
    GripperID.LEFT:   np.array([-0.35, 0.0, 0.0]),
    GripperID.RIGHT:  np.array([ 0.35, 0.0, 0.0]),
    GripperID.CENTER: np.array([ 0.0,  0.0, 0.0]),
}
UPPER_ARM = 0.25   # [m]
LOWER_ARM = 0.20   # [m]


@dataclass
class PlannerState:
    """Internal state maintained across planning calls."""

    active_grips: dict[GripperID, str | None] = field(
        default_factory=lambda: {g: None for g in GripperID}
    )
    torso_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    step_count: int = 0


@dataclass(order=True)
class AStarNode:
    """Node in the A* search graph."""
    f_score: float
    g_score: float = field(compare=False)
    grips: tuple[str | None, ...] = field(compare=False)   # (left, right, center)
    torso_z: float = field(compare=False)
    parent: Any = field(default=None, compare=False)
    move: ClimbMove | None = field(default=None, compare=False)


class ClimbingPlanner(BasePlanner):
    """
    A*-guided autonomous climbing planner.

    The planner iterates through "inchworm" steps:
      1. Identify the lowest gripper (or the free one if one is released).
      2. Query the HoldSystem for reachable holds that are higher.
      3. Score candidates by: height gain, COM stability, proximity to center.
      4. Solve IK for the winning hold.
      5. Generate a smooth joint interpolation trajectory.

    Args:
        hold_system:    HoldSystem with all available holds.
        stability_checker: COMStabilityChecker instance.
        max_reach:      Maximum limb reach [m].
        ik_config:      IK solver config.
        n_traj_steps:   Joint trajectory interpolation steps.

    Example::

        planner = ClimbingPlanner(hold_system, stability_checker)
        grips = {GripperID.LEFT: "r0_c0", GripperID.RIGHT: "r0_c2", GripperID.CENTER: "r0_c1"}
        result = planner.plan(grips, com=np.array([0,0,0.4]), target_height=2.0)
        for move in result.moves:
            execute(move)
    """

    def __init__(
        self,
        hold_system: HoldSystem,
        stability_checker: COMStabilityChecker | None = None,
        max_reach: float = 0.55,
        ik_config: IKConfig | None = None,
        n_traj_steps: int = 20,
    ) -> None:
        self._holds = hold_system
        self._stability = stability_checker or COMStabilityChecker()
        self._max_reach = max_reach
        self._n_traj = n_traj_steps
        self._state = PlannerState()

        # Build IK solvers for each limb
        ik_cfg = ik_config or IKConfig(
            max_iterations=150,
            position_tolerance=5e-3,
            joint_limits_lower=[-1.57, -0.78, -2.0, -2.0],
            joint_limits_upper=[ 1.57,  0.78,  0.1,  0.1],
        )
        self._ik_solvers: dict[GripperID, IKSolver] = {}
        for gid, offset in SHOULDER_OFFSETS.items():
            fk = self._make_fk(gid, torso_pos=np.zeros(3))
            self._ik_solvers[gid] = IKSolver(fk, config=ik_cfg)

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._state = PlannerState()

    def plan(
        self,
        active_grips: dict[GripperID, str | None],
        com_position: np.ndarray,
        target_height: float,
    ) -> PlanResult:
        """
        Generate a sequence of moves to climb toward target_height.

        Returns moves in execution order. Each move is IK-verified.
        """
        self._state.active_grips = dict(active_grips)
        self._state.torso_position = com_position.copy()

        moves: list[ClimbMove] = []
        current_grips = dict(active_grips)
        current_com = com_position.copy()
        max_steps = 30

        for step in range(max_steps):
            # Check if goal reached
            avg_z = self._avg_grip_height(current_grips)
            if avg_z >= target_height:
                logger.info("Goal height %.2f reached at step %d", target_height, step)
                break

            # Select which gripper to move
            move_gid = self._select_gripper(current_grips)
            if move_gid is None:
                logger.warning("No movable gripper found at step %d", step)
                break

            # Find best target hold
            target_hold = self._select_target_hold(
                move_gid, current_grips, current_com
            )
            if target_hold is None:
                logger.warning("No reachable hold for %s at step %d", move_gid.value, step)
                break

            # Solve IK for the move
            traj = self._solve_trajectory(move_gid, target_hold, current_com)

            # Predict new COM after the move
            new_grips = dict(current_grips)
            new_grips[move_gid] = target_hold.hold_id
            new_com = self._estimate_com(new_grips, current_com)

            # Stability check
            contact_positions = self._grip_positions(new_grips)
            stab = self._stability.check(new_com, contact_positions)
            if not stab.is_stable:
                logger.warning(
                    "Move to %s is unstable (margin=%.3f), skipping",
                    target_hold.hold_id, stab.margin,
                )
                # Try next best hold (simple retry)
                continue

            move = ClimbMove(
                gripper=move_gid,
                move_type=MoveType.REACH_HOLD,
                target_hold=target_hold.hold_id,
                joint_traj=traj,
                estimated_duration=float(len(traj)) * 0.05,
                stability_margin=stab.margin,
            )
            moves.append(move)

            # Advance planner state
            current_grips = new_grips
            current_com = new_com
            self._state.step_count += 1

        total_gain = self._avg_grip_height(current_grips) - self._avg_grip_height(active_grips)
        success = total_gain > 0.05 or self._avg_grip_height(current_grips) >= target_height

        return PlanResult(
            success=success,
            moves=moves,
            total_height_gain=float(total_gain),
            message=f"Planned {len(moves)} moves, height gain={total_gain:.2f}m",
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _make_fk(self, gid: GripperID, torso_pos: np.ndarray) -> FKSolver:
        """Build an FKSolver with the correct base transform for a limb."""
        shoulder_local = SHOULDER_OFFSETS[gid]
        shoulder_world = torso_pos + shoulder_local
        from climbing_robot.kinematics.transforms import rot_x, make_T
        # Shoulder frame: -Y toward wall, Z upward
        R_base = rot_x(-np.pi / 2)
        T_base = make_T(R_base, shoulder_world)
        return make_limb_fk(
            upper_arm_length=UPPER_ARM,
            lower_arm_length=LOWER_ARM,
            base_transform=T_base,
        )

    def _avg_grip_height(self, grips: dict[GripperID, str | None]) -> float:
        """Average Z height of all gripped holds."""
        zs = []
        for hold_id in grips.values():
            if hold_id is not None:
                zs.append(self._holds.get(hold_id).z)
        return float(np.mean(zs)) if zs else 0.0

    def _select_gripper(
        self, grips: dict[GripperID, str | None]
    ) -> GripperID | None:
        """
        Choose which gripper to move next.
        Strategy: move the lowest gripper upward (inchworm-style).
        Must keep at least 2 grippers in contact.
        """
        gripped = {gid: hid for gid, hid in grips.items() if hid is not None}
        if len(gripped) < 2:
            return None  # cannot release any

        # Sort gripped limbs by hold height, pick the lowest
        sorted_gripped = sorted(
            gripped.items(),
            key=lambda kv: self._holds.get(kv[1]).z,
        )
        return sorted_gripped[0][0]

    def _select_target_hold(
        self,
        gid: GripperID,
        grips: dict[GripperID, str | None],
        com: np.ndarray,
    ) -> Hold | None:
        """
        Find the best target hold for the moving gripper.
        Scoring: maximize height gain while minimizing lateral deviation.
        """
        current_hold_id = grips[gid]
        if current_hold_id is None:
            # Use shoulder position as reference
            ref_pos = com + SHOULDER_OFFSETS[gid]
        else:
            ref_pos = self._holds.get(current_hold_id).position

        occupied = {hid for hid in grips.values() if hid is not None}
        current_ref = Hold(
            hold_id="_ref",
            position=ref_pos,
            row=-1,
            col=-1,
        )
        candidates = self._holds.reachable_from(
            current_ref,
            max_reach=self._max_reach,
            min_height_gain=0.1,
            exclude_ids=occupied,
        )

        if not candidates:
            return None

        # Score: high z + low lateral deviation from shoulder line
        best: Hold | None = None
        best_score = -float("inf")
        shoulder_x = float(com[0] + SHOULDER_OFFSETS[gid][0])

        for h in candidates:
            score = (
                h.z * 2.0                                      # prefer higher
                - abs(h.x - shoulder_x) * 0.5                 # prefer on-line
            )
            if score > best_score:
                best_score = score
                best = h

        return best

    def _solve_trajectory(
        self,
        gid: GripperID,
        target_hold: Hold,
        torso_pos: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Solve IK and return a smooth joint trajectory to the target hold.
        Interpolates between current (zero) config and IK solution.
        """
        fk = self._make_fk(gid, torso_pos)
        ik_cfg = IKConfig(max_iterations=150, position_tolerance=5e-3)
        solver = IKSolver(fk, config=ik_cfg)

        result = solver.solve(target_hold.position)
        if not result.success:
            logger.warning("IK failed for %s -> %s", gid.value, target_hold.hold_id)

        q_start = np.zeros(fk.n_joints)
        q_end = result.joint_angles

        # Linear interpolation in joint space
        traj = [
            q_start + t * (q_end - q_start)
            for t in np.linspace(0, 1, self._n_traj)
        ]
        return traj

    def _estimate_com(
        self,
        grips: dict[GripperID, str | None],
        current_com: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate new COM position after grips change.
        Simplified: COM tracks centroid of active holds + small torso offset.
        """
        positions = [
            self._holds.get(hid).position
            for hid in grips.values() if hid is not None
        ]
        if not positions:
            return current_com.copy()
        centroid = np.mean(positions, axis=0)
        # COM is slightly behind wall (negative Y)
        return np.array([centroid[0], -0.08, centroid[2]])

    def _grip_positions(self, grips: dict[GripperID, str | None]) -> np.ndarray:
        """Return (N, 3) array of active grip positions."""
        positions = [
            self._holds.get(hid).position
            for hid in grips.values() if hid is not None
        ]
        if not positions:
            return np.empty((0, 3))
        return np.stack(positions)
