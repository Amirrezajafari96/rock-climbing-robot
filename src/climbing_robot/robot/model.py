"""
Robot model abstraction.

RobotModel wraps the simulator and exposes robot-specific queries:
  - Current joint configuration
  - Gripper end-effector positions
  - COM computation
  - Joint control interface
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from climbing_robot.planner.base_planner import GripperID
from climbing_robot.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)

# Joint indices in the actuator control vector (matches climbing_robot.xml order)
JOINT_CTRL_MAP: dict[GripperID, list[int]] = {
    GripperID.LEFT:   [0, 1, 2],
    GripperID.RIGHT:  [3, 4, 5],
    GripperID.CENTER: [6, 7, 8],
}

GRIPPER_SITE_NAMES: dict[GripperID, str] = {
    GripperID.LEFT:   "left_gripper_site",
    GripperID.RIGHT:  "right_gripper_site",
    GripperID.CENTER: "center_gripper_site",
}

BODY_MASSES: dict[str, float] = {
    "torso":            2.0,
    "left_upper_arm":   0.3,
    "left_lower_arm":   0.2,
    "right_upper_arm":  0.3,
    "right_lower_arm":  0.2,
    "center_upper_arm": 0.4,
    "center_lower_arm": 0.25,
}


@dataclass
class RobotConfig:
    """Physical robot configuration parameters."""
    n_limbs: int = 3
    upper_arm_length: float = 0.25
    lower_arm_length: float = 0.20
    shoulder_span: float = 0.70      # total horizontal span [m]
    total_mass: float = 3.75         # [kg]
    grip_force_threshold: float = 0.5  # [N] minimum touch force for grip


class RobotModel:
    """
    High-level robot model interface.

    Sits above the simulator and provides robot-semantic methods:
    set_limb_joints, get_gripper_position, compute_com, etc.

    Args:
        simulator: Active simulator instance.
        config:    RobotConfig for this robot.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        config: RobotConfig | None = None,
    ) -> None:
        self._sim = simulator
        self._cfg = config or RobotConfig()

    # ── Joint control ─────────────────────────────────────────────────────

    def set_limb_joints(
        self, gid: GripperID, joint_angles: np.ndarray
    ) -> None:
        """
        Set position targets for a specific limb's joints.

        Args:
            gid:          Which limb.
            joint_angles: 3-element array [shoulder_x, shoulder_z, elbow].
        """
        state = self._sim.get_state()
        ctrl = state.ctrl.copy()
        indices = JOINT_CTRL_MAP[gid]
        for i, idx in enumerate(indices):
            ctrl[idx] = float(joint_angles[i])
        self._sim.set_control(ctrl)

    def set_all_joints(self, ctrl_vector: np.ndarray) -> None:
        """Set the full 9-DOF control vector directly."""
        self._sim.set_control(ctrl_vector)

    def get_joint_angles(self, gid: GripperID) -> np.ndarray:
        """Return current joint angles for one limb (from sensor/state)."""
        state = self._sim.get_state()
        indices = JOINT_CTRL_MAP[gid]
        return state.qpos[7 + min(indices): 7 + max(indices) + 1].copy()

    # ── Kinematics ────────────────────────────────────────────────────────

    def get_gripper_position(self, gid: GripperID) -> np.ndarray:
        """World-frame position of a gripper end-effector."""
        return self._sim.get_site_position(GRIPPER_SITE_NAMES[gid])

    def get_all_gripper_positions(self) -> dict[GripperID, np.ndarray]:
        return {gid: self.get_gripper_position(gid) for gid in GripperID}

    def compute_com(self) -> np.ndarray:
        """Compute COM from body masses and simulator body positions."""
        total_mass = sum(BODY_MASSES.values())
        com = np.zeros(3)
        for body_name, mass in BODY_MASSES.items():
            try:
                pos = self._sim.get_body_position(body_name)
                com += mass * pos
            except KeyError:
                logger.warning("Body '%s' not found in simulator", body_name)
        return com / total_mass

    # ── Contact sensing ───────────────────────────────────────────────────

    def is_gripping(self, gid: GripperID) -> bool:
        """
        Returns True if the gripper has significant contact force.
        Uses the touch sensor reading from the simulator.
        """
        sensor_map = {
            GripperID.LEFT:   "left_touch",
            GripperID.RIGHT:  "right_touch",
            GripperID.CENTER: "center_touch",
        }
        try:
            touch = self._sim.get_sensor(sensor_map[gid])
            return float(np.linalg.norm(touch)) > self._cfg.grip_force_threshold
        except KeyError:
            return False

    def get_active_grippers(self) -> dict[GripperID, bool]:
        return {gid: self.is_gripping(gid) for gid in GripperID}
