"""
Forward Kinematics (FK) solver using DH parameters.

Each limb of the climbing robot is modeled as a 3-DOF serial chain:
  Joint 0: shoulder_x  — hinge around X
  Joint 1: shoulder_z  — hinge around Z (twist)
  Joint 2: elbow       — hinge around X

DH convention: standard (Hartenberg & Denavit, 1955).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from climbing_robot.kinematics.transforms import dh_transform, make_T

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DHLink:
    """
    One link in a DH kinematic chain.

    Attributes:
        a:     link length [m]
        alpha: link twist  [rad]
        d:     link offset [m]
        theta_offset: constant angle added to the joint variable [rad]
    """

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0


@dataclass
class FKResult:
    """Result of a forward kinematics computation."""

    end_effector_pos: np.ndarray      # world-frame position [3]
    end_effector_rot: np.ndarray      # world-frame rotation [3,3]
    transforms: list[np.ndarray]      # T_base_to_joint_i for each joint
    joint_positions: list[np.ndarray] # world-frame position of each joint


class FKSolver:
    """
    Forward kinematics for a 3-DOF limb using DH parameters.

    The robot has three identical limb chains (left, right, center)
    that differ only in their base transform (shoulder origin on torso).

    Args:
        links:       DH parameter list, one per joint.
        base_transform: 4x4 homogeneous transform from world to limb base.

    Example::

        links = [
            DHLink(a=0.0, alpha=np.pi/2, d=0.0),   # shoulder_x
            DHLink(a=0.0, alpha=0.0,     d=0.0),   # shoulder_z
            DHLink(a=0.25, alpha=0.0,    d=0.0),   # elbow (upper arm)
        ]
        solver = FKSolver(links, base_transform=np.eye(4))
        result = solver.solve([0.3, 0.1, -1.2])
        print(result.end_effector_pos)
    """

    def __init__(
        self,
        links: list[DHLink],
        base_transform: np.ndarray | None = None,
    ) -> None:
        if not links:
            raise ValueError("links list must not be empty")
        self._links = links
        self._base_T = base_transform if base_transform is not None else np.eye(4)

    @property
    def n_joints(self) -> int:
        return len(self._links)

    def solve(self, joint_angles: list[float] | np.ndarray) -> FKResult:
        """
        Compute forward kinematics for a given joint configuration.

        Args:
            joint_angles: List of n_joints joint angles [rad].

        Returns:
            FKResult with end-effector pose and per-joint transforms.

        Raises:
            ValueError: if joint_angles length != n_joints.
        """
        q = np.asarray(joint_angles, dtype=float)
        if len(q) != self.n_joints:
            raise ValueError(
                f"Expected {self.n_joints} joint angles, got {len(q)}"
            )

        T = self._base_T.copy()
        transforms: list[np.ndarray] = []
        joint_positions: list[np.ndarray] = []

        for i, (link, angle) in enumerate(zip(self._links, q)):
            theta = angle + link.theta_offset
            T_i = dh_transform(link.a, link.alpha, link.d, theta)
            T = T @ T_i
            transforms.append(T.copy())
            joint_positions.append(T[:3, 3].copy())

        return FKResult(
            end_effector_pos=T[:3, 3].copy(),
            end_effector_rot=T[:3, :3].copy(),
            transforms=transforms,
            joint_positions=joint_positions,
        )

    def jacobian(self, joint_angles: list[float] | np.ndarray) -> np.ndarray:
        """
        Compute the geometric Jacobian J (6 x n_joints).

        The Jacobian maps joint velocities to end-effector twist:
            [v; ω] = J * dq/dt

        Uses the standard z-column cross-product formula for revolute joints.

        Returns:
            J: (6, n_joints) numpy array.
        """
        q = np.asarray(joint_angles, dtype=float)
        result = self.solve(q)

        pe = result.end_effector_pos  # end-effector position

        J = np.zeros((6, self.n_joints))
        T = self._base_T.copy()

        for i, (link, angle) in enumerate(zip(self._links, q)):
            theta = angle + link.theta_offset
            z_i = T[:3, 2]          # z-axis of joint i frame
            p_i = T[:3, 3]          # origin of joint i frame
            J[:3, i] = np.cross(z_i, pe - p_i)   # linear velocity part
            J[3:, i] = z_i                         # angular velocity part
            T_i = dh_transform(link.a, link.alpha, link.d, theta)
            T = T @ T_i

        return J


# ── Robot-specific FK factory ─────────────────────────────────────────────────


def make_limb_fk(
    upper_arm_length: float = 0.25,
    lower_arm_length: float = 0.20,
    base_transform: np.ndarray | None = None,
) -> FKSolver:
    """
    Build an FKSolver for one climbing-robot limb.

    Link layout:
      - shoulder_x  : rotates around X, zero offset
      - shoulder_z  : rotates around Z (shoulder twist), zero offset
      - elbow       : rotates around X, upper_arm_length offset

    The end-effector is at the gripper sphere center, lower_arm_length
    from the elbow joint.

    Args:
        upper_arm_length: distance from shoulder to elbow [m].
        lower_arm_length: distance from elbow to gripper  [m].
        base_transform:   4x4 transform from world to shoulder origin.
    """
    links = [
        DHLink(a=0.0,              alpha=np.pi / 2, d=0.0),  # shoulder_x
        DHLink(a=0.0,              alpha=-np.pi / 2, d=0.0), # shoulder_z
        DHLink(a=upper_arm_length, alpha=np.pi / 2, d=0.0),  # elbow (upper arm)
        DHLink(a=lower_arm_length, alpha=0.0,        d=0.0), # gripper tip
    ]
    return FKSolver(links, base_transform=base_transform)
