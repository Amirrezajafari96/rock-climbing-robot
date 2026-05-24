"""
Inverse Kinematics (IK) solver.

Method: Jacobian pseudoinverse with null-space optimization.
  - Primary task:    position end-effector at target [x, y, z]
  - Secondary task:  stay near joint limit centers (null-space)
  - Step size:       damped least-squares (Levenberg-Marquardt)

Reference:
  Siciliano et al., "Robotics: Modelling, Planning and Control", 2009.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import pinv

from climbing_robot.kinematics.forward import FKSolver

logger = logging.getLogger(__name__)


@dataclass
class IKConfig:
    """Configuration for the IK solver."""

    max_iterations: int = 200
    position_tolerance: float = 1e-3    # [m] convergence threshold
    step_size: float = 0.5              # gradient step scale
    damping: float = 1e-4              # Tikhonov regularization λ
    null_space_weight: float = 0.1      # secondary task weight
    joint_limits_lower: list[float] = field(default_factory=list)
    joint_limits_upper: list[float] = field(default_factory=list)


@dataclass
class IKResult:
    """Result of an IK solve attempt."""

    success: bool
    joint_angles: np.ndarray           # final joint configuration [n_joints]
    final_error: float                 # residual position error [m]
    iterations: int                    # number of iterations taken
    trajectory: list[np.ndarray] = field(default_factory=list)  # joint configs per iter


class IKSolver:
    """
    Jacobian pseudoinverse IK solver for a serial-chain limb.

    Supports:
      - 3-DOF position-only task (x, y, z)
      - Joint limit clamping
      - Null-space secondary task (stay near rest pose)
      - Warm-starting from an initial guess

    Args:
        fk_solver: FKSolver for the same kinematic chain.
        config:    IKConfig with tunable parameters.

    Example::

        fk = make_limb_fk(base_transform=shoulder_T)
        ik = IKSolver(fk)
        result = ik.solve(target_pos=np.array([0.0, -0.05, 0.55]))
        if result.success:
            sim.set_joint_angles(result.joint_angles)
    """

    def __init__(self, fk_solver: FKSolver, config: IKConfig | None = None) -> None:
        self._fk = fk_solver
        self._cfg = config or IKConfig()

        n = self._fk.n_joints
        if not self._cfg.joint_limits_lower:
            self._cfg.joint_limits_lower = [-np.pi] * n
        if not self._cfg.joint_limits_upper:
            self._cfg.joint_limits_upper = [np.pi] * n

        self._lower = np.array(self._cfg.joint_limits_lower)
        self._upper = np.array(self._cfg.joint_limits_upper)
        self._rest_pose = (self._lower + self._upper) / 2.0

    # ── Public API ────────────────────────────────────────────────────────

    def solve(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,
        *,
        record_trajectory: bool = False,
    ) -> IKResult:
        """
        Solve IK for a target end-effector position.

        Args:
            target_pos:       Desired end-effector position [x, y, z].
            q_init:           Initial joint configuration. If None, uses
                              the midpoint of joint limits.
            record_trajectory: If True, store q at each iteration.

        Returns:
            IKResult with solution quality information.
        """
        target = np.asarray(target_pos, dtype=float)
        q = q_init.copy() if q_init is not None else self._rest_pose.copy()
        trajectory: list[np.ndarray] = []

        cfg = self._cfg

        for i in range(cfg.max_iterations):
            fk_result = self._fk.solve(q)
            error = target - fk_result.end_effector_pos
            err_norm = float(np.linalg.norm(error))

            if record_trajectory:
                trajectory.append(q.copy())

            if err_norm < cfg.position_tolerance:
                logger.debug("IK converged in %d iterations, error=%.4f", i, err_norm)
                return IKResult(
                    success=True,
                    joint_angles=q,
                    final_error=err_norm,
                    iterations=i,
                    trajectory=trajectory,
                )

            # Jacobian (position rows only: first 3 rows)
            J = self._fk.jacobian(q)[:3, :]

            # Damped least-squares pseudoinverse: J^+ = J^T (J J^T + λI)^{-1}
            lam = cfg.damping
            JJT = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJT + lam * np.eye(3))

            # Primary task: move toward target
            dq_primary = cfg.step_size * J_dls @ error

            # Secondary task: stay near rest pose via null-space projection
            N = np.eye(self._fk.n_joints) - J_dls @ J  # null-space projector
            dq_null = cfg.null_space_weight * (self._rest_pose - q)
            dq_secondary = N @ dq_null

            q = q + dq_primary + dq_secondary

            # Clamp to joint limits
            q = np.clip(q, self._lower, self._upper)

        # Did not converge
        fk_final = self._fk.solve(q)
        final_err = float(np.linalg.norm(target - fk_final.end_effector_pos))
        logger.warning(
            "IK did not converge after %d iterations, final_error=%.4f",
            cfg.max_iterations, final_err,
        )
        return IKResult(
            success=False,
            joint_angles=q,
            final_error=final_err,
            iterations=cfg.max_iterations,
            trajectory=trajectory,
        )

    def solve_batch(
        self,
        target_positions: list[np.ndarray],
        q_init: np.ndarray | None = None,
    ) -> list[IKResult]:
        """
        Solve IK for a sequence of target positions (warm-starts each from previous).

        Useful for computing joint trajectories to a hold.
        """
        results: list[IKResult] = []
        q = q_init

        for target in target_positions:
            result = self.solve(target, q_init=q)
            results.append(result)
            if result.success:
                q = result.joint_angles  # warm-start next solve

        return results

    def is_reachable(self, target_pos: np.ndarray) -> bool:
        """Quick reachability check (does IK converge?)."""
        result = self.solve(target_pos)
        return result.success
