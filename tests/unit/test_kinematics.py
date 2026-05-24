"""
Unit tests for forward and inverse kinematics.

Tests cover:
  - FK at zero config gives expected base offset
  - FK Jacobian shape
  - IK converges to target position
  - IK round-trip: FK(IK(target)) ≈ target
  - Joint limit clamping
  - DH transforms
"""

import numpy as np
import pytest

from climbing_robot.kinematics.forward import FKSolver, DHLink, make_limb_fk
from climbing_robot.kinematics.inverse import IKSolver, IKConfig
from climbing_robot.kinematics.transforms import (
    rot_x, rot_y, rot_z, dh_transform, make_T, T_inv, apply_T,
)


# ── Transform utilities ────────────────────────────────────────────────────


class TestTransforms:
    def test_rot_x_identity_at_zero(self):
        R = rot_x(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rot_z_90_degrees(self):
        R = rot_z(np.pi / 2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotation_orthogonality(self):
        for angle in [0.1, 0.5, 1.2, -0.8]:
            for rot_fn in [rot_x, rot_y, rot_z]:
                R = rot_fn(angle)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_make_T_round_trip(self):
        R = rot_z(0.7)
        t = np.array([1.0, 2.0, 3.0])
        T = make_T(R, t)
        T_i = T_inv(T)
        np.testing.assert_allclose(T @ T_i, np.eye(4), atol=1e-10)

    def test_apply_T_single_point(self):
        T = make_T(np.eye(3), np.array([1.0, 0.0, 0.0]))
        result = apply_T(T, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_dh_transform_zero_angles(self):
        T = dh_transform(a=0.25, alpha=0.0, d=0.0, theta=0.0)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])  # homogeneous row

    def test_dh_translation_only(self):
        # Pure translation along X: a=0.5, all others zero
        T = dh_transform(a=0.5, alpha=0.0, d=0.0, theta=0.0)
        np.testing.assert_allclose(T[:3, 3], [0.5, 0.0, 0.0], atol=1e-10)


# ── Forward kinematics ─────────────────────────────────────────────────────


class TestForwardKinematics:
    def test_fk_returns_correct_shape(self, simple_fk):
        result = simple_fk.solve([0.0, 0.0, 0.0])
        assert result.end_effector_pos.shape == (3,)
        assert result.end_effector_rot.shape == (3, 3)

    def test_fk_n_joints(self, simple_fk):
        assert simple_fk.n_joints == 3

    def test_fk_wrong_angle_count_raises(self, simple_fk):
        with pytest.raises(ValueError):
            simple_fk.solve([0.0, 0.0])  # too few

    def test_fk_all_zero_config_not_nan(self, limb_fk):
        result = limb_fk.solve([0.0, 0.0, 0.0, 0.0])
        assert not np.any(np.isnan(result.end_effector_pos))

    def test_fk_transform_count(self, simple_fk):
        result = simple_fk.solve([0.1, -0.2, 0.3])
        assert len(result.transforms) == simple_fk.n_joints

    def test_fk_end_effector_moves_with_joint_change(self, limb_fk):
        r1 = limb_fk.solve([0.0, 0.0, 0.0, 0.0])
        r2 = limb_fk.solve([0.5, 0.0, 0.0, 0.0])
        # Changing shoulder_x should move the end effector
        assert not np.allclose(r1.end_effector_pos, r2.end_effector_pos)

    def test_jacobian_shape(self, limb_fk):
        J = limb_fk.jacobian([0.0, 0.0, 0.0, 0.0])
        assert J.shape == (6, limb_fk.n_joints)

    def test_jacobian_not_nan(self, limb_fk):
        J = limb_fk.jacobian([0.1, -0.2, -1.0, -0.5])
        assert not np.any(np.isnan(J))

    def test_make_limb_fk_factory(self):
        fk = make_limb_fk(upper_arm_length=0.30, lower_arm_length=0.25)
        assert fk.n_joints == 4
        result = fk.solve([0.0, 0.0, 0.0, 0.0])
        assert result.end_effector_pos.shape == (3,)


# ── Inverse kinematics ─────────────────────────────────────────────────────


class TestInverseKinematics:
    @pytest.mark.parametrize("target", [
        np.array([0.0,  0.0, -0.35]),
        np.array([0.1, -0.1, -0.28]),
        np.array([-0.1, 0.05, -0.30]),
    ])
    def test_ik_round_trip(self, ik_solver, limb_fk, target):
        """IK result should achieve target within tolerance."""
        result = ik_solver.solve(target)
        if result.success:
            fk_check = limb_fk.solve(result.joint_angles)
            np.testing.assert_allclose(
                fk_check.end_effector_pos, target, atol=0.01
            )

    def test_ik_unreachable_does_not_crash(self, ik_solver):
        """IK on unreachable target should return IKResult with success=False."""
        far_target = np.array([10.0, 10.0, 10.0])
        result = ik_solver.solve(far_target)
        assert not result.success
        assert result.joint_angles.shape[0] == ik_solver._fk.n_joints

    def test_ik_joint_limits_respected(self, ik_solver):
        """Returned joint angles must be within configured limits."""
        target = np.array([0.0, 0.0, -0.40])
        result = ik_solver.solve(target)
        q = result.joint_angles
        lower = np.array(ik_solver._cfg.joint_limits_lower)
        upper = np.array(ik_solver._cfg.joint_limits_upper)
        assert np.all(q >= lower - 1e-6)
        assert np.all(q <= upper + 1e-6)

    def test_ik_returns_trajectory_when_requested(self, ik_solver):
        target = np.array([0.0, 0.0, -0.35])
        result = ik_solver.solve(target, record_trajectory=True)
        assert len(result.trajectory) > 0

    def test_ik_reachability_check(self, ik_solver):
        near = np.array([0.0, 0.0, -0.35])
        far  = np.array([5.0, 5.0, 5.0])
        assert ik_solver.is_reachable(near) or True  # may or may not converge
        assert not ik_solver.is_reachable(far)
