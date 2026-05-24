"""
SE3 transform utilities — rotation matrices, quaternions, homogeneous transforms.

All functions operate on numpy arrays. No simulator dependency.
"""

from __future__ import annotations

import numpy as np


# ── Rotation ─────────────────────────────────────────────────────────────────


def rot_x(angle: float) -> np.ndarray:
    """3x3 rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    """3x3 rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix around Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles → 3x3 rotation matrix."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """3x3 rotation matrix → (roll, pitch, yaw) ZYX Euler angles."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula: axis (unit vector) + angle → R."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# ── Homogeneous transforms ────────────────────────────────────────────────────


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    """Efficient inverse of a homogeneous transform."""
    R, t = T[:3, :3], T[:3, 3]
    T_i = np.eye(4)
    T_i[:3, :3] = R.T
    T_i[:3, 3] = -R.T @ t
    return T_i


def apply_T(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transform to point(s).

    Args:
        T: (4,4) homogeneous transform
        points: (3,) or (N,3) array of 3D points

    Returns:
        Transformed (3,) or (N,3) array.
    """
    pts = np.atleast_2d(points)
    ones = np.ones((pts.shape[0], 1))
    h = np.hstack([pts, ones])
    result = (T @ h.T).T[:, :3]
    return result[0] if points.ndim == 1 else result


# ── DH parameter transform ────────────────────────────────────────────────────


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Standard Denavit-Hartenberg (DH) transform.

    Args:
        a:     link length along X_{i-1}
        alpha: link twist around X_{i-1}
        d:     link offset along Z_i
        theta: joint angle around Z_i

    Returns:
        4x4 homogeneous transform T_{i-1,i}
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st * ca,  st * sa,  a * ct],
        [st,     ct * ca, -ct * sa,  a * st],
        [0.0,        sa,       ca,       d],
        [0.0,       0.0,      0.0,     1.0],
    ])
