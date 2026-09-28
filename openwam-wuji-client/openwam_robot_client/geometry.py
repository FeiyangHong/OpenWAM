from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"quaternion must be finite with shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-10:
        raise ValueError("quaternion norm is zero")
    quat /= norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def quaternion_xyzw_to_matrix(value: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(value).astype(np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError(f"rotation matrix must be finite with shape (3,3), got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        x, y, z, w = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            x, y, z, w = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
        elif index == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            x, y, z, w = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            x, y, z, w = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    return normalize_quaternion(np.array([x, y, z, w]))


def pose7_to_eef9_column(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"pose must be finite with shape (7,), got {pose.shape}")
    rotation = quaternion_xyzw_to_matrix(pose[3:])
    return np.concatenate((pose[:3], rotation[:, 0], rotation[:, 1])).astype(np.float32)


def eef9_column_to_pose7(eef: np.ndarray) -> np.ndarray:
    eef = np.asarray(eef, dtype=np.float32)
    if eef.shape != (9,) or not np.isfinite(eef).all():
        raise ValueError(f"EEF must be finite with shape (9,), got {eef.shape}")
    column0 = eef[3:6]
    column1 = eef[6:9]
    norm0 = float(np.linalg.norm(column0))
    if norm0 < 1e-8:
        raise ValueError("rot6d first column is degenerate")
    column0 = column0 / norm0
    column1 = column1 - float(np.dot(column0, column1)) * column0
    norm1 = float(np.linalg.norm(column1))
    if norm1 < 1e-8:
        raise ValueError("rot6d second column is degenerate")
    column1 = column1 / norm1
    rotation = np.stack((column0, column1, np.cross(column0, column1)), axis=1)
    return np.concatenate((eef[:3], matrix_to_quaternion_xyzw(rotation))).astype(np.float32)


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = normalize_quaternion(start).astype(np.float64)
    end = normalize_quaternion(end).astype(np.float64)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end, dot = -end, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(start + fraction * (end - start))
    theta = math.acos(dot)
    return normalize_quaternion(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * start
        + math.sin(fraction * theta) / math.sin(theta) * end
    )


def clip_pose_step(target: np.ndarray, previous: np.ndarray, xyz_limit: float, rotation_limit_rad: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    previous = np.asarray(previous, dtype=np.float32)
    delta = target[:3] - previous[:3]
    distance = float(np.linalg.norm(delta))
    if xyz_limit > 0.0 and distance > xyz_limit:
        target[:3] = previous[:3] + delta * (xyz_limit / distance)
    target[3:] = normalize_quaternion(target[3:])
    previous_quat = normalize_quaternion(previous[3:])
    angle = 2.0 * math.acos(float(np.clip(abs(np.dot(previous_quat, target[3:])), 0.0, 1.0)))
    if rotation_limit_rad > 0.0 and angle > rotation_limit_rad:
        target[3:] = slerp(previous_quat, target[3:], rotation_limit_rad / angle)
    return target
