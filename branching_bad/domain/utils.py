import numpy as np
import torch as th


def euler2quat(euler_angles):
    roll, pitch, yaw = euler_angles
    # roll, yaw, pitch = euler_angles
    # pitch, roll, yaw = euler_angles
    # pitch, yaw, roll = euler_angles
    # yaw, roll, pitch = euler_angles
    # yaw, pitch, roll = euler_angles
    # cr, cp, cy = th.cos(euler_angles / 2)
    # sr, sp, sy = th.sin(euler_angles / 2)
    cr, cp, cy = np.cos(euler_angles / 2)
    sr, sp, sy = np.sin(euler_angles / 2)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return np.stack([qw, qx, qy, qz])


def get_points(res):
    points = np.stack(np.meshgrid(
        range(res), range(res), indexing="ij"), axis=-1)
    points = points.astype(np.float32)
    points = ((points + 0.5) / res - 0.5) * 2
    points = th.from_numpy(points).float()
    points = th.reshape(points, (-1, 2))
    return points
