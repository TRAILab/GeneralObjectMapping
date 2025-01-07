"""

Deal with differentiable cameras during NeRF rendering.

"""

import numpy as np
import torch
from shap_e.models.nn.camera import (
    DifferentiableCameraBatch,
    DifferentiableProjectiveCamera,
)


def create_pan_cameras_with_grad(size: int, device: torch.device) -> DifferentiableCameraBatch:
    """
    Automatically generate 20 views around the object.
    """
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0))
            .float()
            .to(device)
            .requires_grad_(True),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device).requires_grad_(True),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device).requires_grad_(True),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device).requires_grad_(True),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )


def pose_to_xyz(t_co):
    """
    @ t_cw: 4x4 matrix

    @ return: origin, x, y, z
        Note that we consider scale, so the x,y,z coordinates may not be
        unit vectors.
    """

    # 1. get R, t from pose
    R = t_co[:3, :3]
    t = t_co[:3, 3]

    # 2. get x, y, z from R
    # R_trans = R.T
    # x = -R_trans[0, :]
    # y = -R_trans[1, :]

    R_inv = torch.inverse(R)

    x = R_inv[:, 0]  # c 's x axis in object frame
    y = R_inv[:, 1]

    z = R_inv[:, 2]

    # R_T_inv = torch.inverse(R.T)
    origin = -R_inv @ t  # c 's center in object frame

    return origin, x, y, z


def convert_intrinsics_to_fov(K):
    # Extract the focal length and principal point from K
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Assume the sensor size is the same as the image size
    width = 2 * cx
    height = 2 * cy

    # Convert the focal length to field of view
    x_fov = 2 * np.arctan(width / (2 * fx))
    y_fov = 2 * np.arctan(height / (2 * fy))

    return x_fov, y_fov, width, height


def create_cameras_with_grad_from_pose(
    t_cw: torch.Tensor, K: torch.Tensor, resize=1.0
) -> DifferentiableCameraBatch:
    """
    Automatically generate 20 views around the object.

    @ origins:
    @ x:
    @ y:
    @ z:

    @ resize: divide the areas by this scale
    """
    origins = []
    xs = []
    ys = []
    zs = []

    # transform pose into camera
    origin, x, y, z = pose_to_xyz(t_cw)

    origins.append(origin)
    xs.append(x)
    ys.append(y)
    zs.append(z)

    # convert K to fov
    x_fov, y_fov, width, height = convert_intrinsics_to_fov(K)

    # round
    width = int(width / resize)
    height = int(height / resize)

    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.stack(origins, axis=0),
            x=torch.stack(xs, axis=0),
            y=torch.stack(ys, axis=0),
            z=torch.stack(zs, axis=0),
            width=width,
            height=height,
            x_fov=x_fov,
            y_fov=y_fov,
        ),
    )
