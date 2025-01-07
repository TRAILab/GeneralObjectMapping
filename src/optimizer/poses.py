"""
Utils functions to deal with poses in optimization.
"""

import math

import numpy as np
import torch

from src.utils import SE3
from src.utils.geometry import Oplus_se3


def extract_delta_pose_variable(t_wo, device="cuda", dtype=torch.float32, optimize_pose=True):
    """
    Preprocess pose to generate required structure.

    @ t_wo: SE(3) + Scale;
    """
    if t_wo is None:
        # init a se3 pose of zeros(6,1)
        t_wo = np.eye(4)

    # Decouple scale from pose
    trans_wo, q_wo, s_wo = SE3.decompose_mat4(t_wo)
    t_wo_noscale = SE3.compose_mat4(trans_wo, q_wo, np.ones(3))

    pose_ini = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=dtype, device=device)

    dPose_wo = pose_ini.clone().detach().requires_grad_(optimize_pose)

    ## 3-DOF Scale
    scale_log_x = math.log(s_wo[0])
    scale_log_y = math.log(s_wo[1])
    scale_log_z = math.log(s_wo[2])

    scale_ini = torch.tensor(
        [scale_log_x, scale_log_y, scale_log_z], dtype=dtype, device=device
    )  # log

    dScale_log_wo = scale_ini.clone().detach().requires_grad_(optimize_pose)

    # t_wo_noscale to torch
    t_wo_noscale = torch.from_numpy(t_wo_noscale).to(dtype=dtype, device=device)

    return t_wo_noscale, dPose_wo, dScale_log_wo


def get_pose_from_delta(t_noscale, dPose, dScale_log):
    """

    Input:
        @t_noscale: 4x4 mat, w/o scale
        @dPose: 6x1 vector, w/o scale
        @dScale_log: 3x1 vector, w/o scale

    Output:
        @t_update: 4x4 mat, w/ scale

    """
    device = t_noscale.device

    t_noscale_update = Oplus_se3(t_noscale, dPose).to(device)  # w/o scale

    scale_4 = torch.cat([torch.exp(dScale_log), torch.tensor([1.0]).to(device)], -1)
    scaleMtx = torch.diag(scale_4)

    t_update = torch.mm(t_noscale_update, scaleMtx)
    return t_update


def get_current_pose_from_delta_pose(
    t_ob_noscale, dPose_ob, dScale_log_ob, init_pose_bo: np.array, flag_optimize_pose
):

    if flag_optimize_pose:
        t_ob_cur = get_pose_from_delta(t_ob_noscale, dPose_ob, dScale_log_ob)
        t_bo_cur = torch.inverse(t_ob_cur)
    else:
        # print("Use original pose for reconstruction experiments.")
        t_bo_cur = torch.from_numpy(init_pose_bo).float().cuda()
        t_ob_cur = torch.inverse(t_bo_cur)

    return t_ob_cur, t_bo_cur
