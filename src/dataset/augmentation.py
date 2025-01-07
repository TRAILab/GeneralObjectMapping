import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils import SE3


def generate_gt_scannet_noise(t_wo, noise_level, deterministic=True):
    t_wo_noisy = t_wo.copy()

    # get the length of diagonal line of the cuboid
    trans_wo, q_wo, s_wo = SE3.decompose_mat4(t_wo_noisy)
    # diagnal_length = np.linalg.norm(s_wo)

    sigma_trans_dis = (0.1 * s_wo) * noise_level
    sigma_rot_rad = np.pi / 180.0 * 10.0 * noise_level
    sigma_scale = 0.1 * noise_level

    # sample from a gaussian distribution
    if deterministic:
        # set a random seed for np
        # the seed is from the init pose
        noise_trans_dis = sigma_trans_dis
        noise_rot_rad = sigma_rot_rad
        noise_scale = sigma_scale
    else:
        noise_trans_dis = np.random.normal(scale=sigma_trans_dis, size=3)
        noise_rot_rad = np.random.normal(scale=sigma_rot_rad, size=1)
        noise_scale = np.random.normal(scale=sigma_scale, size=3)

    # translation
    t_wo_noisy[0:3, 3] += noise_trans_dis

    r = R.from_euler("y", noise_rot_rad, degrees=False)
    rot_per = r.as_matrix()

    t_wo_noisy[:3, :3] = np.matmul(t_wo_noisy[:3, :3], rot_per)  # right multiplication

    # scale: note x,y,z have 3 different scales
    scale_noise = 1.0 + noise_scale
    t_wo_noisy[:3, :3] = t_wo_noisy[:3, :3] * scale_noise

    out = t_wo_noisy

    return out
