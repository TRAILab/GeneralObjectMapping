"""
Loss functions during optimization.
"""

import numpy as np
import torch


def calculate_3d_loss(
    shape_model, t_ob_cur, latent, pts_3d_b, points_sample_each_iter=None, loss_type="mse"
):
    """

    Sample 3D points from pointcloud.
    Get SDF Loss equal to Zero.

    """
    # random choose config_sample_points_each_iter pts from pts_3d_b
    if points_sample_each_iter is None or len(pts_3d_b) < points_sample_each_iter:
        pts_3d_b_sample = pts_3d_b  # get all
    else:
        pts_3d_b_sample = pts_3d_b[
            np.random.choice(len(pts_3d_b), points_sample_each_iter, replace=False)
        ]

    # transform (N,3) points pts_3d_b, use R and t
    pts_3d_o = (pts_3d_b_sample[..., None, :3] * t_ob_cur[:3, :3]).sum(-1) + t_ob_cur[:3, 3]

    # Clip the points to the range of [-1, 1]
    pts_3d_o = torch.clamp(pts_3d_o, -1, 1)

    loss_3d, sdfs = shape_model.get_3d_surface_loss(latent, pts_3d_o, loss_type)

    # if use loss_3d_away, then add it to loss_3d
    # loss_3d = loss_3d + loss_3d_away
    loss_3d = loss_3d

    # for visualization
    output = {"pts_3d_b_sample": pts_3d_b_sample}

    return loss_3d, output


def get_loss_regularizer(
    latent, init_latent, prior_latents, method="point", prior_latent_distribution_mean_std=None
):
    """
    Calculate the loss for regularizing the latent.

    The latent should be near the initialized latent (given by diffusion prior).

    @ prior_latents: None, or a stacked tensor of latents. Size: (N,1024x1024)

    @ method: point or distribution
    """
    if method == "single" or prior_latents is None or len(prior_latents) == 1:
        loss_regularizer = (latent - init_latent).square().mean()
    else:
        # two types: point / distribution
        if method == "point-average":
            # option 1: average
            loss_regularizer = (prior_latents - latent).square().mean()
        elif method == "point-minimum":
            # option 2: minimum; note the chosen one will be zero, and almost the same
            loss_regularizer = (prior_latents - latent).square().mean(1).min()
        elif method == "distribution":
            # reconstruct a distribution and constrain with the distribution
            if prior_latent_distribution_mean_std is None:
                raise ValueError("Prior Latent Distribution Mean and Std should be provided.")

            mean = prior_latent_distribution_mean_std[0]
            std = prior_latent_distribution_mean_std[1]

            # Constraining the latent to the distribution; directly calculate
            epsilon = 1e-8
            loss_regularizer = (
                (latent - mean).square() / (std * std + epsilon)
            ).mean()  # + torch.log(std)

        elif method == "energy_score":
            # TODO: Energy score!
            raise NotImplementedError
        else:
            # L2 loss / KL Divergence / Energy Score ?
            raise NotImplementedError

    return loss_regularizer


def calculate_away_loss(pts_3d_o, dis=1.0):
    """
    Calculate the loss for points away from the range of [-1, 1].

    @ pts_3d_o: (N,3) tensor, points in object coordinate
    @ dis: distance to the boundary
    """

    # only consider points outside of the range
    # inside range: x,y,z all in [-1, 1]
    pts_3d_o_inrange = torch.abs(pts_3d_o) - dis
    pts_3d_o_inrange[pts_3d_o_inrange < 0] = 0

    loss = torch.sum(pts_3d_o_inrange)

    # average loss
    loss = loss / len(pts_3d_o)

    return loss
