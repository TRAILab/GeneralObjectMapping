"""
Basic functions for diffusion prior parameters
"""

import numpy as np
import torch


def sigma_to_t(sigma, diffusion):
    from scipy import interpolate

    alpha_cumprod_to_t = interpolate.interp1d(
        diffusion.alphas_cumprod, np.arange(0, diffusion.num_timesteps)
    )

    alpha_cumprod = 1.0 / (sigma**2 + 1)
    if alpha_cumprod > diffusion.alphas_cumprod[0]:
        return 0
    elif alpha_cumprod <= diffusion.alphas_cumprod[-1]:
        return diffusion.num_timesteps - 1
    else:
        return float(alpha_cumprod_to_t(alpha_cumprod))


def calculate_alpha_t_legacy(sigmas, it):
    s_t = calculate_s_t(sigmas, it)

    alpha_t = (1 - s_t.square()).sqrt()

    return alpha_t


def get_sqrt_alphas_cumprod(sigma, shape_model):
    """
    # Use pre-cacalculated constant values for prior constraints

    # @sigma: use sigma_to_t function to find its corresponding t in (0,1024)

    # @it: iteration of current diffusion step; note that, larger the it, variable contains less noise.
    """

    diffusion = shape_model.diffusion
    sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod

    t = round(sigma_to_t(sigma.cpu(), diffusion))

    # note the N of sqrt_alphas_cumprod is 1024, we need to change it (0-64) to (0-1024)
    output = sqrt_alphas_cumprod[t]

    return output


def get_sqrt_one_minus_alphas_cumprod(sigma, shape_model):
    """
    # Use pre-cacalculated constant values for prior constraints
    # @it: iteration of current diffusion step; note that, larger the it, variable contains less noise.
    """

    diffusion = shape_model.diffusion
    sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod

    t = round(sigma_to_t(sigma.cpu(), diffusion))

    output = sqrt_one_minus_alphas_cumprod[t]

    return output


def calculate_s_t(sigmas, it, normalize=True):

    if normalize:
        pass

    sigmas_rev = sigmas.flip(dims=[0])

    it_keep = len(sigmas) - it - 1

    s_t = 1.0 - torch.exp(-torch.sum(sigmas_rev[: it_keep + 1]))

    return s_t
