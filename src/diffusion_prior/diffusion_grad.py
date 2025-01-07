"""
Codes to calculate diffusion grads from prior model.
"""

import numpy as np
import torch

from src.diffusion_prior.basic import (
    get_sqrt_alphas_cumprod,
    get_sqrt_one_minus_alphas_cumprod,
)


def calculate_current_diffusion_iteration(
    iter_num_total, iter_num_diffusion, it_cur, diffusion_prior_valid_start_iter, method="current"
):
    """
    Get a diffusion step, which defines the noise level of current variables, depending on diffusion strategy.

    Args:
        - method:
            @current: from t0 to t1, averagely go through each timestamps
            @uniform: randomly sample from [0,T]
    """

    if diffusion_prior_valid_start_iter is None:
        diffusion_prior_valid_start_iter = 0

    if method == "uniform":
        diffusion_it = np.random.randint(0, iter_num_diffusion)
    elif method == "current":
        if it_cur < diffusion_prior_valid_start_iter:
            diffusion_it = None
        else:
            diffusion_stage_it = it_cur - diffusion_prior_valid_start_iter

            diff_opt_diff = iter_num_total - iter_num_diffusion

            diffusion_it = diff_opt_diff + diffusion_stage_it
    else:
        raise NotImplementedError

    return diffusion_it


def run_diffusion_step(latent, cond_data, sigmas, shape_model, grad_method, it, **kwargs):
    """
    Run a diffusion step, which updates the latent variables with the diffusion prior, according to the step number.

    Args:
        - latent: the latent variables to be updated. Note after the diffusion step, the latent variables would be updated.
        - grad_method:
            @ noise_plus_denoise: first add noise with current timestamp, then denoise it, and subtract this noise
    """

    # make sure cond image is valid
    assert cond_data is not None, "cond_data should be provided if use diffusion prior."

    ##########################
    # Preprocess Variables
    ##########################
    # the current latent would be x_t (with a noisy level of t) in the diffusion prior
    x_t = latent

    # get the noise schedule according to grad_method
    if grad_method == "start":
        sigma = 1.0
        next_sigma = None
    elif grad_method == "noise_plus_denoise":
        sigma = sigmas[it]
        x_0_hat = latent.clone().detach()

        p1 = get_sqrt_alphas_cumprod(sigma, shape_model)
        p2 = get_sqrt_one_minus_alphas_cumprod(sigma, shape_model)

        noise = torch.randn_like(x_t)

        x_t_bar = p1 * x_0_hat + p2 * noise

        # Change x_t scale from 1, to sigma
        x_t_curve = x_t_bar * sigma

        next_sigma = None

        x_t = x_t_curve
    else:
        sigma = sigmas[it]
        next_sigma = sigmas[it + 1]

    ##########################
    # Diffuse Stage
    ##########################

    # Use the Shap-E Diffusion Prior Model to predict the noise of the variables at the current step
    # Note the output has different meanings according to grad_method
    # image is used as condition
    diffusion_prior_grad = shape_model.get_diffusion_prior_grad(
        x_t, cond_data, sigma=sigma, grad_method=grad_method, next_sigma=next_sigma
    )

    # Update the latent variables with the diffusion prior
    shape_model.update_latent_with_diffusion_prior(
        latent, diffusion_prior_grad, grad_method, **kwargs
    )

    # For logging
    if grad_method == "noise_plus_denoise":
        grad_output = diffusion_prior_grad + noise
    else:
        grad_output = diffusion_prior_grad

    diffusion_prior_grad_num_output = torch.norm(grad_output).item()

    output = {"diffusion_prior_grad_num_output": diffusion_prior_grad_num_output}

    return output
