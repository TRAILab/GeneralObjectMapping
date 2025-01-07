"""

Functions related to pose optimization.

"""

import os
import time

import numpy as np
import torch
from PIL import Image
from shap_e.diffusion.k_diffusion import get_sigmas_karras

from src.diffusion_prior.diffusion_grad import (
    calculate_current_diffusion_iteration,
    run_diffusion_step,
)
from src.optimizer.losses import calculate_3d_loss, get_loss_regularizer
from src.optimizer.poses import (
    extract_delta_pose_variable,
    get_current_pose_from_delta_pose,
)
from src.optimizer.visualizer import (
    generate_gif,
    render_images_for_optimization_process,
    visualize_results,
)


def get_lr_scheduler(optimizer, type="exp"):
    if type == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.99
        )  # TODO: Ablade the effectiveness
    else:
        scheduler = None

    return scheduler


def manually_update_latent(latent, shape_lr_balance, sigmas, it):
    """
    Scale up the latent update from geometric constraints with a coefficient;

    When only using original update from optimizer/lr, the scale of the grad is
    too small compared with the grad of diffusion.

    --
    Scale the grad of latent, so that it matches the scale of "sigmas"
    Goal: noise (each dimension), should come from a Gaussian with sigma.
    """

    diff_sigma = sigmas[it] - sigmas[it + 1]
    shape_loss_grad = latent.grad / latent.grad.norm() * diff_sigma * shape_lr_balance

    latent.data = latent.data - shape_loss_grad


def optimize_shape_pose_with_observations_world_multiview(
    init_latent,
    init_pose_bo: np.array,
    observation_list: list,
    K: torch.Tensor,
    shape_model,
    iter_num=100,
    device="cuda",
    dtype=torch.float32,
    vis=None,
    cond_data=None,
    args=None,
    save_dir=None,
    init_latent_normalization_weight=None,
    prior_latents=None,
    t_bo_gt=None,
    loss_type_2d_render="mse",
    pts_3d_w=None,
):
    """
    Multi-view optimizations for poses and shapes under the world frame.

    Args:
        @init_pose: 4x4 mat, with scale
        @t_wc_vis: only use for visualizing in open3d.

        @ observation structure:
            {
                'pts_3d_b',
                'rgb',
                'mask',
                'depth',

                'pts_3d_sampled_b', # sampled points in empty space; and truncated areas
            }

        Coordinates:
            b: basic coordinate; can be world or camera depending on the input data.

        @ cond_data: used for diffusion prior. An image as np.array, or a text description

        @ prior_latents: if not None, use those latents to regularize the latent update. torch.Tenosr: (N, 1024x1024)

        @ loss_type_2d_render: default mse loss, or huber loss

        @ pts_3d_w: np.array, world frame points, used for 3D loss
    """

    time_start = time.time()

    """
    Initialize Config Parameters
    """
    # Config: latents and poses
    flag_optimize_latent = args.flag_optimize_latent
    flag_optimize_pose = args.flag_optimize_pose
    if not flag_optimize_pose:
        print("Close Pose Optimization.")

    # Diffusion Prior: Only valid when flag_optimize_latent is True
    use_diffusion_prior = args.diffusion_prior_state and flag_optimize_latent
    grad_method = args.grad_method  # if using step, directly replace
    if grad_method == "none":
        use_diffusion_prior = False
        print("Close Diffusion Prior.")

    # Geometric Constraints using renderings
    geometric_constraints_state = args.geometric_constraints_state
    if not geometric_constraints_state:
        print("Close Geometric Constriants. Also close pose estimation.")

    # Optimization Parameters
    prior_weight = args.prior_weight
    lr = args.lr
    vis_jump = args.vis_jump
    shape_lr_balance = args.shape_lr_balance

    w_zero_norm = args.weight_latent_zero_norm

    loss_type_2d_render = args.loss_type_2d_render

    # we use the same loss (MSE or Huber) for 3d as 2d
    loss_type_3d = loss_type_2d_render

    # Efficiency Balance: Sample points / Render rays
    points_sample_each_iter = args.geometric_constraints_3dloss_points_sample_each_iter
    render_ray_num = args.geometric_constraints_2dloss_render_ray_num

    # Loss weight
    w_2d = args.geometric_constraints_w_2d
    w_3d = args.geometric_constraints_w_3d
    w_depth = args.geometric_constraints_w_depth

    """
    Load 3D points and Poses
    """
    # pts_3d_w stores the 3D points in world frame
    if pts_3d_w is None:
        # If not provided, construct a large point cloud from all observations
        pts_3d_all_w = []
        for observation in observation_list:
            pts_3d_w = observation["pts_3d_w"]
            if isinstance(pts_3d_w, np.ndarray):
                pts_3d_w = torch.from_numpy(pts_3d_w)
            pts_3d_b = pts_3d_w.to(dtype)
            pts_3d_all_w.append(pts_3d_b)

        pts_3d_all_w_cpu = torch.cat(pts_3d_all_w, dim=0)

        pts_3d_all_w = pts_3d_all_w_cpu.to(device)  # TODO: Downsample points in 3D
    else:
        pts_3d_all_w_cpu = torch.from_numpy(pts_3d_w).to(dtype)
        pts_3d_all_w = pts_3d_all_w_cpu.to(device)

    # Load poses and move to cuda
    for observation in observation_list:
        t_cw = observation["t_cw"]
        if isinstance(t_cw, np.ndarray):
            t_cw = torch.from_numpy(t_cw)

        # Preprocess, get a cuda pose
        observation["t_cw_cuda"] = t_cw.to(dtype).to(device)

    """
    Initialize Optimization Variables for Poses and Latents
    - We optimize a delta pose w.r.t. the initial pose.
    - Require grad for optimization.
    """
    init_pose_ob = np.linalg.inv(init_pose_bo)
    t_ob_noscale, dPose_ob, dScale_log_ob = extract_delta_pose_variable(
        init_pose_ob, device=device, dtype=dtype, optimize_pose=flag_optimize_pose
    )

    # Latent
    latent = init_latent.clone().detach().requires_grad_(flag_optimize_latent).to(device)

    """
    Optimizer and Scheduler
    """
    optimizer = torch.optim.Adam([latent, dPose_ob, dScale_log_ob], lr=lr)

    lr_scheduler = get_lr_scheduler(optimizer, args.lr_scheduler_type)

    """
    Loss Logs
    """
    loss_list = []
    latent_list = []
    pose_bo_list = []

    loss_rgb_list = []
    loss_3d_list = []
    loss_depth_list = []
    loss_regularizer_list = []
    loss_zero_norm_list = []

    lr_list = []

    # detail information
    latent_update_diffusion_prior_list = []
    latent_update_loss_list = []
    dPose_ob_update_list = []
    dScale_log_ob_update_list = []

    """
    Diffusion Schedule Setup following Shap-E
    """
    iter_num_diffusion = 64
    steps, sigma_min, sigma_max, rho = (iter_num_diffusion, 0.001, 160, 7.0)
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    # preprocess priors if multi-views/modalities available
    prior_latent_distribution_mean_std = None
    if prior_latents is not None and args.loss_regularizer_method == "distribution":
        # construct a distribution
        mean = prior_latents.mean(0)
        std = prior_latents.std(0)
        prior_latent_distribution_mean_std = [mean, std]

    """
    Record the init state
    """
    t_ob_cur, t_bo_cur = get_current_pose_from_delta_pose(
        t_ob_noscale, dPose_ob, dScale_log_ob, init_pose_bo, flag_optimize_pose
    )
    latent_list.append(latent.clone().detach().cpu())
    pose_bo_list.append(t_bo_cur.clone().detach().cpu())

    """
    Start Optimization
        During iterations, we use gradients from diffusion prior to update the latent.
        And, we use gradients from geometric constraints to update both the shape and the latent.
    """
    time_optimization_start = time.time()

    # print(" > Prepare Time: {:.3f}".format(time_optimization_start - time_start))

    for it in range(iter_num):

        time_it_start = time.time()

        # Codes for the skipping config
        run_optimization_step = True
        if args.skip_optimization_after is not None and it >= args.skip_optimization_after:
            run_optimization_step = False

        #########################################################
        # A. Start Optimization using Differentiable Rendering for K steps
        #########################################################
        for opt_it in range(args.opt_num_per_diffuse):

            time_it_optx2_start = time.time()

            if run_optimization_step:

                t_ob_cur, t_bo_cur = get_current_pose_from_delta_pose(
                    t_ob_noscale, dPose_ob, dScale_log_ob, init_pose_bo, flag_optimize_pose
                )

                """
                1. 3D Losses

                Calculate 3D Loss: Use all 3D points from all frames.
                - The SDF values of the points on the surface should be zero.
                """
                if w_3d > 0:
                    loss_3d, output_3d_loss = calculate_3d_loss(
                        shape_model,
                        t_ob_cur,
                        latent,
                        pts_3d_all_w,
                        points_sample_each_iter,
                        loss_type_3d,
                    )
                else:
                    loss_3d = torch.Tensor([0])
                    output_3d_loss = None

                time_it_optx2_3d_loss = time.time()

                """
                2. 2D Losses

                Images gotten from the differentiable rendering from NeRFs; 
                - The rendered images should be similar to the observed images.
                """
                # Considering Multiview 2D Loss
                loss_rgb_frame_list = []
                loss_depth_frame_list = []
                render_ray_num_f = round(render_ray_num / len(observation_list))

                """TODO: This loop is slow, let's speed up."""
                for observation in observation_list:
                    rgb = observation["rgb"]
                    mask = observation["mask"]
                    depth = observation["depth_image"]
                    t_cw = observation["t_cw_cuda"]

                    t_cam_obj = t_cw @ t_bo_cur
                    loss_rgb_f, loss_depth_f = shape_model.get_2d_render_loss(
                        latent,
                        t_cam_obj,
                        K,
                        mask,
                        rgb,
                        depth,
                        ray_num=render_ray_num_f,
                        loss_type=loss_type_2d_render,
                    )

                    loss_rgb_frame_list.append(loss_rgb_f)
                    loss_depth_frame_list.append(loss_depth_f)

                loss_rgb = torch.mean(torch.stack(loss_rgb_frame_list))
                loss_depth = torch.mean(torch.stack(loss_depth_frame_list))

                time_it_optx2_2d_loss = time.time()

                """
                Gradient Calculation and Update for Latent and Pose
                """
                optimizer.zero_grad()

                loss = loss_rgb * w_2d + loss_depth * w_depth
                if w_3d > 0:
                    loss += loss_3d * w_3d

                if w_zero_norm > 0:
                    # regularization term for latent to be near zero
                    loss_zero_norm = latent.square().mean()
                    loss += loss_zero_norm * w_zero_norm
                else:
                    loss_zero_norm = torch.Tensor([0])

                # Prior Regularizer Term: The latent should be near the initialized latent (given by diffusion prior).
                if init_latent_normalization_weight is not None:
                    loss_regularizer = get_loss_regularizer(
                        latent,
                        init_latent,
                        prior_latents,
                        method=args.loss_regularizer_method,
                        prior_latent_distribution_mean_std=prior_latent_distribution_mean_std,
                    )

                    loss += init_latent_normalization_weight * loss_regularizer
                else:
                    loss_regularizer = torch.Tensor([0])

                time_it_optx2_regularization = time.time()

                # Backward
                loss.backward()

                time_it_optx2_backward = time.time()

                latent_update_prior = latent.clone().detach().cpu()
                dPose_ob_prior = dPose_ob.clone().detach().cpu()
                dScale_log_ob_prior = dScale_log_ob.clone().detach().cpu()

                # Update the latent and pose with flexible learning rate
                if geometric_constraints_state and args.manually_update_latent_with_scale:
                    manually_update_latent(latent, shape_lr_balance, sigmas, it)

                # Gradient Clipping
                clip_val = args.optimization_pose_clip_val
                torch.nn.utils.clip_grad_norm_(dPose_ob, clip_val)
                torch.nn.utils.clip_grad_norm_(dScale_log_ob, clip_val)

                if geometric_constraints_state:
                    optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Log the update
                latent_update = latent.clone().detach().cpu() - latent_update_prior
                dPose_ob_update = dPose_ob.clone().detach().cpu() - dPose_ob_prior
                dScale_log_ob_update = dScale_log_ob.clone().detach().cpu() - dScale_log_ob_prior

                time_it_optx2_post_process = time.time()

                """Report the time"""
                # print(" >>> x2 Time 3d Loss: {:.3f}".format(time_it_optx2_3d_loss - time_it_optx2_start))
                # print(" >>> x2 Time 2d Loss: {:.3f}".format(time_it_optx2_2d_loss - time_it_optx2_3d_loss))
                # print(" >>> x2 Time Regularization: {:.3f}".format(time_it_optx2_regularization - time_it_optx2_2d_loss))
                # print(" >>> x2 Time Backward: {:.3f}".format(time_it_optx2_backward - time_it_optx2_regularization))
                # print(" >>> x2 Time Post Process: {:.3f}".format(time_it_optx2_post_process - time_it_optx2_backward))

            ###################################################
            # record data
            latent_update_loss_list.append(latent_update.norm())
            dPose_ob_update_list.append(dPose_ob_update.norm())
            dScale_log_ob_update_list.append(dScale_log_ob_update.norm())

            loss_list.append(loss.item())

            latent_list.append(latent.clone().detach().cpu())
            pose_bo_list.append(t_bo_cur.clone().detach().cpu())

            loss_rgb_list.append(loss_rgb.item())
            loss_3d_list.append(loss_3d.item())
            loss_depth_list.append(loss_depth.item())
            loss_regularizer_list.append(loss_regularizer.item())
            loss_zero_norm_list.append(loss_zero_norm.item())

            lr_list.append(optimizer.param_groups[0]["lr"])

        time_it_opt_end = time.time()

        #########################################################
        # B. Start Diffusion for K steps
        #########################################################

        latent_update_prior = latent.clone().detach().cpu()

        diffusion_prior_grad_num_output = 0

        cond_after_K_steps = it >= args.diffusion_prior_valid_start_iter
        if use_diffusion_prior and cond_after_K_steps:
            # Get a diffusion step according to the diffusion scheduler and current iteration
            it_diffuse = calculate_current_diffusion_iteration(
                iter_num,
                iter_num_diffusion,
                it,
                args.diffusion_prior_valid_start_iter,
                method="uniform",
            )

            time_it_diffuse_calculation_start = time.time()

            dif_step_output = run_diffusion_step(
                latent,
                cond_data,
                sigmas,
                shape_model,
                grad_method,
                it_diffuse,
                prior_weight=prior_weight,
                lr=lr,  # only useful when grad_method==start
                prior_fusion_weight=args.prior_fusion_weight,  # only useful when grad_method==noise_plus_diffuse
            )

            diffusion_prior_grad_num_output = dif_step_output["diffusion_prior_grad_num_output"]

            time_it_diffuse_run_diffusion_end = time.time()

        # save latent update brought by diffusion prior
        latent_update = latent.clone().detach().cpu() - latent_update_prior
        # save to list
        latent_update_diffusion_prior_list.append(latent_update.norm())

        #########################################
        # Print Iteration Information

        if it % 10 == 0:
            print(
                "Iter: {}/{}".format(it, iter_num),
                # 'Opt_iter:', opt_it,
                "Loss: {:.3f}".format(loss.item()),
                "Loss RGB: {:.3f}".format(loss_rgb.item()),
                "Loss Depth: {:.3f}".format(loss_depth.item()),
                "Loss 3D: {:.3f}".format(loss_3d.item()),
                # 'GT Loss 3D: {:.3f}'.format(gt_loss_3d.item()) if gt_pts_3d_b is not None else 'None',
                # 'Loss 3D Away:', loss_3d_away.item(),
                "Loss Norm: {:.3f}".format(loss_zero_norm.item()) if w_zero_norm > 0 else "NoNorm",
                "Loss Regul: {:.5f}".format(loss_regularizer.item()),
                "lr: {:.3f}".format(optimizer.param_groups[0]["lr"]),
                " | ",
                "DifPrior Grad Norm: {:.3f}".format(diffusion_prior_grad_num_output),
                "Latent Norm: {:.3f}".format(torch.norm(latent).item()),
                "Pose Norm: {:.3f}".format(torch.norm(dPose_ob).item()),
                "Scale Norm: {:.3f}".format(torch.norm(dScale_log_ob).item()),
            )

            # Print Computation Time:
            # print(
            #     " > Opt Time: {:.3f}".format(time_it_opt_end - time_it_start),
            #     " > Time calculate_current_diffusion_iteration: {:.3f}".format(time_it_diffuse_calculation_start - time_it_opt_end),
            #     " > Time run_diffusion_step: {:.3f}".format(time_it_diffuse_run_diffusion_end - time_it_diffuse_calculation_start),
            # )

    time_optimization_end = time.time()
    print(" - Optimization Time:", time_optimization_end - time_optimization_start)

    history = {
        "loss": loss_list,
        "loss_rgb": loss_rgb_list,
        "loss_depth": loss_depth_list,
        "loss_3d": loss_3d_list,
        "loss_regularizer": loss_regularizer_list,
        "loss_zero_norm": loss_zero_norm_list,
        "latent": latent_list,
        "pose_bo": pose_bo_list,
        "lr": lr_list,
        "latent_update": {
            "diffusion_prior": latent_update_diffusion_prior_list,
            "loss": latent_update_loss_list,
        },
        "pose_update": {
            "pose": dPose_ob_update_list,
            "scale": dScale_log_ob_update_list,
        },
    }

    """
    Visualize the Loss Plots
    """
    weights = {
        "w_2d": w_2d,
        "w_3d": w_3d,
        "w_depth": w_depth,
        "w_regularizer": init_latent_normalization_weight,
        "w_zero_norm": w_zero_norm,
    }
    save_dir_plot = os.path.join(save_dir, "plot")
    os.makedirs(save_dir_plot, exist_ok=True)
    visualize_results(history, weights, save_dir=save_dir_plot)  # visualize pyplot for losses

    """
    Visualization: Render RGB/Depth Image and Compare with Observations
    """
    if args.visualize_rendered_image:
        print("Begin rendering images for visualization...")

        # Render images to each observation
        skip = 1
        mid_ob_list = range(0, len(observation_list), skip)

        render_resolution_scale = 2

        os.makedirs(save_dir, exist_ok=True)

        images_list = []  # Store a list of PIL images
        for ob_order in mid_ob_list:
            # Render to the first view
            t_cw = observation_list[ob_order][
                "t_cw_cuda"
            ]  # Temperally treat basic as camera frame.
            t_cam_obj = t_cw @ t_bo_cur

            im_render = shape_model.render_image(
                latent,
                t_cam_obj,
                K,
                resize_scale=8.0 / render_resolution_scale,
                background=torch.Tensor([255, 255, 255]),
            )

            images_list.append(im_render[0])

        # concatenate images in the list into a single image in a row, and save; note that the image is in PIL format
        save_im_name = os.path.join(save_dir, "render_mid.png")
        for i in range(len(images_list)):
            im_np = np.array(images_list[i])
            if i == 0:
                im_concat = im_np
            else:
                im_concat = np.concatenate((im_concat, im_np), axis=1)

        # save the concatenated image
        im_concat = Image.fromarray(im_concat)
        im_concat.save(save_im_name)

        print(f"Save rendered images to {save_im_name}")

        # Render 360 deg gif
        print("Rendering 360 deg gif ...")
        render_resolution = 64 * render_resolution_scale
        save_dir_render_im = os.path.join(save_dir, "render_360deg.gif")
        os.makedirs(os.path.dirname(save_dir_render_im), exist_ok=True)
        shape_model.render_images_for_vis(
            latent, render_resolution, save_dir_render_im, background=torch.Tensor([255, 255, 255])
        )

    """
    Outputs
    """
    t_ob_cur, t_bo_cur = get_current_pose_from_delta_pose(
        t_ob_noscale, dPose_ob, dScale_log_ob, init_pose_bo, flag_optimize_pose
    )
    output = {
        "latent": latent.detach().cpu(),
        "pose_bo": t_bo_cur.detach().cpu(),
        "history": history,
        "observations": {
            "pts_3d_b": pts_3d_all_w_cpu  # TODO: Currently it's the last frame, even not the obs frame.
        },
    }

    return output
