"""

Codes related to solve an optimization problem.

"""

import os

import cv2
import numpy as np
import torch

from src.dataset.loader import (
    construct_observation_from_dataset,
    filter_valid_observations_for_priors,
    load_observations,
)
from src.optimizer.pose_solver import (
    optimize_shape_pose_with_observations_world_multiview,
)
from src.optimizer.visualizer import save_meshes_in_mid_iterations
from src.pipelines.initialization import (
    init_coarse_pose,
    init_latent_from_data,
    initialize_pose_with_icp_class,
)
from src.utils.geometry import aggregate_multi_view_points


def process_one_instance_from_all_frames(
    obs_id_list,
    recon_save_dir_prefix_frame,
    dataset_subset,
    shape_model,
    args,
    vis=None,
    icp_matcher=None,
):
    """
    Each Instance Contains Multiple Frames Observations

    Process all frames together to estimate the pose & shape for the instance
    """

    """
    Initialize structures of all the observation frames
        - Contain RGB, Depth, Mask, Cropped images
    """
    observation_images_list = []
    for obs_id in obs_id_list:
        observations = load_observations(obs_id, dataset_subset, mask_source=args.mask_source)
        observation_images_list.append(observations)

    # Filter valid observations for priors
    valid_prior_condition_images, valid_observations = filter_valid_observations_for_priors(
        observation_images_list
    )

    """
    Initialize Latents
    """
    latent_init_method = "center_valid"
    if latent_init_method == "center_valid":
        """Choose the center image of the VALID observations to initialize the latent"""
        init_frame_order = round(len(valid_prior_condition_images) / 2)
        rgb_cropped = valid_prior_condition_images[init_frame_order]
        ob_init = valid_observations[init_frame_order]
    elif latent_init_method == "center":
        """Choose the center image of all observations to initialize the latent"""
        init_frame_order = round(len(obs_id_list) / 2)
        observations = observation_images_list[init_frame_order]
        rgb_cropped = observations["rgb_mask_cropped"]
        ob_init = observations
    else:
        raise NotImplementedError

    """
    Initialize the condition for the Shap-E prior models
    """

    # Get latents for each input images, which will be used during optimization as constraints
    prior_latent_list = []
    prior_latents = None

    category_name = dataset_subset.get_category_name()
    if args.diffusion_condition_type == "image":
        cond_data = rgb_cropped
    elif args.diffusion_condition_type == "multi-views":
        # use all images after valid conditioning filtering
        cond_data = valid_prior_condition_images

        # generate latents for each cond_data, and put into prior_latent_list
        for cond_id, im in enumerate(cond_data):
            latent = shape_model.get_latent_from_image(
                im, cache=True, cache_dir=args.cache_dir
            )  # init latent
            prior_latent_list.append(latent)

            # if open debug mode, reconstruct shapes and save
            if args.debug:
                debug_recon_mesh_dir = os.path.join(recon_save_dir_prefix_frame, "debug")
                os.makedirs(debug_recon_mesh_dir, exist_ok=True)

                mesh = shape_model.get_shape_from_latent(latent)
                shape_model.save_shape(
                    mesh,
                    os.path.join(debug_recon_mesh_dir, "prior_latent_mesh_{}.ply".format(cond_id)),
                )

                cv2.imwrite(
                    os.path.join(debug_recon_mesh_dir, "prior_latent_im_{}.png".format(cond_id)),
                    im[:, :, ::-1],  # RGB to BGR
                )

        print("Init prior_latent_list:", len(prior_latent_list))

        # stack the list into a tensor
        prior_latents = torch.stack(prior_latent_list, dim=0).to("cuda")

        # if open debug mode, fuse latents and save mesh
        if args.debug:
            latent_fused = prior_latents.mean(dim=0)
            mesh = shape_model.get_shape_from_latent(latent_fused)
            shape_model.save_shape(
                mesh, os.path.join(debug_recon_mesh_dir, "fused_latent_mesh.ply")
            )

    else:
        # use text description as condition
        if args.text_condition_given is not None:
            # A debug mode, use a fixed text description by --text_condition_given
            text_descript = args.text_condition_given
            print("Use a fixed text description:", text_descript)
        else:
            text_descript = "a " + category_name
            print("Use a default text description:", text_descript)

        # load a text description
        cond_data = text_descript

        print("Condition text: ", cond_data)

    """Get an initial latent from the condition data using the shape model"""
    init_latent = init_latent_from_data(
        shape_model, args.init_latent_method, cond_data=cond_data, cache_dir=args.cache_dir
    )

    """
    Init Pose
    """
    obs_id = ob_init["obs_id"]
    det_list = dataset_subset.get_frame_by_id(obs_id)
    det = det_list[0]
    pts_3d_w = det.surface_points_world
    t_wo = init_coarse_pose(
        det,
        pts_3d_w,
        init_latent,
        method_init_pose=args.method_init_pose,
        manual_rotate="world",
        shape_model=shape_model,
        method_init_pose_noise_level=args.method_init_pose_noise_level,
        icp_matcher=icp_matcher,
        category_name=category_name,
    )
    # For visualization only
    t_wo_gt = det.t_world_bbox_unit_reg

    """
    Construct observation structure for all frame datas
        - Images: RGB, Depth, Mask, Cropped
        - 3D Points
        - Poses
    """
    observation_frames_list = []
    for obs_id in obs_id_list:
        det_list = dataset_subset.get_frame_by_id(obs_id)
        det = det_list[0]  # assume there is only one object in each frame

        observation = construct_observation_from_dataset(
            det, dataset_subset, args.debug_use_gt_points
        )
        observation_frames_list.append(observation)

    """
    Initialize Parameters
    """
    # latent weight, if latent is from prior, given a weight to constrain during optimization
    if args.init_latent_method == "shap-e":
        init_latent_normalization_weight = args.init_latent_normalization_weight
    else:
        init_latent_normalization_weight = None

    """
    Begin Optimization
    """
    output = optimize_shape_pose_with_observations_world_multiview(
        init_latent,
        t_wo,
        observation_frames_list,
        det.K,
        shape_model,
        vis=vis,
        iter_num=args.iter_num,
        cond_data=cond_data,
        args=args,
        save_dir=recon_save_dir_prefix_frame,
        init_latent_normalization_weight=init_latent_normalization_weight,
        prior_latents=prior_latents,
        t_bo_gt=t_wo_gt,
    )

    """
    Save the Meshes from Optimization Output
    """
    # save_dir_frame_instance = os.path.join(recon_save_dir_prefix_frame, f"frame-{obs_id}")
    # save_meshes_in_mid_iterations(output, shape_model, save_dir_frame_instance)

    return output


def process_one_instance_from_all_frames_co3d(
    frames, shape_model, args, vis=None, icp_matcher=None, recon_save_dir="./output"
):
    """
    Each Instance Contains Multiple Frames Observations

    Process all frames together to estimate the pose & shape for the instance
    """

    """
    Generate data from all observed frames
    """
    observation_list = frames

    """
    Begin Optimization
    """

    """
    Init Latents
    """
    valid_prior_condition_images, valid_observations = filter_valid_observations_for_priors(
        observation_list
    )

    latent_init_method = "center_valid"
    if latent_init_method == "center_valid":
        # Update: Before init latnets, filter observations for occluded/partial ones
        init_frame_order = round(len(valid_prior_condition_images) / 2)
        rgb_cropped = valid_prior_condition_images[init_frame_order]
        ob_init = valid_observations[init_frame_order]

    """
    Choose Cond data
    """
    prior_latents = None

    category_name = frames[0]["category"]
    if args.diffusion_condition_type == "image":
        cond_data = rgb_cropped
    elif args.diffusion_condition_type == "multi-views":
        raise NotImplementedError
    else:
        text_descript = "a " + category_name

        # load a text description
        cond_data = text_descript

        print("condition text: ", cond_data)

    # TODO: if debug, save the cropped images and reconstructed shapes
    init_latent = init_latent_from_data(
        shape_model, args.init_latent_method, cond_data=cond_data, cache_dir=args.cache_dir
    )

    # latent weight, if latent is from prior, given a weight to constrain during optimization
    if args.init_latent_method == "shap-e":
        init_latent_normalization_weight = args.init_latent_normalization_weight
    else:
        init_latent_normalization_weight = None

    """
    Init Pose
    """
    # obs_id = ob_init['obs_id']
    # det_list = dataset_subset.get_frame_by_id(obs_id)
    # det = det_list[0] # contain all information you need
    # TODO: use all points from all views to initilaize; or at least use filtered observations
    pts_3d_w_aggregate = aggregate_multi_view_points(valid_observations)
    # pts_3d_w_aggregate = ob_init['pts_3d_w']

    pts_3d_w = pts_3d_w_aggregate

    # An accurate init pose is VERY important for the final performance, since we are
    # an optimization-based system.
    # We implement a simple way to get initialized pose by matching points in world to
    # a template mesh.
    t_wo = initialize_pose_with_icp_class(icp_matcher, pts_3d_w, category_name)
    # t_co = np.linalg.inv(det.T_world_cam) @ t_wo

    # For visualization only
    # t_wo_gt = det.t_world_bbox_unit_reg
    t_wo_gt = None

    """
    3D Sainity check: visualize all shapes, coordinates [DONE]
    """
    # sainity_check_3d()

    """
    Begin Optimization
    """
    K = ob_init["K"]
    output = optimize_shape_pose_with_observations_world_multiview(
        init_latent,
        t_wo,
        observation_list,
        K,
        shape_model,
        vis=vis,
        iter_num=args.iter_num,
        cond_data=cond_data,
        args=args,
        save_dir=recon_save_dir,
        init_latent_normalization_weight=init_latent_normalization_weight,
        prior_latents=prior_latents,
        t_bo_gt=t_wo_gt,
        pts_3d_w=pts_3d_w_aggregate,
    )

    # Process each frame

    # visualize output
    # save_dir_frame_instance = os.path.join(recon_save_dir,
    #                                         f'frame-{obs_id}')
    # save_meshes_in_mid_iterations(output, shape_model, recon_save_dir)

    # all frames are processed
    return output
