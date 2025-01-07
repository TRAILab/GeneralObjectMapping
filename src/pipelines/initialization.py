"""

Initialize variables from first frame.

"""

import numpy as np
import torch

from src.dataset.augmentation import generate_gt_scannet_noise
from src.dataset.scannet.scannet import regularize_gt_bbox


def init_latent_from_data(shape_model, method, **kwargs):
    if method == "random":
        init_latent = shape_model.get_random_latent()
    elif method == "shap-e":
        # args: rgb_cropped
        init_latent = shape_model.get_latent_from_image(
            kwargs["cond_data"], cache=True, cache_dir=kwargs["cache_dir"]
        )  # init latent
    elif method == "text_prior":
        # use text model to generate a latent
        init_latent = shape_model.get_latent_from_text(
            kwargs["cond_data"], cache=True, cache_dir=kwargs["cache_dir"]
        )
    elif method == "zero":
        # size (1024x1024,)  device: cuda
        init_latent = torch.zeros((1024 * 1024), device="cuda")
    elif method == "random_unit":
        # random init from a unit gaussian distirbution;
        # note that if using text-condition prior optimization, use this one
        init_latent = shape_model.get_random_latent(sigma=1.0)
    else:
        raise NotImplementedError

    return init_latent


def init_coarse_pose(
    det,
    pts_3d_c,
    init_latent,
    method_init_pose="icp",
    manual_rotate="none",
    shape_model=None,
    method_init_pose_noise_level=None,
    icp_matcher=None,
    category_name=None,
):
    """
    method_init_pose = 'global'
    method_init_pose = 'gt_norm'  # directly use gt deepsdf pose; but the rotation is ambiguity
    method_init_pose = 'icp'

    @ manual_rotate: cam or world

    set shape_model if method_init_pose = 'icp'
    """
    # init pose from camera to object
    # sample points from mesh
    # pts_obj = mesh.sample(10000)

    if method_init_pose == "icp_class":
        """
        Update, a complete ICP algorithm with filtering and all, the same performance as UNC.
        """
        t_world_mesh = icp_matcher.match(det, pts_3d_c, category_name)

        if t_world_mesh is None:
            raise ValueError("Fail to Init with ICP Matcher")

        # rotate from mesh to bbox!
        # rotate around X axis for -90 deg
        rot_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        t_world_bbox = t_world_mesh.copy()
        t_world_bbox[:3, :3] = np.matmul(t_world_bbox[:3, :3], rot_mat)

        # debug : visualize the matching, before and after ICP
        # icp_matcher.visualize()

        return t_world_bbox

    elif method_init_pose == "random_basic" or method_init_pose == "icp":

        # get the translation of pts_3d_c
        pts_3d_c_mean = np.mean(pts_3d_c, axis=0)
        # get the scale of pts_3d_c
        pts_3d_c_scale = np.max(pts_3d_c, axis=0) - np.min(pts_3d_c, axis=0)

        # scale coefficient, make length 1/2
        pts_3d_c_scale = pts_3d_c_scale / 2

        # init pose from camera to object
        # so that we can transform a cuboid in range (-1,1)^3, into this object
        t_cam_obj_init = np.eye(4)
        t_cam_obj_init[:3, 3] = pts_3d_c_mean
        t_cam_obj_init[:3, :3] = np.diag(pts_3d_c_scale)

        t_co = t_cam_obj_init

        """
        A rotation fix: make the Z axis Up, as the same as the !!!Camera Frame!!!
        """
        if manual_rotate == "cam":
            # rotate around X axis for 90 deg
            rot_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        elif manual_rotate == "world":
            # rotate around X axis for -90 deg
            rot_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

            t_co[:3, :3] = np.matmul(t_co[:3, :3], rot_mat)

        if method_init_pose == "random_basic":
            return t_co

        try:
            # If using ICP, we further search 4 rotations and converge to minimum loss!
            out = icp_with_rotation_search(t_co, shape_model, init_latent, det, pts_3d_c)
        except:
            # if failed, still use origin one
            print("Fail on icp_with_rotation_search")
            out = t_co

    elif method_init_pose == "gt_scannet":

        out = det.t_world_bbox_unit_reg

    elif method_init_pose == "gt_scannet_noise":
        t_wo = det.t_world_obj  # GT Transform, ShapeNet Original Mesh to World

        """
        A param similar to the Normalization param of DeepSDF
        Stored in the Scan2CAD annotations for each model.
        """
        t_obj_box = det.t_obj_box  # Box is a normalized box

        # Goal: Add noise over GT Transformation.

        pose_goal = "t_obj_box"

        if pose_goal == "t_obj_box":

            t_world_obj_noisy = generate_gt_scannet_noise(t_wo, method_init_pose_noise_level)

            ts_world_box_noisy = t_world_obj_noisy @ t_obj_box

            ts_world_box_noisy_regu = regularize_gt_bbox(ts_world_box_noisy)

            out = ts_world_box_noisy_regu

        elif pose_goal == "final":
            ts_world_box = t_wo @ t_obj_box
            ts_world_box_regu = regularize_gt_bbox(ts_world_box)
            ts_world_box_regu_noisy = generate_gt_scannet_noise(
                ts_world_box_regu, method_init_pose_noise_level
            )

            out = ts_world_box_regu_noisy

    elif method_init_pose == "gt_scannet_noise_icp":
        raise NotImplementedError

    elif method_init_pose == "gt_scannet_scaled":
        """
        Note that NeRF space is not filling all the bounding boxes.
        Use a manual coeff to address this.
        """
        # Make the size larger x2
        out = det.t_world_bbox_unit_reg
        out[:3, :3] = out[:3, :3] * 2

    elif method_init_pose == "gt_scannet_icp":
        # First load GT Pose
        # Then, use prior shape to find rotations
        t_wo_norot = det.t_world_bbox_unit_reg

        try:
            # note pts_3d_c is world coordinate in fact
            t_wo = icp_with_rotation_search(t_wo_norot, shape_model, init_latent, det, pts_3d_c)
        except:
            # if failed, still use origin one
            print("Fail on icp_with_rotation_search")
            t_wo = t_wo_norot

        out = t_wo

    else:

        # Debug test, save mesh
        mesh = shape_model.get_shape_from_latent(init_latent)
        from utils.data_trans import trimesh_to_open3d

        mesh_o3d = trimesh_to_open3d(mesh)
        # sample points using open3d
        pts_obj = np.asarray(
            mesh_o3d.sample_points_uniformly(number_of_points=10000).points
        ).astype(np.float32)

        if "t_cam_deepsdf" in det:
            t_cam_obj_init = det.t_cam_deepsdf
        else:
            t_cam_obj_init = det.T_cam_obj

        # even smaller; with a scale of 0.5
        t_deepsdf_nerf = np.diag([0.5, 0.5, 0.5, 1])
        t_cam_obj_init = t_cam_obj_init @ t_deepsdf_nerf

        print("!!!! Cancel manual init adjustment.")

        input = {"pts_obj": pts_obj, "pts_observation": pts_3d_c, "init_t_co": t_cam_obj_init}

        """
        Config: Initialization
        """
        input["det"] = det

        t_co = init_object_pose_from_camera(
            input, method=method_init_pose
        )  # use icp method, need mesh, and observation pts_c

        out = t_co

    return out


def initialize_pose_with_icp_class(icp_matcher, pts_3d_w, category_name):
    """
    Update, a complete ICP algorithm with filtering and all, the same performance as UNC.
    """

    # Note this method will automatically unproject points from the frame,
    # it does not use the input points.
    t_world_mesh = icp_matcher.match_points(pts_3d_w, category_name)

    if t_world_mesh is None:
        raise ValueError("Fail to Init with ICP Matcher")

    rot_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    scannet_categories = ["chair", "table"]
    if category_name in scannet_categories:
        raise NotImplementedError

    else:
        # NO rotation for CO3D dataset
        rot_mat = rot_mat @ rot_mat @ rot_mat

    t_world_bbox = t_world_mesh.copy()
    t_world_bbox[:3, :3] = np.matmul(t_world_bbox[:3, :3], rot_mat)

    return t_world_bbox
