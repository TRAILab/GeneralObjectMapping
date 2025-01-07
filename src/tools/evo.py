"""

Evo File:

Evaluate the reconstruction results in a quantitive way.

"""

import os

import numpy as np
import quaternion
import torch

from src.dataset.loader import construct_observation_from_dataset
from src.dataset.scannet.scannet import inverse_regularize_gt_bbox

# optimizer/pose_solver.py
from src.optimizer.pose_solver import calculate_3d_loss
from src.utils import SE3
from src.utils.data_trans import trimesh_to_open3d
from src.visualization.visualizer import visualize_all_frames


def get_2d_loss(latent, t_bo_cur, shape_model, obs_id_list, dataset_subset, args, save_dir=None):
    # Optional: visualize input images
    b_visualize_images = save_dir is not None
    if b_visualize_images:
        visualize_all_frames(
            None,
            obs_id_list,
            save_dir,
            dataset_subset.obj_id,
            dataset_subset.scene_name,
            dataset_subset,
            args=args,
        )

    ##########################
    # Get 2D Loss
    # Project current latent into each camera views, and compare with the observations!
    # Considering each views
    observation_list = []
    for obs_id in obs_id_list:
        det_list = dataset_subset.get_frame_by_id(obs_id)
        det = det_list[0]

        observation = construct_observation_from_dataset(det, dataset_subset, False)

        observation_list.append(observation)

    """
    Codes for 2D rendering; Comment it for speeding up
    """
    # render_ray_num = render_ray_num_evo

    # Considering Multiview 2D Loss
    loss_rgb_frame_list = []
    loss_depth_frame_list = []
    psnr_frame_list = []

    # render_ray_num_f = round(render_ray_num / len(observation_list))
    # render_ray_num_f = render_ray_num  # each frame is the same.

    for observation in observation_list:
        rgb = observation["rgb"]
        mask = observation["mask"]
        depth = observation["depth_image"]
        t_cw = torch.from_numpy(
            observation["t_cw"]
        ).cuda()  # Temperally treat basic as camera frame.

        t_cam_obj = t_cw @ t_bo_cur.cuda()
        loss_rgb_f, loss_depth_f, psnr_f = shape_model.get_2d_render_loss(
            latent,
            t_cam_obj,
            det.K,
            mask,
            rgb,
            depth,
            # ray_num=render_ray_num_f)
            dense=True,
            open_psnr=True,
        )  # Dense for evaluation!

        loss_rgb_frame_list.append(loss_rgb_f.detach().cpu())
        loss_depth_frame_list.append(loss_depth_f.detach().cpu())
        psnr_frame_list.append(psnr_f.detach().cpu())

    # average to final loss 2d
    loss_rgb = torch.mean(torch.stack(loss_rgb_frame_list))
    loss_depth = torch.mean(torch.stack(loss_depth_frame_list))
    psnr = torch.mean(torch.stack(psnr_frame_list))

    b_visualize_rendered_image = save_dir is not None
    if b_visualize_rendered_image:
        # optional: visualize rendered images to the frames
        # Render images to each observation!
        mid_ob_list = range(len(observation_list))
        # mid_ob = round(len(observation_list) / 2)

        os.makedirs(save_dir, exist_ok=True)

        images_list = []  # Store a list of PIL images
        for ob_order in mid_ob_list:
            # Render to the first view
            t_cw = observation_list[ob_order]["t_cw"]
            t_cw = torch.from_numpy(t_cw).cuda()
            t_cam_obj = t_cw @ t_bo_cur.cuda()

            im_render = shape_model.render_image(latent, t_cam_obj, det.K, resize_scale=4.0)

            # save_dir_gif = os.path.join(save_dir, f"render_mid_{ob_order}.png")

            # im_render[0].save(save_dir_gif)
            images_list.append(im_render[0])

        # concatenate images in the list into a single image in a row, and save; note that the image is in PIL format
        images_list[0].save(save_dir, save_all=True, append_images=images_list[1:])
        print(f"Save rendered images to {save_dir}")

    return loss_rgb, loss_depth, psnr, loss_rgb_frame_list, loss_depth_frame_list, psnr_frame_list


def get_test_2d_loss(
    latent, t_bo_cur, shape_model, dataset_subset, obs_id_list, args, save_dir=None
):
    """
    Test 2d losses on test images
    """
    # Sample Test Images, that are not inside obs_id_list
    num_total_frames = len(dataset_subset)

    # note that obs_id_list is averagely sampled from the sequence
    # we add an offset to it to generate unseen images

    # first calculate the distance between each obs
    obs_dis = num_total_frames // len(obs_id_list)

    # add the offset;
    # remove the last frame since it is the same as the first one
    obs_id_list_offset = np.array(obs_id_list[:-1]) + round(obs_dis / 2)
    # use % to make sure it is in the range
    obs_id_list_offset = obs_id_list_offset % num_total_frames

    # Get loss
    return get_2d_loss(
        latent, t_bo_cur, shape_model, obs_id_list_offset, dataset_subset, args, save_dir
    )


def chamfer_distance(
    point_cloud1: torch.Tensor, point_cloud2: torch.Tensor, max_batch_size: int = None
) -> torch.Tensor:
    """
    Calculate the Chamfer Distance between two point clouds.
    :param point_cloud1: Point cloud 1 with shape (batch_size, npoint, 3)
    :param point_cloud2: Point cloud 2 with shape (batch_size, npoint, 3)
    :param max_batch_size: Maximum batch size to process at once to avoid running out of GPU memory
    :return: Chamfer distance with shape (batch_size,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(point_cloud1, np.ndarray):
        point_cloud1 = torch.from_numpy(point_cloud1)

    if isinstance(point_cloud2, np.ndarray):
        point_cloud2 = torch.from_numpy(point_cloud2)

    if max_batch_size is None:
        max_batch_size = point_cloud1.shape[0]

    distances = []

    for i in range(0, point_cloud1.shape[0], max_batch_size):
        pc1 = point_cloud1[i : i + max_batch_size].to(device)
        pc2 = point_cloud2[i : i + max_batch_size].to(device)

        dist_matrix = torch.cdist(pc1, pc2)

        min_dist_12 = dist_matrix.min(dim=2)[0].mean(dim=1)
        min_dist_21 = dist_matrix.min(dim=1)[0].mean(dim=1)

        distance = min_dist_12 + min_dist_21
        distances.extend(distance.tolist())

    return distances


def points_to_o3d(points: torch.Tensor):
    import open3d as o3d

    points_np = points.cpu().numpy()  # Convert tensor to numpy array
    pcd = o3d.geometry.PointCloud()  # Create a PointCloud object
    pcd.points = o3d.utility.Vector3dVector(points_np)  # Assign points
    return pcd


def find_best_pose_with_rot_search(
    gt_points_w: torch.Tensor, pts_o: torch.Tensor, t_wo, with_scaling=False
):
    """
    Search four rotations around Z axis of t_bo_cur

    calculate CD

    return minimum one, and the corresponding minimum pose.
    """
    # First, do not consider scaling

    # Construct Points to O3d
    pts_obj_o3d = points_to_o3d(pts_o)
    pts_3d_c_o3d = points_to_o3d(gt_points_w)
    init_t_co = t_wo

    from optimizer.initializer import search_rotation_with_icp

    best_fitness, best_trans_icp = search_rotation_with_icp(
        pts_obj_o3d,
        pts_3d_c_o3d,
        init_t_co,
        rotation_search_method="four",
        with_scaling=with_scaling,
    )

    return best_trans_icp


def to_str_safe(val):
    # if val is tensor (maybe on gpu), only extract value
    if torch.is_tensor(val):
        val = val.detach().cpu().numpy()

    # val: string or value
    try:
        valid_str = str(val)
    except:
        valid_str = "None"
    return valid_str


def split_evo_list_into_categories(evo_list, get_category=None):
    """

    Output: a dict from category to list of evo

    @get_category: a function
    """

    dict_category_to_list = {}

    if evo_list is None:
        return dict_category_to_list

    for evo in evo_list:

        # Content of evo
        # scene0207_01_ins_order_11
        name = evo["name"]

        # extract from this structure
        scene_name, ins_order_id = name.split("_ins_order_")
        ins_order_id = int(ins_order_id)

        # Access its category with name and id
        if get_category is not None:
            category = get_category(scene_name, ins_order_id)
        else:
            raise ValueError

        if category in dict_category_to_list:
            dict_category_to_list[category].append(evo)
        else:
            dict_category_to_list[category] = [evo]

    return dict_category_to_list


def print_evo_list_with_categories(evo_list, get_category=None):
    """
    category-aware version

    @get_category: a function to get category
    """

    # Step 1: Filter evo_list into category-level list
    dict_category_to_list = split_evo_list_into_categories(evo_list, get_category)

    # Run print_evo_list for each category
    dict_category_to_result = {}
    for cat in dict_category_to_list:
        cat_evo_list = dict_category_to_list[cat]
        average_str, title = print_evo_list(cat_evo_list, silence=True)

        dict_category_to_result[cat] = [average_str, title]

    # Summarize result and output
    return dict_category_to_result


def print_evo_list(evo_list, silence=False):
    """
    Extract important information from evo_list and output

    Update: ignore and record the invalid case.
    """

    if evo_list is None or len(evo_list) == 0:
        return "", ""

    # Step 1: Construct a key dict with values
    open_iou = "iou" in evo_list[0]

    # Step 2: Output the string, and check if the key-value is valid, or output None

    # Print as a table into one line
    # First row: title
    # Second row: Value
    # print('Name,loss_3d_ob,loss_3d_gt,cd,loss_2d_rgb_test,loss_2d_depth_test,loss_2d_psnr_test,loss_2d_rgb_train,loss_2d_depth_train,loss_2d_psnr_train,N_objects')

    if open_iou:
        title = "Name,iou,pose_trans,pose_rot,pose_scale,cd,cd_best,loss_3d_gt,loss_3d_ob,loss_2d_rgb_train,loss_2d_depth_train,loss_2d_psnr_train,N_objects"
    else:
        title = "Name,pose_trans,pose_rot,pose_scale,cd,cd_best,loss_3d_gt,loss_3d_ob,loss_2d_rgb_train,loss_2d_depth_train,loss_2d_psnr_train,N_objects"

    if not silence:
        print(title)

    # First, print every data
    valid_mask = []
    for evo in evo_list:
        str_list = []

        str_list.append(to_str_safe(evo["name"]))
        if "loss_3d_ob" in evo:
            # valid one

            # Name,pose_trans,pose_rot,pose_scale,cd,cd_best,loss_3d_gt,loss_3d_ob,loss_2d_rgb_train,loss_2d_depth_train,loss_2d_psnr_train,N_objects

            if open_iou:
                str_list.append(to_str_safe(evo["iou"]))

            # add pose error
            str_list.append(to_str_safe(evo["pose_error"]["translation"]))
            str_list.append(to_str_safe(evo["pose_error"]["rotation"]))
            str_list.append(to_str_safe(evo["pose_error"]["scale"]))

            str_list.append(to_str_safe(evo["cd"]))
            str_list.append(to_str_safe(evo["cd_best"]))

            # generate str_list, change value into str, change to None if not exist or fail
            str_list.append(to_str_safe(evo["loss_3d_gt"]))
            str_list.append(to_str_safe(evo["loss_3d_ob"]))

            # str_list.append(to_str_safe(evo['loss_2d_test']['rgb']))
            # str_list.append(to_str_safe(evo['loss_2d_test']['depth']))
            # str_list.append(to_str_safe(evo['loss_2d_test']['psnr']))

            str_list.append(to_str_safe(evo["loss_2d_train"]["rgb"]))
            str_list.append(to_str_safe(evo["loss_2d_train"]["depth"]))
            str_list.append(to_str_safe(evo["loss_2d_train"]["psnr"]))

            str_list.append(to_str_safe(1))
        else:
            str_list.append("None")

        # final output
        if not silence:
            print(",".join(str_list))

        valid = not ("None" in str_list)
        valid_mask.append(valid)
    valid_mask = np.array(valid_mask)

    # only use those valid, with valid_mask=True
    evo_list_valid = np.array(evo_list)[valid_mask]

    # check if >0
    if len(evo_list_valid) == 0:
        print("No valid data in evo_list")
        return "", ""

    if open_iou:
        iou_average = np.stack([evo["iou"] for evo in evo_list_valid]).mean()

    loss_3d_ob_average = torch.stack([evo["loss_3d_ob"] for evo in evo_list_valid]).mean()
    loss_3d_gt_average = torch.stack([evo["loss_3d_gt"] for evo in evo_list_valid]).mean()
    cd_average = np.stack([evo["cd"] for evo in evo_list_valid]).mean()
    cd_best_average = np.stack([evo["cd_best"] for evo in evo_list_valid]).mean()

    # loss_2d_rgb_average_test = torch.stack([evo['loss_2d_test']['rgb'] for evo in evo_list_valid]).mean()
    # loss_2d_depth_average_test = torch.stack([evo['loss_2d_test']['depth'] for evo in evo_list_valid]).mean()
    # loss_2d_psnr_average_test = torch.stack([evo['loss_2d_test']['psnr'] for evo in evo_list_valid]).mean()

    # train
    loss_2d_rgb_average_train = torch.stack(
        [evo["loss_2d_train"]["rgb"] for evo in evo_list_valid]
    ).mean()
    loss_2d_depth_average_train = torch.stack(
        [evo["loss_2d_train"]["depth"] for evo in evo_list_valid]
    ).mean()
    loss_2d_psnr_average_train = torch.stack(
        [evo["loss_2d_train"]["psnr"] for evo in evo_list_valid]
    ).mean()

    # pose
    error_translation = np.stack(
        [evo["pose_error"]["translation"] for evo in evo_list_valid]
    ).mean()
    error_rotation = np.stack([evo["pose_error"]["rotation"] for evo in evo_list_valid]).mean()
    error_scale = np.stack([evo["pose_error"]["scale"] for evo in evo_list_valid]).mean()

    # Output
    str_list = []

    str_list.append("Average")

    # Name,pose_trans,pose_rot,pose_scale,cd,cd_best,loss_3d_gt,loss_3d_ob,loss_2d_rgb_train,loss_2d_depth_train,loss_2d_psnr_train,N_objects

    if open_iou:
        str_list.append(to_str_safe(iou_average))

    # generate str_list, change value into str, change to None if not exist or fail
    str_list.append(to_str_safe(error_translation))
    str_list.append(to_str_safe(error_rotation))
    str_list.append(to_str_safe(error_scale))

    str_list.append(to_str_safe(cd_average))
    str_list.append(to_str_safe(cd_best_average))

    str_list.append(to_str_safe(loss_3d_gt_average))
    str_list.append(to_str_safe(loss_3d_ob_average))

    str_list.append(to_str_safe(loss_2d_rgb_average_train))
    str_list.append(to_str_safe(loss_2d_depth_average_train))
    str_list.append(to_str_safe(loss_2d_psnr_average_train))

    str_list.append(to_str_safe(len(evo_list_valid)))

    def calculate_percentage(data, thresholds, inverse=False):
        percentages = []
        for threshold in thresholds:

            if inverse:
                # larger, better
                count = sum(1 for item in data if item > threshold)
            else:
                count = sum(1 for item in data if item < threshold)
            percentage = (count / len(data)) * 100
            percentages.append(percentage)
        return percentages

    def calculate_percentage_pose(data, thresholds):
        percentages = []
        for threshold in thresholds:

            # smaller better
            count = sum(1 for item in data if all(i < j for i, j in zip(item, threshold)))

            percentage = (count / len(data)) * 100
            percentages.append(percentage)
        return percentages

    # Further Count Valid Percentage: CD<Threshold; Trans,Rot,Scale < Threshold
    cd_thresh_list = [0.05, 0.1, 0.15, 0.2]
    pose_thresh_list = [[0.1, 10, 10], [0.2, 20, 20], [0.3, 30, 30]]  # m, deg, %

    cd_data = [evo["cd"] for evo in evo_list_valid]
    pose_data = [
        [
            evo["pose_error"]["translation"],
            evo["pose_error"]["rotation"],
            evo["pose_error"]["scale"],
        ]
        for evo in evo_list_valid
    ]

    cd_percentages = calculate_percentage(cd_data, cd_thresh_list)
    pose_percentages = calculate_percentage_pose(pose_data, pose_thresh_list)

    # Output
    for p in cd_percentages:
        str_list.append(to_str_safe(p))
    for p in pose_percentages:
        str_list.append(to_str_safe(p))

    if open_iou:
        iou_thresh_list = [0.25, 0.5, 0.75]
        iou_data = [evo["iou"] for evo in evo_list_valid]
        iou_percentages = calculate_percentage(iou_data, iou_thresh_list, inverse=True)
        for p in iou_percentages:
            str_list.append(to_str_safe(p))
    ################################

    # final output
    average_str = ",".join(str_list)

    if not silence:
        print(average_str)

    # add new threshold count into title, e.g., cd<k, pose<k-k-k
    for cd_th in cd_thresh_list:
        item_name = f",cd<{cd_th}"
        title += item_name
    for p_th in pose_thresh_list:
        item_name = f",pose<{p_th[0]}-{p_th[1]}-{p_th[2]}"
        title += item_name

    if open_iou:
        for iou_th in iou_thresh_list:
            item_name = f",iou>{iou_th}"
            title += item_name

    return average_str, title


@torch.no_grad()
def evaluate_output(
    latent,
    pose_bo,
    shape_model,
    obs_id_list,
    dataset_subset,
    pts_3d_b=None,
    args=None,
    save_dir=None,
    open_test_evo=False,
):
    """

    Evaluate 3D metrics
        Points distance to GT Mesh
        Points distance to Observation, if pts_3d_b

    Evaluate 2D metrics:
        Projected RGB Images
        Projected Depth Images

    Update: Evaluate Pose Metrics.

    """
    t_bo_cur = pose_bo
    t_ob_cur = t_bo_cur.inverse()

    latent = latent.cuda()

    """
    Calculate 3D Loss: Use all points from all frames. Randomly sample.
    """
    pts_sample_evo = None
    loss_3d_ob, output_3d_loss = calculate_3d_loss(
        shape_model, t_ob_cur, latent, pts_3d_b, points_sample_each_iter=pts_sample_evo
    )

    # Sample GT Mesh Points in b (world) coordinate
    try:
        gt_points_w = dataset_subset.get_gt_sampled_points_in_world(10000)
        loss_3d_gt, output_3d_loss = calculate_3d_loss(
            shape_model, t_ob_cur, latent, gt_points_w, points_sample_each_iter=pts_sample_evo
        )

        # Update: Add chamfer distance evaluation
        # Sample points from latent shape
        shape = shape_model.get_shape_from_latent(latent)  # TriMesh
        # sample points
        shape_o3d = trimesh_to_open3d(shape)  # this is the area of box_reg
        # sample points,
        n_sample_pts = 10000
        pts1_sampled = shape_o3d.sample_points_uniformly(number_of_points=n_sample_pts)
        pts1_sampled = torch.from_numpy(np.asarray(pts1_sampled.points)).float()

        # transform points, and bbox to world (N, 3) @ (4,4) -> (N, 3)
        pts1_sampled_world = pts1_sampled.squeeze(0) @ t_bo_cur[:3, :3].T + t_bo_cur[:3, 3]

        cd_error_2 = chamfer_distance(gt_points_w.unsqueeze(0), pts1_sampled_world.unsqueeze(0))
        print("cd_error_2:", cd_error_2)
        cd = cd_error_2[0]

        # Update: a new metrics, calculate minimum CD considering 4 rotations.
        t_bo_best = find_best_pose_with_rot_search(gt_points_w, pts1_sampled, t_bo_cur)

        # calculate CD again
        t_bo_best = torch.from_numpy(t_bo_best).float()
        pts1_sampled_world_best = pts1_sampled.squeeze(0) @ t_bo_best[:3, :3].T + t_bo_best[:3, 3]
        cd_error_best = chamfer_distance(
            gt_points_w.unsqueeze(0), pts1_sampled_world_best.unsqueeze(0)
        )
        print("cd_error_best:", cd_error_best)
        cd_best = cd_error_best[0]

        debug_visualize = False
        if debug_visualize:
            # store pointcloud, gt_points_w, pts1_sampled_world_best, pts1_sampled_world
            import open3d as o3d

            o3d_gt_points_w = points_to_o3d(gt_points_w)
            o3d_pts1_sampled_world_best = points_to_o3d(pts1_sampled_world_best)
            o3d_pts1_sampled_world = points_to_o3d(pts1_sampled_world)

            # Save point clouds to disk
            os.makedirs("temp_output", exist_ok=True)
            o3d.io.write_point_cloud("temp_output/gt_points_w.ply", o3d_gt_points_w)
            o3d.io.write_point_cloud(
                "temp_output/pts1_sampled_world_best.ply", o3d_pts1_sampled_world_best
            )
            o3d.io.write_point_cloud("temp_output/pts1_sampled_world.ply", o3d_pts1_sampled_world)

            print("Store ply to temp_output ...")

            # best scale
            o3d_pts1_sampled_world_best_scale = points_to_o3d(pts1_sampled_world_best_scale)
            o3d.io.write_point_cloud(
                "temp_output/o3d_pts1_sampled_world_best_scale.ply",
                o3d_pts1_sampled_world_best_scale,
            )

    except:
        loss_3d_gt = None
        cd = None
        cd_best = None

    loss_rgb, loss_depth, psnr, loss_rgb_frame_list, loss_depth_frame_list, psnr_list = get_2d_loss(
        latent, t_bo_cur, shape_model, obs_id_list, dataset_subset, args
    )

    """
    Update: Besides Training 2D Loss, also evaluate Test 2D Loss on Unseen images
    """
    if open_test_evo:
        (
            loss_rgb_test,
            loss_depth_test,
            psnr_test,
            loss_rgb_frame_list_test,
            loss_depth_frame_list_test,
            psnr_list_test,
        ) = get_test_2d_loss(
            latent, t_bo_cur, shape_model, dataset_subset, obs_id_list, args, save_dir
        )
    else:
        (
            loss_rgb_test,
            loss_depth_test,
            psnr_test,
            loss_rgb_frame_list_test,
            loss_depth_frame_list_test,
            psnr_list_test,
        ) = (torch.Tensor([-1]), torch.Tensor([-1]), torch.Tensor([-1]), [], [], [])

    # Update: Pose Evo
    # Translation, Rotation, Scale...
    gt_data = dataset_subset.load_object_observations_from_scene(
        dataset_subset.scene_name, dataset_subset.obj_id, load_mesh=False, load_image=False
    )
    sym = gt_data["sym"]

    T_est = t_bo_cur.cpu().numpy()

    T_gt_t_world_obj = gt_data["t_world_obj"]

    """
    The pose error coordinate, must be T_world_OBJ_ORIGIN; 
    Not add normalization for NeRF or SDF, so that they are comparable.
    """
    # Need a transformation from world_[bbox_unit_reg] into world_[obj]
    T_est_t_world_obj = transform_t_world_bbox_reg_To_t_world_obj(T_est, gt_data["t_obj_box"])
    # T_gt_t_world_obj = transform_t_world_bbox_reg_To_t_world_obj(T_gt, gt_data['t_obj_box'])

    error_translation, error_rotation, error_scale = calculate_pose_error(
        T_est_t_world_obj, T_gt_t_world_obj, sym
    )

    #########################################

    # Loss output
    evo = {
        "loss_3d_ob": loss_3d_ob,
        "loss_3d_gt": loss_3d_gt,
        "cd": cd,
        "cd_best": cd_best,  # Search with ICP using 4 rotations
        "loss_2d_train": {
            "rgb": loss_rgb,
            "depth": loss_depth,
            "psnr": psnr,
            "rgb_frames": loss_rgb_frame_list,
            "depth_frames": loss_depth_frame_list,
            "psnr_frames": psnr_list,
        },
        "loss_2d_test": {
            "rgb": loss_rgb_test,
            "depth": loss_depth_test,
            "psnr": psnr_test,
            "rgb_frames": loss_rgb_frame_list_test,
            "depth_frames": loss_depth_frame_list_test,
            "psnr_frames": psnr_list_test,
        },
        "pose_error": {
            "translation": error_translation,
            "rotation": error_rotation,
            "scale": error_scale,
        },
    }
    return evo


def calculate_pose_error(T_est, T_gt, sym="None"):
    """
    @ input: T 4x4 matrix, with scale
    @ sym: symmetry type

    Ref to the official Scan2CAD code: https://github.com/skanti/Scan2CAD/blob/master/Routines/Script/EvaluateBenchmark.py
    """

    t, q, s = SE3.decompose_mat4(T_est)

    # gt
    t_gt, q_gt, s_gt = SE3.decompose_mat4(T_gt)

    error_translation = np.linalg.norm(t - t_gt, ord=2)
    error_scale = 100.0 * np.abs(np.mean(s / s_gt) - 1)

    # --> resolve symmetry
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [
            calc_rotation_diff(
                q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])
            )
            for i in range(m)
        ]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [
            calc_rotation_diff(
                q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])
            )
            for i in range(m)
        ]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [
            calc_rotation_diff(
                q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0])
            )
            for i in range(m)
        ]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q, q_gt)
        # debug output
        # print('T_est:', T_est)
        # print('q:', q)
        # print('t:', t)
        # print('s:', s)
        # print('q_gt:', q_gt)
        # print('error_rotation:', error_rotation)

    # # -> define Thresholds
    # threshold_translation = 0.2 # <-- in meter
    # threshold_rotation = 20 # <-- in deg
    # threshold_scale = 20 # <-- in %
    # # <-

    # is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale

    return error_translation, error_rotation, error_scale


def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def transform_t_world_bbox_reg_To_t_world_obj(t_world_bbox_reg, t_obj_box):
    # Step 1: t_world_bbox
    t_world_bbox = inverse_regularize_gt_bbox(t_world_bbox_reg)

    # Step 2: t_world_obj
    t_world_obj = t_world_bbox @ np.linalg.inv(t_obj_box)

    return t_world_obj
