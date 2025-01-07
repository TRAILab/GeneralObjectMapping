"""

Codes related to data processing and loading.

"""

import cv2
import numpy as np
import torch


def is_true_on_edge(mask, percent=0.05):
    # Calculate 10% of the image length and height
    edge_x = int(percent * mask.shape[1])
    edge_y = int(percent * mask.shape[0])

    # Check the extended edges
    return (
        np.any(mask[:edge_y, :])
        or np.any(mask[-edge_y:, :])
        or np.any(mask[:, :edge_x])
        or np.any(mask[:, -edge_x:])
    )


def filter_valid_observations_for_priors(observation_list):
    """
    For each observation, check if it's valid:
        * not in the image edges
        * if all in edges, keep all
    """

    valid_images = []
    valid_observations = []
    for ob in observation_list:
        mask = ob["mask"]
        # check if mask in the edges
        if is_true_on_edge(mask):
            continue

        valid_images.append(ob["rgb_mask_cropped"])
        valid_observations.append(ob)

    # check if none, return all
    if len(valid_images) == 0:
        print("Valid check: all images in the edges. Use all observations.")

        valid_images = [ob["rgb_mask_cropped"] for ob in observation_list]

        valid_observations = observation_list

    return valid_images, valid_observations


def load_observations(obs_id, dataset_subset=None, mask_source="gt"):
    """
    Load Observations from Dataset:
        - RGB
        - Depth
        - Mask
        - Crop RGB with Mask
        ...
    """

    frame = dataset_subset.get_one_frame(obs_id, load_image=True)

    rgb = frame.rgb
    depth = frame.depth

    if mask_source == "gt":
        use_gt = True
    elif mask_source == "mask2former":
        use_gt = False
    else:
        raise NotImplementedError

    mask = dataset_subset._load_mask(
        dataset_subset.scene_name, dataset_subset.obj_id, frame, use_gt=use_gt
    )

    """
    Crop RGB with Mask and Save
    """
    # Find coordinates of non-zero (valid) pixels in the mask
    non_zero_indices = np.nonzero(mask)
    min_x, min_y = np.min(non_zero_indices, axis=1)
    max_x, max_y = np.max(non_zero_indices, axis=1)

    # Crop with no Background
    background = np.zeros_like(rgb)
    background[mask != 0] = rgb[mask != 0]
    rgb_mask_cropped = background[min_x : max_x + 1, min_y : max_y + 1]
    # cropped_rgb_mask_save_name = os.path.join(save_frame_im_dir, f'rgb_cropped_mask_f{obs_id}.png')
    # cv2.imwrite(cropped_rgb_mask_save_name, cropped_rgb_mask)

    # cv2 default BGR, changing to RGB
    rgb_mask_cropped = rgb_mask_cropped[:, :, ::-1]

    output = {
        "rgb_mask_cropped": rgb_mask_cropped,
        "depth": depth,
        "rgb": rgb,
        "mask": mask,
        "obs_id": obs_id,
    }
    return output


def construct_observation_from_dataset(det, dataset_subset, b_debug_use_gt_points=False):
    """
    Construct a frame data from the dataset subset:
        - Images, Points, Camera Poses.

    Args:
        @ det: A frame of current observation
        @ dataset_subset: ScanNetSubset class
    """
    # Note: frame.rgb directly loads image using opencv and store as BGR
    rgb_real = cv2.cvtColor(det.rgb, cv2.COLOR_BGR2RGB)

    """
    Debug Mode: use GT Points
    """
    if b_debug_use_gt_points:
        print("USE GT POINTS AS 3D Constraints.")
        # Debug: load gt pts_3d_c
        # load gt mesh in world, sample points, get gt_points_w, then transform to camera
        try:
            gt_points_w = dataset_subset.get_gt_sampled_points_in_world(10000)

            # transform Nx3 points into Nx3 points with 4x4 transform
            from utils.geometry import transform_3d_points

            T_cam_world = torch.from_numpy(det.T_world_cam).float().inverse()
            pts_3d_c_gt = transform_3d_points(T_cam_world, gt_points_w)
        except:
            # if gt file does not exist
            pts_3d_c_gt = None

            raise ValueError("Mesh Not Found")
    else:
        pts_3d_c_gt = None

    pts_3d_c = det.surface_points

    pts_3d_w = det.surface_points_world

    """
    Construct an observation structure with all the data
    """
    observation = {
        "pts_3d_c": pts_3d_c,  # camera
        "pts_3d_w": pts_3d_w,  # world
        "rgb": rgb_real,
        "depth": det.depth,
        "mask": det.mask,
        "depth_image": det.depth_image,
        "t_wc": det.T_world_cam,
        "t_cw": np.linalg.inv(det.T_world_cam),
        "sub_id": det.sub_id,
    }

    if pts_3d_c_gt is not None:
        # When using GT points, replace the original points
        observation["pts_3d_b"] = pts_3d_c_gt

    return observation


def init_frames_for_instance(
    args, LOOP_INS_ID, scene_order, scene_detail, dataset_subset=None, sample_method="equal"
):
    """
    @option_select_frame: when using dataset_subset_package, choose how to sample frames from selected frames.

    @sample_method: when NOT using dataset_subset_package, how to sample frames for each instance.
        * equal: equally starting from 0 with the same interval to cover all the frames.
        * center: sample frames around the center frame to make sure the observation quality is high and in the center.

    """
    # Select the frames for each instances
    # dataset frames selection
    # get all the observations of this instance
    if args.dataset_name != "scannet":
        raise NotImplementedError

    view_num = args.view_num

    if args.use_gt_association:
        # get frames from package
        if args.dataset_subset_package is not None:
            all_frame_list_ins = scene_detail["scene_ins_frames"]
            # beside the single view frame, we further sample frames from the multi-view frames
            all_frame_list = all_frame_list_ins[scene_order][LOOP_INS_ID]

            # option_select_frame = 'interval'  # max interval

            option_select_frame = args.option_select_frame

            if option_select_frame == "stage_3":
                # Only 3 stage: single/sparse/dense, corresponding to 1,3,10 views;
                # Assume the ID is, 0,1,2,3,4,5,6,7,8,9

                view_groups = {
                    1: [5],
                    3: [5, 3, 7],
                    10: [5, 0, 1, 2, 3, 4, 6, 7, 8, 9],
                }

                if not view_num in view_groups:
                    raise ValueError("Only support 1/3/10 views.")

                # Debug, output options:
                print(" - Select from", view_groups[view_num])
                print(" - all_frame_list:", all_frame_list)

                selected_frames = [all_frame_list[i] for i in view_groups[view_num]]

                all_frame_list = selected_frames
            else:
                # consider view num
                single_frame_list_ins = scene_detail["scene_ins_frames_single"]
                single_frame_id = single_frame_list_ins[scene_order][LOOP_INS_ID][0]

                if view_num > 1:
                    """
                    We need to make sure the first frame is the same as the single view frame.
                    For the remaining frames, we sample in a deterministic way so that each run is the same.
                    """

                    # remove the single view frame
                    all_frame_list_no_single = [
                        frame_id for frame_id in all_frame_list if frame_id != single_frame_id
                    ]
                    N_sample = view_num - 1

                    if option_select_frame == "interval":
                        # sample in a deterministic way
                        all_frame_list_no_single = sorted(all_frame_list_no_single)
                        all_frame_list_no_single = all_frame_list_no_single[
                            :: len(all_frame_list_no_single) // N_sample
                        ]
                        selected_frames = all_frame_list_no_single[:N_sample]

                    elif option_select_frame == "close":  # get closest frame!
                        frame_dist = np.abs(np.array(all_frame_list_no_single) - single_frame_id)
                        # find top K minimum dis indices
                        min_dist_indices = np.argsort(frame_dist)[:N_sample]
                        selected_frames = [all_frame_list_no_single[i] for i in min_dist_indices]
                        # sort
                        selected_frames = sorted(selected_frames)

                    # add the single view frame
                    all_frame_list = [single_frame_id] + selected_frames
                else:
                    all_frame_list = [single_frame_id]

            obs_id_list = all_frame_list
        else:
            """
            The view is not specified.
            """
            if args.frame_id is not None:
                obs_id_list = [args.frame_id]
            else:
                # Sample frames from all valid frames
                max_frame_num = view_num

                # Get the observed frames list!
                # dataset_subset = ScanNetSubset(args.sequence_dir, scene_name, LOOP_INS_ID, load_image = False)
                num_total_frames = len(dataset_subset)

                if sample_method == "equal":
                    # equally sample max_frame_num frames from num_total_frames
                    obs_id_list = np.round(
                        np.linspace(0, num_total_frames - 1, max_frame_num)
                    ).astype(int)
                elif sample_method == "center":
                    # sample frames around the center frame
                    # if single
                    if max_frame_num == 1:
                        obs_id_list = [num_total_frames // 2]
                    else:
                        # Only when the number of frames is one, the center sample_method is valid.
                        raise NotImplementedError
    else:
        # get all the observations of this instance
        object_obs_frames = scene.objects[obj_id].observations
        num_total_frames = len(object_obs_frames)
        print("===> number of observations of the object:", num_total_frames)

        if args.frame_id is not None:
            obs_id_list = [args.frame_id]
        else:
            max_frame_num = 3  # for single frame, consider how many frames

            # only randomly consider two frames
            # Replace: false to select once each
            obs_id_list = np.round(np.linspace(0, num_total_frames - 1, max_frame_num)).astype(int)

    return obs_id_list


def init_instance_order_list_in_scene(
    dataset, scene_name, args, input=None, scene_order=None, category="chair", filter_valid=True
):
    """
    Two options to init observations:
    1) use origin inputs: then we need to consider all frames
    2) use gt data association

    @ filter_valid: if open, ignore those without 3d associations
    """
    if args.dataset_name != "scannet":
        raise NotImplementedError

    # If not using gt association, init association with automatic matching
    output = {}

    if args.use_gt_association:
        # If using gt association,
        # Use input package
        if args.dataset_subset_package is not None:
            # if package is in, use it
            ins_orders_list = input["ins_orders_list_all_scenes"][scene_order]
        else:
            # or, manually load
            if args.obj_id is None:
                if category == "all":
                    category = None
                ins_orders_list = dataset.load_objects_orders_from_scene_with_category(
                    scene_name, category=category
                )
            else:
                ins_orders_list = [args.obj_id]

            # if open filtering valid instances
            if filter_valid:
                print("- Filtering valid instances, before:", ins_orders_list)
                ins_order_list_valid = []

                ind_2_scannet = dataset.scan2cad.load_ind_2_scannet(scene_name)

                for ins_id in ins_orders_list:
                    # check if this instance is valid
                    # get the index of the object
                    if ind_2_scannet[ins_id] != -1:
                        ins_order_list_valid.append(ins_id)

                # update
                ins_orders_list = ins_order_list_valid

                print("- After:", ins_orders_list)

    else:
        """
        Automatically get objects assocaition with observations
        """
        # reconstruct a scene by yourself
        scene = construct_scene(dataset, args.scene_name)
        scene.visualize_objects(vis)
        associate_obj_to_gt(dataset, scene)

        if args.obj_id is None:
            ins_orders_list = scene.get_object_indices_with_category("chair")
        else:
            ins_orders_list = [args.obj_id]

        output["scene"] = scene

    return ins_orders_list, output


def init_scene_list(args, dataset):
    output = {}
    if args.dataset_subset_package is not None:
        # load from the dataset subset package
        dataset_subset_data = torch.load(args.dataset_subset_package)
        # subset_scannet = {
        #     'scene_names': selected_scene_names,
        #     'scene_ins_list': selected_chair_indices,
        #     'n_scenes': len(selected_scene_names),
        #     'n_objects': n_selected_instances,
        #     'category': category,
        #     'version': 'v1',
        #     'description': 'A subset of ScanNet, with only chair instances. Used for debugging.',
        #     'time': time.asctime(time.localtime(time.time()))
        # }
        selected_scene_names = dataset_subset_data["scene_names"]
        ins_orders_list_all_scenes = dataset_subset_data["scene_ins_list"]
        scene_ins_frames_single = dataset_subset_data["scene_ins_frames_single"]
        scene_ins_frames = dataset_subset_data["scene_ins_frames"]
        category_list = dataset_subset_data["category_list"]

        print("==> load dataset subset from:", args.dataset_subset_package)
        print("  Description:", dataset_subset_data["description"])
        print("  Scenes:", dataset_subset_data["n_scenes"])
        print("  Objects:", dataset_subset_data["n_objects"])

        # Update jobs to divide inputs
        if args.jobs_num is not None and args.job_id is not None:
            N = args.jobs_num
            i = args.job_id

            # Calculate Range
            n_objects = len(
                selected_scene_names
            )  # Note that each scene contains only one instance, scene_names can repeat

            # Calculate start and end indices for the current job
            start_index = i * n_objects // N
            end_index = (i + 1) * n_objects // N if i < N - 1 else n_objects

            print("====== Consider Jobs ", i, "of", N, "======")
            print("start_index:", start_index)
            print("end_index:", end_index)
            print("====== Consider Jobs ", i, "of", N, "======")

            # Select the objects for the current job
            selected_scene_names = selected_scene_names[start_index:end_index]
            ins_orders_list_all_scenes = ins_orders_list_all_scenes[start_index:end_index]
            scene_ins_frames_single = scene_ins_frames_single[start_index:end_index]
            scene_ins_frames = scene_ins_frames[start_index:end_index]
            category_list = category_list[start_index:end_index]

        if args.continue_from_scene is not None:
            # update, instead of continuing, we only process objects from this scene
            b_only_consider_scene = True

            start_scene_idx = -1
            if args.continue_from_instance is not None:
                # Also specify an instance id
                # Find this one
                for i in range(len(selected_scene_names)):
                    if (
                        selected_scene_names[i] == args.continue_from_scene
                        and ins_orders_list_all_scenes[i][0] == args.continue_from_instance
                    ):
                        start_scene_idx = i

                # Check if we find it
                if start_scene_idx == -1:
                    raise ValueError(
                        "Cant find the instance:",
                        args.continue_from_scene,
                        args.continue_from_instance,
                    )

            else:
                # start from this scene_name

                if args.continue_from_scene in selected_scene_names:

                    if b_only_consider_scene:
                        # find all indices of this scene
                        # group all instances in this scene
                        def find_indices(lst, element):
                            return [i for i, x in enumerate(lst) if x == element]

                        selected_inds = find_indices(selected_scene_names, args.continue_from_scene)

                        print("==> Process only scene:", args.continue_from_scene)
                        print("==> Process only scene:", selected_inds)

                    else:
                        start_scene_idx = selected_scene_names.index(args.continue_from_scene)

                        print("==> continue from scene:", args.continue_from_scene)
                else:
                    raise ValueError("continue from scene not in the dataset subset package")

            if b_only_consider_scene:

                def extract_elements(lst, slices):
                    return [lst[s] for s in slices]

                selected_scene_names = extract_elements(selected_scene_names, selected_inds)
                ins_orders_list_all_scenes = extract_elements(
                    ins_orders_list_all_scenes, selected_inds
                )
                scene_ins_frames_single = extract_elements(scene_ins_frames_single, selected_inds)
                scene_ins_frames = extract_elements(scene_ins_frames, selected_inds)
                category_list = extract_elements(category_list, selected_inds)
            else:
                selected_scene_names = selected_scene_names[start_scene_idx:]
                ins_orders_list_all_scenes = ins_orders_list_all_scenes[start_scene_idx:]
                scene_ins_frames_single = scene_ins_frames_single[start_scene_idx:]
                scene_ins_frames = scene_ins_frames[start_scene_idx:]
                category_list = category_list[start_scene_idx:]

        output["ins_orders_list_all_scenes"] = ins_orders_list_all_scenes

        # both single and multi-view frames should be considered
        output["scene_ins_frames_single"] = scene_ins_frames_single
        output["scene_ins_frames"] = scene_ins_frames

        output["selected_scene_names"] = selected_scene_names
        output["category_list"] = category_list
    else:
        if args.scene_name is None:
            # iterating over all the scenes
            val_scene_names = dataset.get_scene_name_list()
            MAX_SCENE_CONSIDERATION = 20  # debug
            SCENE_START_IDX = 0

            selected_scene_names = val_scene_names[SCENE_START_IDX:][:MAX_SCENE_CONSIDERATION]
        else:
            selected_scene_names = [args.scene_name]

    return selected_scene_names, output
