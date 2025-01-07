"""

Go through the whole validation set and count the number of instances for each category

Also count the Num with correct GT 3D data association with Scan2CAD.

"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.dataset.scannet import ScanNet, ScanNetSubset


def load_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--category", type=str, default="chair", help="Category to process")

    # mask_path_root
    parser.add_argument(
        "--mask_path_root",
        type=str,
        default=None,
        help="Category to process",
    )

    # dataset roor
    parser.add_argument("--dataset_root", type=str, default="/scannet")

    # save_dir
    parser.add_argument("--save_dir", type=str, default="./")

    # add min_obs_ratio, default 0.2
    parser.add_argument(
        "--min_obs_ratio", type=float, default=0.2, help="Minimum observation ratio"
    )

    args = parser.parse_args()
    return args


def generate_scannet_subset(
    max_obj_num=None,
    category="chair",
    max_frame_num=10,
    min_obs_ratio=0.2,
    gt_mask=True,
    dataset_root="data/scannet",
    save_root="./",
    mask_path_root=None,
):
    """
    Generate a subset of objects in ScanNet dataset (e.g., 100 instances of chairs) for experiments purpose.

    Args:
        - max_frame_num: number of frames to be selected for each object.
        Also, ignore those objects with fewer obesrvations than this.

        - min_obs_ratio: make sure the observation ratio of the frames are larger than this,
        which can filter those severe bad masks.

        - max_obj_num: maximum number of objects to be selected. If None, select all.
        If set, randomly select scenes, so that the accumulated valid objects are larger than max_obj_num

    """
    print("load dataset ...")
    dataset = ScanNet(dataset_root)

    print("get valid scene list ...")
    # get list of scenes
    val_scene_names = dataset.get_scene_name_list()

    num_valid_all_list = []
    valid_instance_indices_scenes_list = []
    for scene_name in val_scene_names:
        # load the orders of all the instances of a specific category in this scene
        object_orders = dataset.load_objects_orders_from_scene_with_category(scene_name, category)

        """Valid Masks Checking"""
        # We pre-matching the GT masks IDs (mask_id) to the instance_id.
        # We only keep those objects with correct GT 3D data association with Scan2CAD.
        # For invalid ones, the scannetid is -1.
        order_2_scannetid = dataset.scan2cad.load_ind_2_scannet(scene_name)

        n_valid = 0
        valid_instance_indices = []
        for obj_id in object_orders:
            if order_2_scannetid[obj_id] > 0:
                n_valid += 1
                valid_instance_indices.append(obj_id)

        # count valid number and all number
        num_valid_all_list.append([n_valid, len(object_orders)])

        valid_instance_indices_scenes_list.append(valid_instance_indices)

    """Start Selecting Splits"""
    # random seed
    np.random.seed(0)

    num_valid_all_arr = np.array(num_valid_all_list)

    # randomly rank all the scenes
    n_selected_instances = 0
    selected_scene_names = []
    selected_instance_indices = []
    selected_scene_ins_frames = []
    selected_scene_ins_frames_single = []

    if max_obj_num is not None:
        random_indices = np.random.permutation(len(val_scene_names))
    else:
        random_indices = range(len(val_scene_names))  # keep the origin order

    print("process valid objects ...")
    for i in tqdm(range(len(val_scene_names))):
        # Selected scene indice in this loop
        scene_indice = random_indices[i]

        scene_name = val_scene_names[scene_indice]

        n_valid = num_valid_all_arr[scene_indice, 0]
        n_total = num_valid_all_arr[scene_indice, 1]

        if n_valid > 0:
            print("scene_name: ", scene_name, "n_valid: ", n_valid, "n_total: ", n_total)

            # valid objects orders in this scene
            obj_inds = valid_instance_indices_scenes_list[scene_indice]

            selected_frame_list = []
            selected_frame_list_single = []
            obj_inds_valid = []
            for obj_id in obj_inds:
                """
                We use ScanNetSubset class to deal with all object-oriented operations for ONE specific object inside a scene.
                """
                dataset_subset = ScanNetSubset(
                    dataset_root,
                    scene_name,
                    obj_id,
                    load_image=False,
                    mask_path_root=mask_path_root,
                )
                num_total_frames = len(dataset_subset)

                # Stop if we have selected enough frames of this object
                if num_total_frames < max_frame_num:
                    continue

                """
                Start Selecting Frames of this Objects into Splits

                Condition:
                    1. Select High Quality Observations, with Minimum Observation Ratio (0.2)
                """
                # randomly sample one frame from all the valid frames
                if min_obs_ratio is not None:
                    # Check valid ratio range
                    assert min_obs_ratio < 1.0 and min_obs_ratio > 0.0

                    # randomly pertube the frames, then check the observation ratio one by one until we get 10 frames
                    sub_frame_indices_random = np.random.permutation(num_total_frames)

                    sub_selected_valid_frame_indices = []
                    for sub_frame_ind in sub_frame_indices_random:
                        try:
                            """Calculate the observation ratio of this frame"""
                            obs_ratio = dataset_subset.get_observation_ratio(
                                sub_frame_ind, gt_mask=gt_mask
                            )
                        except:
                            print("Error in observation ratio calculation. Please Check Try-catch.")
                            obs_ratio = 0

                        if obs_ratio > min_obs_ratio:
                            # Add a valid frame
                            sub_selected_valid_frame_indices.append(sub_frame_ind)

                        # Check if we have enough frames
                        if len(sub_selected_valid_frame_indices) >= max_frame_num:
                            break

                    if len(sub_selected_valid_frame_indices) < max_frame_num:
                        # This object has no enough selected frames
                        print(
                            "This object has no enough selected frames: ",
                            scene_name,
                            " obj_id: ",
                            obj_id,
                            " valid: ",
                            len(sub_selected_valid_frame_indices),
                        )
                        continue

                    selected_frame = np.array(sub_selected_valid_frame_indices)
                else:
                    # If no requirements for observation ratio,
                    # randomly select num_total_frames frames.
                    selected_frame = np.random.choice(
                        num_total_frames, max_frame_num, replace=False
                    )

                # sort
                selected_frame = np.sort(selected_frame)
                selected_frame_list.append(selected_frame)

                # randomly select one frame for single-view experiments
                selected_frame_single = selected_frame[
                    np.random.choice(max_frame_num, 1, replace=False)
                ]
                selected_frame_list_single.append(selected_frame_single)

                # record the valid object id
                obj_inds_valid.append(obj_id)

            n_valid_selected = len(obj_inds_valid)
            n_selected_instances += n_valid_selected
            selected_scene_names.append(scene_name)
            selected_scene_ins_frames.append(selected_frame_list)
            selected_scene_ins_frames_single.append(selected_frame_list_single)
            selected_instance_indices.append(obj_inds_valid)

        # add ckpts for the splits generation, since it takes quite a long time
        if i % 10 == 0 and i > 0:
            subset_scannet = {
                "scene_names": selected_scene_names,
                "scene_ins_list": selected_instance_indices,
                "scene_ins_frames": selected_scene_ins_frames,  # a list of list of frames, for each object
                "scene_ins_frames_single": selected_scene_ins_frames_single,  # a list of list of frames, for each object
                "n_scenes": len(selected_scene_names),
                "n_objects": n_selected_instances,
                "category": category,
                "version": "v3",
                "description": f"A subset of ScanNet, with {category} instances. Filter out those objects with less than {max_frame_num} frames, and with observation ratio >= {min_obs_ratio}",
                # "time": time.asctime(time.localtime(time.time())),
                "max_obj_num": max_obj_num,
                "min_obs_ratio": min_obs_ratio,
                "args": {
                    "SELECTED_FRAME_NUM": max_frame_num,
                },
            }

            save_name_ckpt = f"{save_root}/scannet_subset_f{max_frame_num}_{category}_r{min_obs_ratio}_ckpt{i}.pth"
            torch.save(subset_scannet, save_name_ckpt)

            print("Save ckpt to ", save_name_ckpt)

        # when we have enough objects
        if max_obj_num is not None and n_selected_instances > max_obj_num:
            break

    """Save the final splits"""
    subset_scannet = {
        "scene_names": selected_scene_names,
        "scene_ins_list": selected_instance_indices,
        "scene_ins_frames": selected_scene_ins_frames,  # a list of list of frames, for each object
        "scene_ins_frames_single": selected_scene_ins_frames_single,  # a list of list of frames, for each object
        "n_scenes": len(selected_scene_names),
        "n_objects": n_selected_instances,
        "category": category,
        "version": "v3",
        "description": f"A subset of ScanNet, with {category} instances. Filter out those objects with less than {max_frame_num} frames, and with observation ratio >= {min_obs_ratio}",
        # "time": time.asctime(time.localtime(time.time())),
        "max_obj_num": max_obj_num,
        "min_obs_ratio": min_obs_ratio,
        "args": {
            "SELECTED_FRAME_NUM": max_frame_num,
        },
    }

    torch.save(
        subset_scannet,
        f"{save_root}/scannet_subset_f{max_frame_num}_{category}_r{min_obs_ratio}_all.pth",
    )

    print("Done.")


if __name__ == "__main__":
    args = load_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Specify a category each time.
    generate_scannet_subset(
        category=args.category,
        max_obj_num=None,
        gt_mask=True,
        dataset_root=args.dataset_root,
        save_root=save_dir,
        mask_path_root=args.mask_path_root,
        min_obs_ratio=args.min_obs_ratio,
    )
