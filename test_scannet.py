"""

Load scannet dataset to generate croped images, 
then use Shap-E to reconstruct.

"""

import os
import json
import time

import numpy as np
import open3d as o3d
import torch

from src.utils.args import load_args
from src.shape_model.shape_io import Shap_E

from src.tools.evo import evaluate_output
from src.tools.evo import print_evo_list

from src.pipelines.process_instance import process_one_instance_from_all_frames

from src.dataset.loader import init_frames_for_instance, init_instance_order_list_in_scene, init_scene_list
from src.visualization.visualizer import visualize_all_frames

from src.dataset.scannet import ScanNet, ScanNetSubset
from src.utils.random import set_random_seed

def main():
    """
    Main function for the test on ScanNet dataset.
    """
    args = load_args()

    save_dir_root = args.save_root
    print("Save result to:", save_dir_root)

    """Configs"""
    skip = args.skip  # skip if the result exists

    # save args (a dict) to a json file
    os.makedirs(save_dir_root, exist_ok=True)
    args_save_name = os.path.join(save_dir_root, "args.json")
    with open(args_save_name, "w") as f:
        json.dump(args, f)

    # set random seed
    set_random_seed(args.random_seed)

    """
    Dataset Loading
    """
    # load scannet
    dataset = ScanNet(args.sequence_dir)
    print("Loading dataset... Done")

    """   
    Init a Shap-E model
    """
    model_condition = args.diffusion_condition_type
    shape_model = Shap_E(grid_size=64, model=model_condition)

    """
    Init ICP Matcher if using ICP
    """
    icp_matcher = None
    if args.method_init_pose == "icp_class":
        from src.utils.icp import ICPMatcher

        # init a static ICPMatcher
        icp_matcher = ICPMatcher(args.icp_model_source)
    else:
        print("Not using ICP Matcher")

    """
    Begin Processing.

    Consider all the scenes
    """
    # Begin iterating over dataset
    selected_scene_names, scene_detail = init_scene_list(args, dataset)

    total_scene_num = len(selected_scene_names)
    print("Total scenes:", total_scene_num)

    ins_time_list = []
    for scene_order, scene_name in enumerate(selected_scene_names):
        print("=" * 15)
        print("=> scene:", scene_name, f"({scene_order}/{total_scene_num})")

        """
        Data Parsing
        """
        # Category Filtering
        if args.specific_category is not None and len(scene_detail) > 0:
            category_list = scene_detail["category_list"]
            obj_category = category_list[scene_order]
            if obj_category != args.specific_category:
                print(f"Skip scene {scene_name} because of category {obj_category}")
                continue

        save_dir_scene = os.path.join(save_dir_root, f"{scene_name}")

        # Load splits for the indices of instances, and all the observations in the scene
        ins_orders_list, instance_detail = init_instance_order_list_in_scene(
            dataset, scene_name, args, scene_detail, scene_order, category=args.dataset_category
        )

        print("=> Consider instances:", ins_orders_list)

        """
        Begin iterating over instances
        """
        for LOOP_INS_ID, obj_id in enumerate(ins_orders_list):
            time_ins_start = time.time()

            print("==> Reconstructing Instance:", obj_id)

            # Init dirs for each instance
            save_dir_scene_instance = os.path.join(save_dir_scene, f"ins_order_{obj_id}")
            os.makedirs(save_dir_scene_instance, exist_ok=True)

            # Init a dataset subset for the instance
            dataset_subset = ScanNetSubset(
                args.sequence_dir,
                scene_name,
                obj_id,
                load_image=False,
                mask_path_root=args.mask_path_root,
            )

            print("==> category:", dataset_subset.get_category_name())

            """
            Sample observations of frames for the instance
            """
            sample_method = args.dataset_frame_sample_method
            obs_id_list = init_frames_for_instance(
                args,
                LOOP_INS_ID, #TODO: Why using LOOP_INS_ID.
                scene_order,
                scene_detail,
                dataset_subset,
                sample_method=sample_method,
            )
            print("===> Frame list:", obs_id_list)

            # Initialize saving dir, using the id of the first frame.
            save_dir_scene_instance_frame = os.path.join(
                save_dir_scene_instance, f"frame-{obs_id_list[0]}"
            )
            os.makedirs(save_dir_scene_instance_frame, exist_ok=True)

            save_dir_scene_instance_frame_output = os.path.join(
                save_dir_scene_instance_frame, "output"
            )

            result_save_dir = os.path.join(save_dir_scene_instance_frame_output, "result.pt")
            if skip and os.path.exists(result_save_dir):
                # Skip if the result exists
                continue

            """
            start reconstruction
            """
            if args.visualize_frames:
                # Optional: Whether to visualize frames into dir
                visualize_all_frames(
                    obj_id,
                    obs_id_list,
                    save_dir_scene_instance_frame,
                    LOOP_INS_ID,
                    scene_name,
                    dataset_subset,
                    args=args,
                    resize_scale=1.0 / 3.0,
                )

            # Activate try-catch if doing large-scale processing
            output = process_one_instance_from_all_frames(
                obs_id_list,
                save_dir_scene_instance_frame,
                dataset_subset,
                shape_model,
                args,
                vis=None,
                icp_matcher=icp_matcher,
            )

            time_ins_end = time.time()

            ins_time_list.append(time_ins_end - time_ins_start)

    # All scenes are processed
    print("Finish all scenes.")


if __name__ == "__main__":
    main()
