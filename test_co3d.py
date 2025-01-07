"""

Update Aug 5
Further support CO3D dataset.

Load scannet dataset to generate croped images, 
then use Shap-E to reconstruct.

"""

import os
import json

import open3d as o3d
import torch

from src.utils.args import load_args
from src.dataset import init_dataset
from src.visualization.visualizer import visualize_all_frames_to_input_dir
from src.pipelines.process_instance import process_one_instance_from_all_frames_co3d
from src.utils.random import set_random_seed

def main():
    """
    Main function for the test on ScanNet dataset.
    """
    args = load_args()

    save_dir_root = args.save_root
    print("Save result to:", save_dir_root)

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
    dataset = init_dataset(args)
    print("Loading dataset... Done")

    """   
    Init a Shap-E model
    """
    from shape_model.shape_io import Shap_E

    grid_size = 64
    model_condition = args.diffusion_condition_type
    shape_model = Shap_E(grid_size=grid_size, model=model_condition)

    """
    Init ICP Matcher if using ICP
    """
    icp_matcher = None
    if args.method_init_pose == "icp_class":
        from utils.icp import ICPMatcher

        # init a static ICPMatcher
        icp_matcher = ICPMatcher(args.icp_model_source)

    """
    Visualization Init
    """
    # init open3d vis
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1080, height=1080)

    """
    Begin Processing.

    Consider all the scenes
    """
    total_scene_num = dataset.get_scene_num()
    print("Total scenes:", total_scene_num)

    for scene_order, scene_name in enumerate(dataset.load_scene_names()):
        print("=" * 15)
        print("=> scene:", scene_name, f"({scene_order}/{total_scene_num})")

        save_dir_scene = os.path.join(save_dir_root, f"{scene_name}")

        result_save_dir = os.path.join(save_dir_scene, "output", "result.pt")
        if os.path.exists(result_save_dir) and args.skip:
            # already done.
            continue

        ##############################################
        # start reconstruction
        ##############################################

        # observations
        frames = dataset.get_frames_structure_for_inputs(scene_name, N=10)

        # Optional: Whether to visualize frames into dir
        if args.visualize_frames:
            visualize_all_frames_to_input_dir(frames, save_dir_scene, resize_scale=1.0 / 3.0)

        output = process_one_instance_from_all_frames_co3d(
            frames,
            shape_model,
            args,
            vis=vis,
            icp_matcher=icp_matcher,
            recon_save_dir=save_dir_scene,
        )

        os.makedirs(os.path.dirname(result_save_dir), exist_ok=True)

        if not args.save_detailed_output:
            # Delete history to save disk
            output = {
                "latent": output["latent"],
                "pose_bo": output["pose_bo"],
            }

        evo = None
        torch.save({"output": output, "evo": evo}, open(result_save_dir, "wb"))

    # All scenes are processed
    print("Finish all scenes.")


if __name__ == "__main__":
    main()
