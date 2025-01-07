"""

Merge a subset so that all categories are included.

"""

import os
import random

import numpy as np
import torch


def set_random_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Set random seed to:", seed)


def main():
    """

    Choose one instances from one scenes of the category split instances.

    Limitation: Please make sure num_ins_per_cat < Scene_num_of_all_categories

    """

    # May 3rd, merge all with 0.5 observation ratio
    merge_splits(
        num_ins_per_cat=None,
        data_dir="/output/dataset_split",
        output_dir="/output/dataset_split/merged/",
        ob_ratio=0.5,
    )

    # Also merge one with 5 instances for each, for quick experiments.
    merge_splits(
        num_ins_per_cat=5,
        data_dir="/output/dataset_split",
        output_dir="/output/dataset_split/merged/",
        ob_ratio=0.5,
    )


def merge_splits(
    num_ins_per_cat=None, data_dir="/scannet/split/", output_dir="/scannet/split/", ob_ratio=0.2
):
    """
    @ num_ins_per_cat: Leave None to consider all objects.

    """
    # settings
    save_dir = data_dir

    # total categories: 7
    category_list = ["chair", "table", "bookshelf", "sofa", "cabinet", "bed", "bathtub"]

    # set random seed
    set_random_seed()

    output_scene_names = []
    output_ins_list = []
    output_ins_frames = []
    output_ins_frames_single = []
    output_categories = []

    n_scenes = 0
    n_objects = 0

    category = "all"

    max_obj_num = None
    min_obs_ratio = None
    args = None

    # Begin samplings with random
    for category in category_list:
        pth_name = os.path.join(save_dir, f"scannet_subset_f10_{category}_r{ob_ratio}_all.pth")

        data = torch.load(pth_name)

        # print category name, scene num, and instances name
        print(
            "category: ",
            data["category"],
            " scene num:",
            data["n_scenes"],
            " objects num:",
            data["n_objects"],
        )

        # Step one: Select num_ins_per_cat scenes, that have >0 instances
        valid_scene_id_list = []
        for k in range(data["n_scenes"]):
            n_obj_scene = len(data["scene_ins_list"][k])

            if n_obj_scene > 0:
                valid_scene_id_list.append(k)

        # randomly choose num_ins_per_cat from valid_scene_id_list
        valid_scene_num = len(valid_scene_id_list)
        if num_ins_per_cat is not None and num_ins_per_cat > valid_scene_num:
            raise ValueError

        # select K not repeat ints
        if num_ins_per_cat is None:
            # consider all
            selected_scene_id = range(valid_scene_num)
        else:
            # filter
            selected_scene_id = random.sample(range(valid_scene_num), num_ins_per_cat)

        for sid in selected_scene_id:
            random_scene_id = valid_scene_id_list[sid]
            sample_scene_name = data["scene_names"][random_scene_id]

            # Consider all instances
            if num_ins_per_cat is None:
                ins_list = range(len(data["scene_ins_list"][random_scene_id]))
            else:
                # Original version: consider only one instance under a scene
                random_ins_id = random.randint(0, len(data["scene_ins_list"][random_scene_id]) - 1)

                ins_list = [random_ins_id]

            for random_ins_id in ins_list:
                sample_ins_id = data["scene_ins_list"][random_scene_id][random_ins_id]
                sample_ins_frames = data["scene_ins_frames"][random_scene_id][random_ins_id]
                sample_ins_frames_single = data["scene_ins_frames_single"][random_scene_id][
                    random_ins_id
                ]

                output_scene_names.append(sample_scene_name)
                output_ins_list.append([sample_ins_id])
                output_ins_frames.append([sample_ins_frames])
                output_ins_frames_single.append([sample_ins_frames_single])
                output_categories.append(category)

                n_scenes += 1
                n_objects += 1

        max_obj_num = data["max_obj_num"]
        min_obs_ratio = data["min_obs_ratio"]
        args = data["args"]

    ## Output
    import time

    subset_scannet = {
        "scene_names": output_scene_names,
        "scene_ins_list": output_ins_list,
        "scene_ins_frames": output_ins_frames,  # a list of list of frames, for each object
        "scene_ins_frames_single": output_ins_frames_single,  # a list of list of frames, for each object
        "n_scenes": n_scenes,
        "n_objects": n_objects,
        "category": category,
        "category_list": output_categories,
        "version": "v1.0",
        "description": f"a merged version with 7 categories, and {num_ins_per_cat} instances per category.",
        # "time": time.asctime(time.localtime(time.time())),
        "max_obj_num": max_obj_num,
        "min_obs_ratio": min_obs_ratio,
        "args": args,
    }

    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        subset_scannet,
        f"{output_dir}/scannet_subset_merged_all_r{min_obs_ratio}_n{num_ins_per_cat}.pth",
    )

    print("all done")


if __name__ == "__main__":

    main()
