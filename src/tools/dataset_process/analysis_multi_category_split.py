"""

Update: Output the instance number and other infomration of the generated splits.

===

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


def default_function():

    # settings

    num_ins_per_cat = 5
    save_dir = "/scannet/split/"

    output_dir = "/scannet/split/"

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
        pth_name = os.path.join(save_dir, f"scannet_subset_f10_{category}_r0.2_all.pth")

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

        # TODO: Extract some information and merged into a new one ...
        # for i in range(num_ins_per_cat):

        # random_scene_id = random.randint(0, len(data['scene_names'])-1)

        # Step one: Select num_ins_per_cat scenes, that have >0 instances
        valid_scene_id_list = []
        for k in range(data["n_scenes"]):
            n_obj_scene = len(data["scene_ins_list"][k])

            if n_obj_scene > 0:
                valid_scene_id_list.append(k)

        # randomly choose num_ins_per_cat from valid_scene_id_list
        valid_scene_num = len(valid_scene_id_list)
        if num_ins_per_cat > valid_scene_num:
            raise ValueError

        # select K not repeat ints
        selected_scene_id = random.sample(range(valid_scene_num), num_ins_per_cat)

        for sid in selected_scene_id:
            random_scene_id = valid_scene_id_list[sid]
            sample_scene_name = data["scene_names"][random_scene_id]

            random_ins_id = random.randint(0, len(data["scene_ins_list"][random_scene_id]) - 1)

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
        "version": "v0.1",
        "description": f"a merged version with 7 categories, and {num_ins_per_cat} instances per category.",
        "time": time.asctime(time.localtime(time.time())),
        "max_obj_num": max_obj_num,
        "min_obs_ratio": min_obs_ratio,
        "args": args,
    }
    torch.save(subset_scannet, f"{output_dir}/scannet_subset_merged_all.pth")

    print("all done")


def analysis_observation_ratio_over_categories():
    split_save_dir = "dataset_split"

    # total categories: 7
    category_list = ["chair", "table", "bookshelf", "sofa", "cabinet", "bed", "bathtub"]

    ratio_list = [0.3, 0.5]

    output_list = []
    for category in category_list:

        for ratio in ratio_list:
            # scratch_exp/april_textcond/dataset_split/scannet_subset_f10_bathtub_r0.3_all.pth
            name_prefix = f"scannet_subset_f10_{category}_r{ratio}_all.pth"

            # data loading
            data = torch.load(os.path.join(split_save_dir, name_prefix))

            # output information:
            n_scenes = data["n_scenes"]
            n_objects = data["n_objects"]

            # add an output string
            out_str = f"{category},{ratio},{n_objects},{n_scenes}"
            output_list.append(out_str)

    # print all
    print("category,ratio,n_obj,n_scene")
    for out_str in output_list:
        print(out_str)


def main():
    """

    Choose one instances from one scenes of the category split instances.

    Limitation: Please make sure num_ins_per_cat < Scene_num_of_all_categories

    """

    analysis_observation_ratio_over_categories()


if __name__ == "__main__":

    main()
