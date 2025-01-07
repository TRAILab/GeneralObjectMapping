"""

Given an experiment dir, loading the result for each instance, and output as a table.

"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os

import torch

from tools.evo import print_evo_list


def get_instance_id(instance_name):
    # Assuming instance id is a number at the end of the instance name
    return int("".join(filter(str.isdigit, instance_name)))


def main(exp_dir):

    print("Processing: ", exp_dir)

    # torch.save({
    #     'output': output,
    #     'evo': evo
    # }, open(os.path.join(save_dir_scene_instance_frame_output, 'result.pt'), 'wb'))

    # check if exist
    if os.path.exists(os.path.join(exp_dir)) is False:
        print("Dir not found in %s" % exp_dir)
        return

    # list scene_names
    scene_name_list = os.listdir(exp_dir)

    # for each scene, load instance order
    evo_list = []
    for scene_name in scene_name_list:
        scene_dir = os.path.join(exp_dir, scene_name)
        if os.path.isdir(scene_dir):
            instance_name_list = os.listdir(scene_dir)
            # rank ins_order_10, according to the final number
            instance_name_list = sorted(instance_name_list, key=get_instance_id)

            for instance_name in instance_name_list:
                instance_dir = os.path.join(scene_dir, instance_name)
                if os.path.isdir(instance_dir):
                    frame_name_list = os.listdir(instance_dir)
                    for frame_name in frame_name_list:
                        frame_dir = os.path.join(instance_dir, frame_name)
                        if os.path.isdir(frame_dir):
                            # load result.pt
                            result_path = os.path.join(frame_dir, "output", "result.pt")
                            if os.path.exists(result_path):
                                result = torch.load(result_path)
                                output = result["output"]
                                evo = result["evo"]
                                # print
                                # print_evo_list(evo)
                                evo["name"] = instance_name
                                evo_list.append(evo)
                            else:
                                print("result.pt not found in %s" % frame_dir)
                        else:
                            print("frame_dir not found in %s" % instance_dir)
                else:
                    print("instance_dir not found in %s" % scene_dir)
        else:
            print("scene_dir not found in %s" % exp_dir)

    # load evo
    # evo = torch.load(os.path.join(exp_dir, 'result.pt'))['evo']

    # print
    print_evo_list(evo_list)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()

    exp_dir = args.exp_dir

    main(exp_dir)
