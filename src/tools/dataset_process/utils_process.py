"""
Utils functions for dataset preprocessing.
"""

import numpy as np

from src.dataset.scannet import ScanNet


def analyze_scene_objects_num():

    category = "chair"

    ###
    dataset = ScanNet("data/scannet")

    # get list of scenes
    val_scene_names = dataset.get_scene_name_list()

    n_chairs_list = []
    for scene_name in val_scene_names:
        # load instances of chair
        object_orders = dataset.load_objects_orders_from_scene_with_category(scene_name, category)

        # load ind2scannet
        ind_2_scannet = dataset.scan2cad.load_ind_2_scannet(scene_name)

        n_valid = 0
        for obj_id in object_orders:
            if ind_2_scannet[obj_id] > 0:
                n_valid += 1

        # count valid and not valid
        n_chairs_list.append([n_valid, len(object_orders)])

    # count all
    n_chairs_arr = np.array(n_chairs_list)

    print("all valid: ", np.sum(n_chairs_arr[:, 0]))
    print("all: ", np.sum(n_chairs_arr[:, 1]))
