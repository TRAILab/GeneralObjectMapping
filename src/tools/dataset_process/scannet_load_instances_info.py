"""

Output the number of instances in each categories.

"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.dataset.scannet import ScanNet, ScanNetSubset


def count_category(dataset, category):

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

    return n_chairs_arr


shapenet_category_to_name = {
    "04379243": "table",
    "03593526": "jar",
    "04225987": "skateboard",
    "02958343": "car",
    "02876657": "bottle",
    "04460130": "tower",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02942699": "camera",
    "02691156": "airplane",
    "03642806": "laptop",
    "02801938": "basket",
    "04256520": "sofa",
    "03624134": "knife",
    "02946921": "can",
    "04090263": "rifle",
    "04468005": "train",
    "03938244": "pillow",
    "03636649": "lamp",
    "02747177": "trash bin",
    "03710193": "mailbox",
    "04530566": "watercraft",
    "03790512": "motorbike",
    "03207941": "dishwasher",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    "03337140": "file cabinet",
    "02773838": "bag",
    "02933112": "cabinet",
    "02818832": "bed",
    "02843684": "birdhouse",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02808440": "bathtub",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap",
}

if __name__ == "__main__":

    ###
    dataset_root = "dataset/scannet_mini"

    dataset = ScanNet(dataset_root)

    # consider all categories types in shapenet_category_to_name
    # select all values
    category_list = [shapenet_category_to_name[k] for k in shapenet_category_to_name.keys()]

    data_list = []
    for category in category_list:
        output = count_category(dataset, category)

        data_list.append([output.sum(0)[0], output.sum(0)[1]])

    # print as a table:
    # category, all valid, all
    print("category, all valid, all")
    for i in range(len(category_list)):
        print(f"{category_list[i]}, {np.sum(data_list[i][0])}, {np.sum(data_list[i][1])}")
