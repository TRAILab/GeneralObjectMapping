"""

Use Shap-E model to generate prior shapes from a given text.


"""

import os
import numpy as np
import torch
from tqdm import tqdm

from src.shape_model.shape_io import Shap_E

def set_random_seed(seed=1):
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Set random seed to:", seed)


def run():
    """Initialize Params"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate prior shapes from text")

    # save_dir
    parser.add_argument("--save_dir", type=str, default="./shap_e_prior_meshes")
    parser.add_argument(
        "--category_file_name",
        type=str,
        default="configs/categories.txt",
    )

    args = parser.parse_args()

    # init Shap-E
    """
    Init a Shap-E model
    """
    grid_size = 64
    shape_model = Shap_E(grid_size=grid_size, model="text")

    # Load random seed 1
    set_random_seed(1)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    category_file_name = args.category_file_name

    # load contents from a text file, each line as a category
    with open(category_file_name, "r") as f:
        category_list = f.readlines()
        # ignore \n
        category_list = [x.strip() for x in category_list]

    print("Process categories:", category_list)

    for category in tqdm(category_list, desc="All Categories"):
        print("processing", category)

        file_name = os.path.join(save_dir, f"{category}.ply")
        if os.path.exists(file_name):
            print("Skip", file_name)
            continue

        text = f"a {category}"

        print("text prompt:", text)

        latent = shape_model.get_latent_from_text(
            text, batch_size=1, guidance_scale=3.0, cache=False, cache_dir="./output/cache/"
        )

        shape = shape_model.get_shape_from_latent(latent)

        shape_model.save_shape(shape, file_name)

        print("Saved to", file_name)

if __name__ == "__main__":
    run()
