"""

This file is testing the sampled shapes from diffusion model.

It loads from ScanNet dataset, and output the shapes.

"""

import os

# add dir .., insert to 0
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


def set_random_seed(seed=1):
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Set random seed to:", seed)


def test_priors_texts(text_descript_list):
    """
    Update: Sample multiple prior instances from a text descriptions.
    """
    # save_dir_root = 'output/test_prior_samples/text'
    save_dir_root = "output/test_prior_samples/text_dbg"

    import datetime

    now = datetime.datetime.now()
    save_dir = os.path.join(save_dir_root, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir:", save_dir)

    set_random_seed()

    """   
    Init a Shap-E model
    """
    from shape_model.shape_io import Shap_E

    grid_size = 64
    shape_model = Shap_E(grid_size=grid_size, model="text")

    for text in text_descript_list:
        text_as_dir = text.replace(" ", "_")
        save_dir_text = os.path.join(save_dir, text_as_dir)
        os.makedirs(save_dir_text, exist_ok=True)

        # Save shapes
        N_sample = 5
        # generate 5 shapes and save
        for i in range(N_sample):
            latent = shape_model.get_latent_from_text(text)

            shape = shape_model.get_shape_from_latent(latent)
            shape_model.save_shape(shape, os.path.join(save_dir_text, f"shape_sample_{i}.ply"))


def test_priors_images():
    save_dir_root = "output/test_prior_samples"
    # get a save_dir with run time prefix
    import datetime

    now = datetime.datetime.now()
    save_dir = os.path.join(save_dir_root, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir:", save_dir)

    # random seed
    set_random_seed()

    # Load dataset images
    # im_dir = 'output/scannet_evo_0123_pure_prior_view_10/scene0568_00/ins_order_2/frame-0/input/rgb_cropped_mask_f27.png'
    # im_dir = 'output/scannet_evo_0123_pure_prior_view_10/scene0568_00/ins_order_5/frame-0/input/rgb_cropped_mask_f19.png'
    # im_dir = 'output/scannet_evo_0123_pure_prior_view_10/scene0568_00/ins_order_9/frame-0/input/rgb_cropped_mask_f28.png'
    im_dir = "output/scannet_evo_0123_pure_prior_view_10/scene0568_00/ins_order_10/frame-0/input/rgb_cropped_mask_f23.png"
    image = cv2.imread(im_dir)
    # bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # copy input image into the dir
    cv2.imwrite(os.path.join(save_dir, "input_image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run diffusion
    # Load diffusion model
    """   
    Init a Shap-E model
    """
    from shape_model.shape_io import Shap_E

    grid_size = 64
    shape_model = Shap_E(grid_size=grid_size)

    # Save shapes
    N_sample = 5
    # generate 5 shapes and save
    for i in range(N_sample):
        latent = shape_model.get_latent_from_image(image)
        shape = shape_model.get_shape_from_latent(latent)
        shape_model.save_shape(shape, os.path.join(save_dir, f"shape_sample_{i}.ply"))


if __name__ == "__main__":
    # test_priors_images()

    # text_descript_list = [
    #     'a chair',
    #     'a green chair',
    #     'a green chair with a back'
    # ]
    # text_descript_list = [
    #     'a hydrant',
    #     'a teddybear',
    #     'an umbrella',
    #     'a kite',
    #     'a motorcycle',
    #     'an orange',
    #     'a banana',
    #     'a remote',
    # ]
    text_descript_list = ["an object"]
    test_priors_texts(text_descript_list)
