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


def test_priors_multi_views(input_prior_image_dir):
    """
    This is a test function,
    (1) generate prior shapes from each views
    (2) fuse multi-view priors, and generate shape for the fused latent
    """
    print("Processing input_prior_image_dir:", input_prior_image_dir)

    # DIR/rgb_cropped_mask_fK.png, K is automatically loaded
    # input_prior_image_dir = 'output/0225_fix3dmetric/10views_regularizer10_wprior/scene0568_00/ins_order_2/frame-0/input'
    # input_view_num = 10

    save_dir_root = "output/test_prior_multi_views"
    # get a save_dir with run time prefix
    import datetime

    now = datetime.datetime.now()
    save_dir = os.path.join(save_dir_root, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir:", save_dir)

    # random seed
    set_random_seed()

    # Load images under dir, with name of rgb_cropped_mask_fK.png
    # Using glob
    import glob

    im_dirs = glob.glob(os.path.join(input_prior_image_dir, "rgb_cropped_mask_f*.png"))
    # rank with fK, K, note K can have many digits
    im_dirs.sort(key=lambda x: int(x.split("f")[-1].split(".png")[0]))

    print("im_dirs:", im_dirs)

    # Load diffusion model
    """   
    Init a Shap-E model
    """
    from shape_model.shape_io import Shap_E

    grid_size = 64
    shape_model = Shap_E(grid_size=grid_size)

    # loop all images, generate meshes
    latent_list = []
    for im_dir in im_dirs:
        image = cv2.imread(im_dir)
        # bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # copy input image into the dir
        im_save_name = os.path.join(save_dir, os.path.basename(im_dir))
        cv2.imwrite(im_save_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save shapes
        N_sample = 1
        # generate 5 shapes and save
        for i in range(N_sample):
            latent = shape_model.get_latent_from_image(image)
            shape = shape_model.get_shape_from_latent(latent)

            # image_name_sample_i
            save_shape_name = os.path.join(
                save_dir, os.path.basename(im_dir).replace(".png", "_sample_{}.ply".format(i))
            )
            shape_model.save_shape(shape, save_shape_name)

            latent_list.append(latent)

            print("save_shape_name:", save_shape_name)

    # fuse all latent by averaging all
    fused_latent = torch.mean(torch.stack(latent_list), dim=0)
    fused_shape = shape_model.get_shape_from_latent(fused_latent)
    save_fused_shape_name = os.path.join(save_dir, "fused_shape.ply")
    shape_model.save_shape(fused_shape, save_fused_shape_name)

    print("all finish")


if __name__ == "__main__":
    # input_prior_image_dir = 'output/0225_fix3dmetric/10views_regularizer10_wprior/scene0568_00/ins_order_2/frame-0/input'
    # test_priors_multi_views(input_prior_image_dir)

    # Deal with all dirs under
    root_dir = "output/0225_fix3dmetric/10views_regularizer10_wprior/scene0568_00"

    # use glob to load
    import glob

    ins_order_dir_list = [
        os.path.basename(x) for x in glob.glob(f"{root_dir}/*") if os.path.isdir(x)
    ]

    # Consider all ins_order_K
    for ins_order_dir in ins_order_dir_list:
        # skip ins_order_2
        if ins_order_dir == "ins_order_2":
            continue

        input_prior_image_dir = f"{root_dir}/{ins_order_dir}/frame-0/input"
        test_priors_multi_views(input_prior_image_dir)

    print("all done")
