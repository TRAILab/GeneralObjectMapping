"""

Visualization codes.

"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.image import resize_image


def visualize_all_frames(
    obj_id,
    obs_id_list,
    save_dir_scene_instance_frame,
    LOOP_INS_ID,
    scene_name,
    dataset_subset,
    args=None,
    resize_scale=1.0,
    separate_images=False,
):
    if separate_images:
        for obs_id in obs_id_list:
            vis_frame_func(
                None,
                obj_id,
                obs_id,
                save_dir_scene_instance_frame,
                LOOP_INS_ID,
                obs_id,
                scene_name,
                dataset_subset,
                args=args,
                resize_scale=resize_scale,
            )
    else:

        """Constrcut the frames"""
        frames = []
        for obs_id in obs_id_list:
            frame = dataset_subset.get_one_frame(obs_id, load_image=True)

            mask = dataset_subset._load_mask(
                dataset_subset.scene_name, dataset_subset.obj_id, frame, use_gt=True
            )

            # Sanity check range, shape of rgb/depth/mask
            # print("==> RGB shape:", frame.rgb.shape)
            # print("==> Depth shape:", frame.depth.shape)
            # print("==> Mask shape:", mask.shape)

            # # Range
            # print("==> RGB range:", frame.rgb.min(), frame.rgb.max())
            # print("==> Depth range:", frame.depth.min(), frame.depth.max())
            # print("==> Mask range:", mask.min(), mask.max())

            depth_255 = (frame.depth / frame.depth.max() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

            frames.append({"frame": frame, "rgb": bgr, "depth": depth_255, "mask": mask})

        visualize_frame_to_input_dir_organized(
            frames, save_dir_scene_instance_frame, resize_scale=resize_scale
        )


def vis_frame_func(
    scene,
    obj_id,
    obs_id,
    recon_save_dir_prefix_frame,
    iter_ins_order=None,
    sub_id=None,
    scene_name=None,
    dataset_subset=None,
    args=None,
    resize_scale=1.0,
):
    # visualize RGB, Depth, Mask
    save_frame_im_dir = os.path.join(recon_save_dir_prefix_frame, "input")
    os.makedirs(save_frame_im_dir, exist_ok=True)

    if scene is not None:
        rgb = scene.objects[obj_id].observations[obs_id].rgb
        depth = scene.objects[obj_id].observations[obs_id].depth
        mask = scene.objects[obj_id].observations[obs_id].mask_inflated
    else:
        frame = dataset_subset.get_one_frame(obs_id, load_image=True)

        rgb = frame.rgb
        depth = frame.depth

        use_gt = True
        if args is not None:
            use_gt = args.mask_source == "gt"
        mask = dataset_subset._load_mask(
            dataset_subset.scene_name, dataset_subset.obj_id, frame, use_gt=use_gt
        )  # loading gt mask.

    # save to the disk
    rgb_save_name = os.path.join(save_frame_im_dir, f"rgb_f{obs_id}.png")
    depth_save_name = os.path.join(save_frame_im_dir, f"depth_f{obs_id}.png")
    mask_save_name = os.path.join(save_frame_im_dir, f"mask_f{obs_id}.png")

    cv2.imwrite(rgb_save_name, resize_image(rgb, resize_scale))
    plt.imsave(depth_save_name, resize_image(depth, resize_scale))

    if mask is None:
        print("Invalid mask for test evo.")
        return

    plt.imsave(mask_save_name, mask)

    """
    Update: Crop RGB with Mask and Save
    """
    # Find coordinates of non-zero (valid) pixels in the mask
    non_zero_indices = np.nonzero(mask)
    if len(non_zero_indices[0]) > 0:
        min_x, min_y = np.min(non_zero_indices, axis=1)
        max_x, max_y = np.max(non_zero_indices, axis=1)

        # Crop the area of the minimum bounding box from the RGB image
        cropped_rgb = rgb[min_x : max_x + 1, min_y : max_y + 1]
        cropped_rgb_save_name = os.path.join(save_frame_im_dir, f"rgb_cropped_f{obs_id}.png")
        cv2.imwrite(cropped_rgb_save_name, resize_image(cropped_rgb, resize_scale))

        ##
        # Crop with no Background
        # background = np.zeros_like(rgb) # Update: use white background
        background = np.ones_like(rgb) * 255
        background[mask != 0] = rgb[mask != 0]
        cropped_rgb_mask = background[min_x : max_x + 1, min_y : max_y + 1]
        cropped_rgb_mask_save_name = os.path.join(
            save_frame_im_dir, f"rgb_cropped_mask_f{obs_id}.png"
        )
        cv2.imwrite(cropped_rgb_mask_save_name, resize_image(cropped_rgb_mask, resize_scale))


def visualize_all_frames_to_input_dir(frames, save_dir, resize_scale=1.0, separate_images=False):
    # visualize RGB, Depth, Mask
    save_frame_im_dir = os.path.join(save_dir, "input")
    os.makedirs(save_frame_im_dir, exist_ok=True)

    if separate_images:
        for frame in frames:
            # Saving one: Save each images separately into disk
            visualize_frame_to_input_dir(frame, save_frame_im_dir, resize_scale=resize_scale)

    else:

        # Saving option 2: Save all images into one image
        visualize_frame_to_input_dir_organized(frames, save_frame_im_dir, resize_scale=resize_scale)


def visualize_frame_to_input_dir_organized(frames, save_dir, resize_scale=1.0):
    """
    This functino organizes the visualization of the frame into a single image.

    Final Ouput Image:
        | RGB | Depth | Mask | Cropped RGB | Cropped RGB with Mask |
    """

    # Get the number of frames
    # num_frames = len(frames)

    # Initialize the final image
    final_image = None

    # Loop over all the frames
    for idx, frame in enumerate(frames):
        rgb_ori = frame["rgb"]
        depth = frame["depth"]
        mask = frame["mask"]

        # Resize the images
        rgb = resize_image(rgb_ori, resize_scale)
        depth = resize_image(depth, resize_scale)

        mask_vis = mask.squeeze().astype(np.uint8) * 255
        mask_vis = resize_image(mask_vis, resize_scale)

        """
        Crop RGB with Mask
        """
        background = np.ones_like(rgb_ori) * 255
        background[mask != 0] = rgb_ori[mask != 0]
        background = resize_image(background, resize_scale)

        # Concatenate the images
        # frame_image = np.concatenate((rgb, depth, mask_vis), axis=1)
        # Note the shape is (H, W, C), (H, W), (H, W), so we need to add the channel dimension to the depth and mask, and repeat the RGB image to 3 channels
        frame_image = np.concatenate(
            (
                rgb,
                np.repeat(depth[:, :, None], 3, axis=2),
                np.repeat(mask_vis[:, :, None], 3, axis=2),
                background,
            ),
            axis=0,
        )

        # Add the frame image to the final image
        if final_image is None:
            final_image = frame_image
        else:
            final_image = np.concatenate((final_image, frame_image), axis=1)

    # Save the final image to the disk
    final_image_save_name = os.path.join(save_dir, "input_frames.png")
    # BGR to RGB
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    # if too large, resize the images
    if final_image.shape[0] > 1000:
        final_image = cv2.resize(
            final_image, (final_image.shape[1] // 2, final_image.shape[0] // 2)
        )

    cv2.imwrite(final_image_save_name, final_image)

    print(f"Saved the organized frames to: {final_image_save_name}")


def visualize_frame_to_input_dir(frame, save_dir, resize_scale=1.0):
    obs_id = frame["frame"].frame_number.item()

    rgb = frame["rgb"]
    depth = frame["depth"]
    mask = frame["mask"]

    # C, H, W -> H, W, C, with torch
    # rgb = rgb.permute(1, 2, 0).cpu().numpy()
    # depth = depth.permute(1, 2, 0).cpu().numpy()
    # mask = mask.permute(1, 2, 0).cpu().numpy()

    # save to the disk
    rgb_save_name = os.path.join(save_dir, f"rgb_f{obs_id}.png")
    depth_save_name = os.path.join(save_dir, f"depth_f{obs_id}.png")
    mask_save_name = os.path.join(save_dir, f"mask_f{obs_id}.png")

    import matplotlib.pyplot as plt

    plt.imsave(rgb_save_name, resize_image(rgb, resize_scale))
    plt.imsave(depth_save_name, resize_image(depth, resize_scale))

    # type mask: True / False
    mask_vis = mask.squeeze().astype(np.uint8) * 255
    plt.imsave(mask_save_name, resize_image(mask_vis, resize_scale))

    """
    Update: Crop RGB with Mask and Save
    """
    # Find coordinates of non-zero (valid) pixels in the mask
    mask = mask.squeeze()
    non_zero_indices = np.nonzero(mask)
    if len(non_zero_indices[0]) > 0:
        min_x, min_y = np.min(non_zero_indices, axis=1)
        max_x, max_y = np.max(non_zero_indices, axis=1)

        # Crop the area of the minimum bounding box from the RGB image
        cropped_rgb = rgb[min_x : max_x + 1, min_y : max_y + 1]
        cropped_rgb_save_name = os.path.join(save_dir, f"rgb_cropped_f{obs_id}.png")
        plt.imsave(cropped_rgb_save_name, resize_image(cropped_rgb, resize_scale))

        ##
        # Crop with no Background
        # background = np.zeros_like(rgb) # Update: use white background
        background = np.ones_like(rgb) * 255
        background[mask != 0] = rgb[mask != 0]
        cropped_rgb_mask = background[min_x : max_x + 1, min_y : max_y + 1]
        cropped_rgb_mask_save_name = os.path.join(save_dir, f"rgb_cropped_mask_f{obs_id}.png")
        plt.imsave(cropped_rgb_mask_save_name, resize_image(cropped_rgb_mask, resize_scale))
