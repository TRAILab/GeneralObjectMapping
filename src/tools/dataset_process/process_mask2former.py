"""

This file needs environment of MMDetection.
Process all images with mask2former model under the dir and store.

"""

"""

Run object detection using mmdet for a given folder
"""

"""

Run MaskRCNN on Pix3D dataset and save the results.
Please run under MMdetection configuration.

"""
import glob
import os

import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm


def largest_area_over_thresh(chair_masks, chair_bboxs, thresh):
    # for those masks have prob > thresh, get the one with largest areas
    # areas = [msk.sum() for msk in chair_masks]
    # probs = chair_bboxs[:, -1]
    # select
    areas = []
    ids = []
    for id, bbox in enumerate(chair_bboxs):
        if bbox[-1] > thresh:
            # save area and id
            areas.append(chair_masks[id].sum())
            ids.append(id)

    if len(areas) == 0:
        # only one, failed.
        return None
    else:
        # get the biggest one
        id_biggest = np.array(areas).argmax()
        ind_max = ids[id_biggest]
    return ind_max


def select_biggest_chair(result, ignore_class=False, method="highest_prob", gt_mask=None):
    """
    @method = 'highest_prob'; largest_area; largest_area_over_thresh
    """
    bboxs, masks = result

    if ignore_class:
        # Go through all classes and find the biggest area and corresponding mask
        area_list = []
        mask_list = []
        for chair_masks in masks:
            if len(chair_masks) == 0:
                # only one, failed.
                area = 0
                msk = None
            else:
                # at least one. get the one with largest areas
                areas = [msk.sum() for msk in chair_masks]

                # max ind
                ind_max = np.array(areas).argmax()
                mask_int8 = chair_masks[ind_max].astype(np.uint8) * 255

                area = areas[ind_max]
                msk = mask_int8

            area_list.append(area)
            mask_list.append(msk)

        ind_max_class = np.array(area_list).argmax()
        mask_int8 = mask_list[ind_max_class]
    else:
        # labels 80
        chair_class_id = 56  # model.CLASSES

        chair_masks = masks[chair_class_id]
        chair_bboxs = bboxs[
            chair_class_id
        ]  # Choose the one with highest prob! which is just the first one.

        if len(chair_masks) == 0:
            # only one, failed.
            return None

        if method == "highest_prob":
            # note that the first one has highest prob
            ind_max = 0

        elif method == "largest_area":
            # at least one. get the one with largest areas
            areas = [msk.sum() for msk in chair_masks]

            # max ind
            ind_max = np.array(areas).argmax()

        elif method == "largest_area_over_thresh":
            thresh = 0.5
            ind_max = largest_area_over_thresh(chair_masks, chair_bboxs, thresh)
        elif method == "largest_area_over_thresh_0.1":
            thresh = 0.1
            ind_max = largest_area_over_thresh(chair_masks, chair_bboxs, thresh)
        elif method == "largest_area_over_thresh_0":
            thresh = 0
            ind_max = largest_area_over_thresh(chair_masks, chair_bboxs, thresh)
        else:
            raise ValueError("Unknown method")

        if ind_max is None:
            return None
        mask_int8 = chair_masks[ind_max].astype(np.uint8) * 255

    return mask_int8


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Method being used")
    parser.add_argument("-i", "--input_source", help="Input source", default="all")

    # add
    parser.add_argument(
        "--mask_selection_method", help="Method being used", default="largest_area_over_thresh"
    )

    # add 4 params
    parser.add_argument("--dataset_dir", help="Dataset dir", default="dataset/scannet_mini")
    parser.add_argument("--save_root_dir", help="Save root dir", default="output/mask2former/")
    parser.add_argument(
        "--config_file",
        help="Config file",
        default="configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
    )
    parser.add_argument(
        "--checkpoint_file",
        help="Checkpoint file",
        default="models/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth",
    )

    return parser.parse_args()


def detect(model, im_name, engine):
    if engine == "mmdet":
        result = inference_detector(model, im_name)
    elif engine == "meta":
        # image = cv2.imread(im_name)
        image = cv2.imread(im_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            result = model.generate(image)

            # change result to mmdet format
            # segmentation : the mask
            # area : the area of the mask in pixels
            # bbox : the boundary box of the mask in XYWH format
            # predicted_iou : the model's own prediction for the quality of the mask
            # point_coords : the sampled input point that generated this mask
            # stability_score : an additional measure of mask quality
            # crop_box : the crop of the image used to generate this mask in XYWH format

            # Change to format:
            # result = [bboxs, masks]
            # masks = [msk1, ..., mskN]
            bboxs = []
            masks = []
            for data in result:
                masks.append(data["segmentation"])
                bboxs.append(data["bbox"])
            result = [bboxs, masks]
        except:
            print("Fail to generate mask for ", im_name)
            result = None

    return result


def show_result(model, im_name, result, out_save_name, detect_engine):
    if detect_engine == "mmdet":
        model.show_result(im_name, result, out_file=out_save_name)
    elif detect_engine == "meta":
        pass


def process_scannet_dataset(dataset_dir, save_dir_root, config, model_ckpt):

    model = initialize_model(config, model_ckpt)
    # model = None

    # Load all the scenes dirs, process and store images
    # use glob to get all
    scene_name_list = glob.glob(os.path.join(dataset_dir, "data/posed_images/", "*"))
    scene_name_list = [os.path.basename(x) for x in scene_name_list]

    # For each scene dir, process all images
    for scene_name in tqdm(scene_name_list):
        save_dir_mask = os.path.join(save_dir_root, scene_name)
        process_scene(dataset_dir, scene_name, model, save_dir_mask)

    print("scannet dataset finishes.")


def initialize_model(config, model_ckpt):
    # config_file = 'configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    # checkpoint_file = 'models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    return model


def process_scene(dataset_dir, scene_name, model, save_dir_mask):
    print("save dir:", save_dir_mask)
    os.makedirs(save_dir_mask, exist_ok=True)

    # preprocess_image_dir = f'data/scannet/data/posed_images/{scene_name}'
    preprocess_image_dir = os.path.join(dataset_dir, "data/posed_images", scene_name)

    # construct an input_lists for all the *.jpg under the dir
    input_lists = glob.glob(os.path.join(preprocess_image_dir, "*.jpg"))

    # sort
    input_lists.sort()

    print("len of images:", len(input_lists))

    for im_name in tqdm(input_lists):
        im_id = im_name.split("/")[-1].split(".")[0]

        mask_save_name = os.path.join(save_dir_mask, im_id + ".png")
        if (not "png" in im_name) and os.path.exists(mask_save_name):
            continue

        result = detect(model, im_name, "mmdet")

        # do not select and save all
        scores = result.pred_instances.scores.detach().cpu().numpy()
        labels = result.pred_instances.labels.detach().cpu().numpy()
        masks = result.pred_instances.masks.detach().cpu().numpy()
        bboxes = result.pred_instances.bboxes.detach().cpu().numpy()

        score_th = 0.5
        scores_th = scores[scores > score_th]
        labels_th = labels[scores > score_th]
        masks_th = masks[scores > score_th]
        bboxes_th = bboxes[scores > score_th]

        # save as torch file
        torch.save(
            {"scores": scores_th, "labels": labels_th, "masks": masks_th, "bboxes": bboxes_th},
            mask_save_name.replace(".png", ".pth"),
        )

    print("done")


if __name__ == "__main__":
    args = parse_args()
    print("Arguments:", args)

    # dataset_dir = 'dataset/scannet_mini'
    # save_root_dir = 'output/mask2former/'

    # config_file = 'configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'
    # checkpoint_file = 'models/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'

    dataset_dir = args.dataset_dir
    save_root_dir = args.save_root_dir
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file

    process_scannet_dataset(dataset_dir, save_root_dir, config_file, checkpoint_file)
