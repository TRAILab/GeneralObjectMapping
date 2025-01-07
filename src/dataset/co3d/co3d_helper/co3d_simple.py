"""

Simple dataset loader for CO3D dataset.

"""

import json
import os
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2,
)
from pytorch3d.implicitron.models.visualization.render_flyaround import render_flyaround
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.implicitron.tools.vis_utils import (
    get_visdom_connection,
    make_depth_image,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from src.dataset.co3d.co3d_helper.data_types import (
    FrameAnnotation,
    SequenceAnnotation,
    load_dataclass_jgzip,
)


def _subsample_pointcloud(p: Pointclouds, n: int):
    n_points = p.num_points_per_cloud().item()
    if n_points > n:
        # subsample the point cloud in case it is bigger than max_n_points
        subsample_idx = torch.randperm(
            n_points,
            device=p.points_padded().device,
        )[:n]
        p = Pointclouds(
            points=p.points_padded()[:, subsample_idx],
            features=p.features_padded()[:, subsample_idx],
        )
    return p


import open3d as o3d


def preprocess_points(point_cloud_tensor):
    """
    Preprocess the point cloud tensor.

    Input:
        - point_cloud_tensor: torch.tensor, (N, 3)

    Output:
        - point_cloud_final: torch.tensor, (N, 3)
    """

    # Construct an open3d Pointcloud
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_tensor.cpu().numpy())

    point_cloud_o3d_ds = point_cloud_o3d.voxel_down_sample(voxel_size=0.05)
    # point_cloud_cam_filtered, ind = point_cloud_cam_ds.remove_statistical_outlier(nb_neighbors=50, std_ratio=3)

    # extract the largest cluster
    labels = np.array(point_cloud_o3d_ds.cluster_dbscan(eps=0.12, min_points=10))
    non_negative_labels = labels[labels >= 0]

    # if there are no clusters, return
    if len(non_negative_labels) == 0:
        return None

    largest_cluster_label = np.argmax(np.bincount(non_negative_labels))
    largest_cluster_mask = labels == largest_cluster_label
    point_cloud_final = point_cloud_o3d_ds.select_by_index(np.where(largest_cluster_mask)[0])

    # output array
    point_cloud_final_arr = torch.from_numpy(np.asarray(point_cloud_final.points)).float()

    return point_cloud_final_arr


class co3d:
    def __init__(self, dataset_root, category="hydrant", subset_name="manyview_dev_0"):
        """
        Co3d loader
        """
        self.dataset_root = dataset_root
        # self.scenes = os.listdir(dataset_root)

        self.category = category
        self.init_official_api(category, subset_name=subset_name)

        # Load dir of camera poses file
        self.frame_annotations_file = os.path.join(dataset_root, "frame_annotations.jgz")
        self.sequence_annotations_file = os.path.join(dataset_root, "sequence_annotations.jgz")

        self.frame_annotations = None
        self.sequence_annotations = None

        self.sequence_to_annotations = None

    def init_official_api(self, category, box_crop=False, subset_name="manyview_dev_0"):
        """
        Load sequence names, and init dataset using official API
        """
        DATASET_ROOT = self.dataset_root

        with open(os.path.join(DATASET_ROOT, "category_to_subset_name_list.json"), "r") as f:
            category_to_subset_name_list = json.load(f)

        # # get the visdom connection
        # viz = get_visdom_connection()

        # # iterate over the co3d categories
        categories = sorted(list(category_to_subset_name_list.keys()))

        if not category in categories:
            raise ValueError(
                f"Category {category} not found in CO3D. Supported categories are {categories}"
            )

        subset_name_list = category_to_subset_name_list[category]

        """
        Note: manyview_x_x only contains single sequence, and train/test/val all come from the same sequence.

        We consider all sequences besides test
        """
        # subset_name = subset_name
        if subset_name not in subset_name_list:
            print("subset_name not in subset_name_list")
            subset_name = subset_name_list[0]
            print("choose the first subset_name:", subset_name)

            if "test" in subset_name:
                raise ValueError("Test set has no depth information.")

        # obtain the dataset
        expand_args_fields(JsonIndexDatasetMapProviderV2)
        dataset_map = JsonIndexDatasetMapProviderV2(
            category=category,
            subset_name=subset_name,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_JsonIndexDataset_args=DictConfig(
                {"remove_empty_masks": False, "load_point_clouds": True, "box_crop": box_crop}
            ),
            dataset_root=self.dataset_root,
        ).get_dataset_map()

        train_dataset = dataset_map["train"]

        self.sequence_names = list(train_dataset.seq_annots.keys())

        self.dataset = train_dataset

        return

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_dir = os.path.join(self.dataset_root, scene_name)

    def init_frame_annotations(self):
        # if first time loading, cache it
        if self.frame_annotations is None:
            # load a jgz file (gzipped json files)
            print("Init frame annotations ...")
            self.frame_annotations = load_dataclass_jgzip(
                self.frame_annotations_file, List[FrameAnnotation]
            )
            print("done loading frame annotations")

        # Organize annotations by scene
        sequence_to_annotations = {}

        for frame_annotation in self.frame_annotations:
            sequence_name = frame_annotation.sequence_name

            if sequence_name not in sequence_to_annotations:
                sequence_to_annotations[sequence_name] = []

            sequence_to_annotations[sequence_name].append(frame_annotation)

        self.sequence_to_annotations = sequence_to_annotations

        print("done organizing frame annotations")

    def init_sequence_annotations(self):
        # if first time loading, cache it
        if self.sequence_annotations is None:
            # load a jgz file (gzipped json files)
            self.sequence_annotations = load_dataclass_jgzip(
                self.sequence_annotations_file, List[SequenceAnnotation]
            )

        print("done loading sequence annotations")

    def load_frame_annotations_for_sequence(self, sequence_name):
        # if first time loading, cache it
        if self.sequence_to_annotations is None:
            self.init_frame_annotations()

        return self.sequence_to_annotations[sequence_name]

    def load_sequence_annotations(self, sequence_name):
        # if first time loading, cache it
        if self.sequence_annotations is None:
            self.init_sequence_annotations()

        for sequence_annotation in self.sequence_annotations:
            if sequence_annotation.sequence_name == sequence_name:
                return sequence_annotation

        return None

    def preprocess_all_sequences_to_nerfstudio(self):
        """
        Preprocess all sequences of co3d specified by category and subset into nerfstudio-supported formats.
        """

        raise NotImplementedError

    def get_frame_datas_official(self, sequence_name, N=None):
        """
        Get the intrinsics of the camera for a given sequence

        using official API.

        N: number of frames to load [optional]
        """

        """
        Processing a specific sequence is not implemented yet.

        Please see process_all_sequences
        """
        # raise NotImplementedError

        # Check if already init
        if self.dataset is None:
            self.init_official_api(self.category)

        # Check if this sequence name inside self.sequence_names
        if sequence_name not in self.sequence_names:
            print("Sequence name not in self.sequence_names")
            return None

        # Here we get a valid scene name from official API after choosing a subset

        # extract data from self.dataset
        train_dataset = self.dataset

        frame_ids_scene = list(self.get_frame_ids_official(sequence_name))
        frame_idx_list = [x[2] for x in frame_ids_scene]
        if N is not None:
            # randomly sample N frames
            frame_idx_list = np.random.choice(frame_idx_list, N, replace=False)

        frame_datas = [train_dataset[i] for i in frame_idx_list]

        return frame_datas

    def get_frame_ids_official(self, sequence_name):
        """
        The frame ids correspond to the sequence name.
        Note all sequences share common idxs starting from 0.
        """
        frame_ids = list(self.dataset.sequence_frames_in_order(sequence_name))

        return frame_ids

    def get_frame_num_official(self, sequence_name):
        return len(self.get_frame_ids_official(sequence_name))

    def unproject_points(self, frame_data, mask_points=True):
        """
        Output:
            - pytorch3d Pointcloud in world coordinate -> torch.tensor

        """

        # Note: the pointcloud is inside world coordinate!
        point_cloud_w = get_rgbd_point_cloud(
            frame_data.camera,
            frame_data.image_rgb.unsqueeze(0),
            frame_data.depth_map.unsqueeze(0),
            (frame_data.fg_probability > 0.5).float().unsqueeze(0) if mask_points else None,
        )

        # change pytorch3d pointcloud to torch tensor
        point_cloud_tensor = point_cloud_w.points_packed()

        # Begin Preprocessing: filtering and downsampling process
        point_cloud_final_world = preprocess_points(point_cloud_tensor)

        return point_cloud_final_world


if __name__ == "__main__":
    # debug
    dataset_root = "data/CO3D"
    dataset = co3d(dataset_root)
