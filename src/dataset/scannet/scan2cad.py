"""
Load Scan2CAD data.
"""

import json
import os

import numpy as np
import open3d as o3d


class Scan2CAD:

    def __init__(self, root_dir, shapenet_dir) -> None:

        self.root_dir = root_dir
        self.shapenet_dir = shapenet_dir

        self.load_data()

    def load_data(self):
        """
        Load data from root dir
        """

        # data/scannet/scannetv2_val.txt
        # scene_list_txt = os.path.join(self.root_dir, 'scannetv2_val.txt')
        # self.scene_list = open(scene_list_txt).read().splitlines()

        self.annotation_file = os.path.join(self.root_dir, "full_annotations.json")

        """
        Update an efficient management: only load the file when it is first used.

        init_annotation_file()
        """
        self.data = None
        self.data_scene_names = None

    def init_annotation_file(self):
        # load from json
        self.data = json.load(open(self.annotation_file))

        # process data structure from: [{'id_scan', ...}, ...] to {'id_scan': {...}, ...}}]
        self.data_scene_names = {d["id_scan"]: d for d in self.data}

    def load_annotation_of_scene(self, scene_name):
        """
        Load annotation of a scene

        data structure:
            {
                'id_scan': 'scene0000_00',
                'trs',
                'n_aligned_models',
                'aligned_models',
                    {
                        'trs'
                        'bbox'
                        'center'
                        'sym'
                        'id_cad'
                        'catid_cad'
                        'keypoints_cad'
                        'keypoints_scan'
                    }
                'id_alignment'

        """

        if self.data_scene_names is None:
            self.init_annotation_file()

        return self.data_scene_names[scene_name]

    def load_objects_of_scene(self, scene_name):
        """
        Load objects of a scene

        data structure:
            {
                'trs'
                'bbox'
                'center'
                'sym'
                'id_cad'
                'catid_cad'
                'keypoints_cad'
                'keypoints_scan'
            }
        """

        return self.load_annotation_of_scene(scene_name)["aligned_models"]

    def ValidMeshCheck(self, mesh_name):
        invalid_mesh_names = ["e7580c72525b4bb1cc786970133d7717"]

        for i_name in invalid_mesh_names:
            if i_name in mesh_name:
                return False

        return True

    def load_shape(self, cat_id, shape_id):
        """

        Load shape from ShapeNet dataset

        @ cat_id: category id
        @ shape_id: shape id

        """

        model_dir = os.path.join(self.shapenet_dir, cat_id, shape_id, "models")

        # load mesh
        mesh_file = os.path.join(model_dir, "model_normalized.obj")

        if os.path.exists(mesh_file) and self.ValidMeshCheck(
            mesh_file
        ):  # There is one mesh that will get error
            mesh = o3d.io.read_triangle_mesh(mesh_file)
        else:
            mesh = "FailToLoad"
            raise ValueError("Mesh fails to load:", mesh_file)

        return mesh

    def load_ind_2_scannet(self, scene_name):
        """
        For GT Mask association, we match the GT instance mask with the obj_id in Scan2CAD.

        The association idx is generated with files:
            preprocess/scannet_instance_matcher.py
        """

        ind_file = os.path.join(
            self.root_dir, "indices_to_scannet", scene_name + "_ind_2_scannet.npy"
        )
        ind_2_scannet = np.load(ind_file)

        return ind_2_scannet

    def save_ind_2_scannet(self, scene_name, ind_2_scannet):
        """
        Save the association idx to file
        """

        ind_file = os.path.join(
            self.root_dir, "indices_to_scannet", scene_name + "_ind_2_scannet.npy"
        )
        np.save(ind_file, ind_2_scannet)
