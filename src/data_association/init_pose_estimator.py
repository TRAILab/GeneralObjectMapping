##########################
#
# This file is part of https://github.com/TRAILab/UncertainShapePose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import copy
import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from src.data_association.scan_matcher import registration


class ObjectModel:
    def __init__(self, name_class, file):
        self.name_class = name_class
        self.point_cloud = o3d.io.read_point_cloud(file)
        self.point_cloud.voxel_down_sample(voxel_size=0.02)
        self.point_cloud.paint_uniform_color([0, 0, 1])

        # Coordinate Align for Shap-E NeRF
        # Original Mesh is not directly aligned with Shap-E NeRF BBox Frame (The Estimated t_world_bbox)
        # print("Align coordinate to Shap-E NeRF ...")
        self.point_cloud = self.align_coordinate_to_NeRF(self.point_cloud)

        self.dim = self.point_cloud.get_max_bound() - self.point_cloud.get_min_bound()
        self.length = self.dim[0]
        self.height = self.dim[1]
        self.width = self.dim[2]

    def get_transformed_model(self, scale=[1, 1, 1], transform=np.eye(4)):
        transformed_pcd = copy.copy(self.point_cloud)
        transformed_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(transformed_pcd.points) * scale
        )
        transformed_pcd.transform(transform)
        return transformed_pcd

    def visualize(self, scale=[1, 1, 1], transform=np.eye(4)):
        o3d.visualization.draw_geometries([self.get_transformed_model(scale, transform)])

    def align_coordinate_to_NeRF(self, point_cloud):
        # Rotate around X axis for -90 deg
        rotation_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(np.radians(-90)), -np.sin(np.radians(-90)), 0],
                [0, np.sin(np.radians(-90)), np.cos(np.radians(-90)), 0],
                [0, 0, 0, 1],
            ]
        )
        rotated_point_cloud = point_cloud.transform(rotation_matrix)

        return rotated_point_cloud


class ModelLibrary:
    """
    Information: Object Models

        How to get an object model:

            The object model, as .ply file, is generated from Shap-E NeRF.

            Please check text-to-3D generation process: test/test_prior_multi_samples.py
    """

    def __init__(self, source="ours"):
        self.path = os.path.dirname(__file__) + "/../../"
        self.models = {}

        self.source = source

        if source != "ours":
            # if specific another source
            objects_path = os.path.join(self.path, source)
        else:
            objects_path = self.path

        for file in os.listdir(objects_path):
            # cat_source.ply
            # if file.endswith(source+".ply"):
            #     name_class = self.parse_name_class(file.split("_")[0])
            #     self.models[name_class] = ObjectModel(name_class, self.path+file)

            if file.endswith(".ply"):
                cat_name = file.split(".")[0]
                name_class = self.parse_name_class(cat_name)
                self.models[name_class] = ObjectModel(name_class, os.path.join(objects_path, file))

    def parse_name_class(self, name_class):
        if name_class in ["car", "truck", "bus"]:
            return "car"
        elif name_class in ["chair"]:
            return "chair"
        elif name_class in ["bowl"]:
            return "bowl"
        elif name_class in ["bottle"]:
            return "bottle"
        elif name_class in ["table", "dining table"]:
            return "table"
        elif name_class in ["sofa", "couch"]:
            return "sofa"
        else:
            return name_class

    def has_model(self, name_class):
        return self.parse_name_class(name_class) in self.models.keys()

    def get_model(self, name_class):
        return self.models[self.parse_name_class(name_class)]

    def get_transformed_model(self, name_class, scale=[1, 1, 1], transform=np.eye(4)):
        return self.models[self.parse_name_class(name_class)].get_transformed_model(
            scale, transform
        )

    def get_transformed_model(self, object):
        return self.models[self.parse_name_class(object.name_class)].get_transformed_model(
            object.estimated_scale, object.estimated_pose
        )

    def visualize(self, name_class, scale=[1, 1, 1], transform=np.eye(4)):
        self.models[self.parse_name_class(name_class)].visualize(scale, transform)


class PoseEstimator:
    def __init__(self, source="ours"):
        self.model_library = ModelLibrary(source=source)

    def estimate_direct(
        self,
        category,
        bbox_length,
        bbox_height,
        bbox_width,
        point_cloud,
        pcd_center,
        rot_axis="y",
        thresh=0.25,
        obj_id=-1,
        rotation_keep_yaw=True,
    ):
        """
        Direct version: explicitly take in params, instead of a complex class

        @ rot_axis: the rotation axis of the Mesh.

        @ rotation_keep_yaw: by Default Open it for ScanNet. Please close it for CO3D.
        """
        object_class = category

        if not self.model_library.has_model(object_class):
            print("Object type:{} not in model library.".format(object_class))
            # return False

            raise ValueError(f"Object type:{object_class} not in model library.")

        print("Estimating pose for object {}: {}".format(obj_id, object_class))

        model = self.model_library.get_model(object_class)

        est_scale = self.estimate_scale_direct(model, bbox_length, bbox_height, bbox_width)
        est_offset = pcd_center

        N_candidate = 18
        rot_candidates = [i * 2 * np.pi / N_candidate for i in range(N_candidate)]

        yaw_best = 0
        trans_best = np.zeros(3)
        rmse_best = np.inf
        fitness_best = -np.inf
        candidate_best = -1
        transform_best = np.eye(4)

        for j in range(N_candidate):
            candidate_rot = R.from_euler(rot_axis, rot_candidates[j]).as_matrix()

            T_candidate = np.eye(4)
            T_candidate[0:3, 0:3] = candidate_rot
            T_candidate[0:3, 3] = est_offset

            transformed_model = copy.copy(model.point_cloud)
            transformed_model.points = o3d.utility.Vector3dVector(
                np.asarray(transformed_model.points) * est_scale
            )

            debug = False
            if debug:
                print("DEBUG: Debug Mode for ICP matching ...")
                debug_output_dir = "./output/icp_debug/"
                os.makedirs(debug_output_dir, exist_ok=True)

                # Debug mode: Output the initial guess
                # Save pointcloud 1: model.point_cloud
                o3d.io.write_point_cloud(
                    os.path.join(debug_output_dir, "model.ply"), model.point_cloud
                )
                # Save scaled model
                o3d.io.write_point_cloud(
                    os.path.join(debug_output_dir, "scaled_model.ply"), transformed_model
                )
                # Save input pointcloud
                o3d.io.write_point_cloud(
                    os.path.join(debug_output_dir, "target_point_cloud.ply"), point_cloud
                )
                # Save initial guess; transform the scaled model with initial guess
                transformed_model_guess = copy.copy(transformed_model)
                transformed_model_guess.transform(T_candidate)
                o3d.io.write_point_cloud(
                    os.path.join(debug_output_dir, f"model_it0_cadidate_{j}.ply"),
                    transformed_model_guess,
                )

            scaled_model_size_norm = np.linalg.norm(
                np.asarray(transformed_model.get_max_bound())
                - np.asarray(transformed_model.get_min_bound())
            )
            max_correspondence_distance = 0.1 * scaled_model_size_norm
            fitness, inlier_rmse, transform_icp = registration(
                transformed_model, point_cloud, max_correspondence_distance, T_candidate, False
            )
            rot_icp = copy.copy(transform_icp[0:3, 0:3])
            euler_icp = R.from_matrix(rot_icp).as_euler("yxz")
            trans_icp = copy.copy(transform_icp[0:3, 3])

            if debug:
                # Output the transformed model
                transformed_model_icp = copy.copy(transformed_model)
                transformed_model_icp.transform(transform_icp)
                o3d.io.write_point_cloud(
                    os.path.join(debug_output_dir, f"model_it_final_cadidate_{j}.ply"),
                    transformed_model_icp,
                )

            # print("candiate:", j)
            # print("fitness: ", fitness)
            # print("inlier_rmse: ", inlier_rmse)
            # print("euler_icp: ", euler_icp)

            if fitness > fitness_best:
                rot_order = "yxz"
                axis_id = rot_order.find(rot_axis)
                yaw_best = euler_icp[axis_id]
                trans_best = trans_icp
                fitness_best = fitness
                candidate_best = j
                transform_best = transform_icp

        print("    Best candidate:", candidate_best)
        print("    Best fitness:", fitness_best, end=", ")
        print("yaw:", yaw_best, end=", ")
        print("trans:", trans_best, end=", ")
        print("scale:", est_scale)

        # Debug: Let rotation is 0
        # print('DEBUG: Rotation=0')
        # print('DEBUG: Rotation=0')
        # print('DEBUG: Rotation=0')
        # yaw_best = 0

        # Whether to only consider yaw angle
        # Note, for CO3D dataset, we should keep the good icp result.
        if rotation_keep_yaw:
            T_best = np.eye(4)
            T_best[0:3, 0:3] = R.from_euler(rot_axis, yaw_best).as_matrix()
            T_best[0:3, 3] = trans_best
        else:
            T_best = copy.copy(transform_best)

        # self.visualize(model, object, est_scale, T_best)

        # DEBUG : Visualize it
        # import datetime
        # time_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # print('Debug ICP Result, save to:', time_prefix)
        # self.visualize(model, object, est_scale, T_best,
        #         save_dir = f'./output/open3d_vis/start_search_{time_prefix}')

        if fitness_best > thresh:
            output = {"success": True, "estimated_scale": est_scale, "estimated_pose": T_best}

        else:
            output = {"success": False, "estimated_scale": None, "estimated_pose": None}

        return output

    def estimate(self, object, rot_axis="y", thresh=0.25):
        """
        @ rot_axis: the rotation axis of the Mesh.
        """
        object_class = object.name_class

        if not self.model_library.has_model(object_class):
            print("Object type:{} not in model library.".format(object_class))
            # return False

            raise ValueError(f"Object type:{object_class} not in model library.")

        print("Estimating pose for object {}: {}".format(object.obj_id, object_class))

        model = self.model_library.get_model(object_class)

        est_scale = self.estimate_scale(model, object)
        est_offset = object.pcd_center

        N_candidate = 18
        rot_candidates = [i * 2 * np.pi / N_candidate for i in range(N_candidate)]

        yaw_best = 0
        trans_best = np.zeros(3)
        rmse_best = np.inf
        fitness_best = -np.inf

        for j in range(N_candidate):
            candidate_rot = R.from_euler(rot_axis, rot_candidates[j]).as_matrix()

            T_candidate = np.eye(4)
            T_candidate[0:3, 0:3] = candidate_rot
            T_candidate[0:3, 3] = est_offset

            transformed_model = copy.copy(model.point_cloud)
            transformed_model.points = o3d.utility.Vector3dVector(
                np.asarray(transformed_model.points) * est_scale
            )

            fitness, inlier_rmse, trans_icp = registration(
                transformed_model, object.point_cloud, 0.1, T_candidate, False
            )
            rot_icp = copy.copy(trans_icp[0:3, 0:3])
            euler_icp = R.from_matrix(rot_icp).as_euler("yxz")
            trans_icp = copy.copy(trans_icp[0:3, 3])

            # print("candiate:", j)
            # print("fitness: ", fitness)
            # print("inlier_rmse: ", inlier_rmse)
            # print("euler_icp: ", euler_icp)

            if fitness > fitness_best:
                rot_order = "yxz"
                axis_id = rot_order.find(rot_axis)
                yaw_best = euler_icp[axis_id]
                trans_best = trans_icp
                fitness_best = fitness

        print("    Best fitness:", fitness_best, end=", ")
        print("yaw:", yaw_best, end=", ")
        print("trans:", trans_best, end=", ")
        print("scale:", est_scale)

        # Debug: Let rotation is 0
        # print('DEBUG: Rotation=0')
        # print('DEBUG: Rotation=0')
        # print('DEBUG: Rotation=0')
        # yaw_best = 0

        T_best = np.eye(4)
        T_best[0:3, 0:3] = R.from_euler(rot_axis, yaw_best).as_matrix()
        T_best[0:3, 3] = trans_best

        # self.visualize(model, object, est_scale, T_best)

        # DEBUG : Visualize it
        # import datetime
        # time_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # print('Debug ICP Result, save to:', time_prefix)
        # self.visualize(model, object, est_scale, T_best,
        #         save_dir = f'./output/open3d_vis/start_search_{time_prefix}')

        if fitness_best > thresh:
            object.estimated_scale = est_scale
            object.estimated_pose = T_best
            return True

        return False

    def estimate_scale_direct(self, model, bbox_length, bbox_height, bbox_width):
        if model.name_class in ["table", "bed", "bookshelf", "bathtub"]:
            return np.array(
                [bbox_length / model.length, bbox_height / model.height, bbox_width / model.width]
            )
        else:
            return bbox_height / model.height * np.ones(3)

    def estimate_scale(self, model, object):
        if model.name_class in ["table", "bed", "bookshelf", "bathtub"]:
            return np.array(
                [
                    object.bbox_length / model.length,
                    object.bbox_height / model.height,
                    object.bbox_width / model.width,
                ]
            )
        else:
            return object.bbox_height / model.height * np.ones(3)

    def visualize(
        self, model, object, scale=[1, 1, 1], transform=np.eye(4), save_dir="./output/open3d_vis"
    ):

        # # create a visualizer object
        # app = o3d.visualization.gui.Application.instance
        # app.initialize()

        # vis = o3d.visualization.O3DVisualizer()
        # vis.show_settings = True

        # vis.add_geometry("MODEL", model.get_transformed_model(scale, transform))
        # vis.add_geometry("OBJECT", object.point_cloud)

        # vis.reset_camera_to_default()

        # visualize the scene
        # app.add_window(vis)
        # app.run()

        # Save To Ply
        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Get transformed model and point cloud
        transformed_model = model.get_transformed_model(scale, transform)
        point_cloud = object.point_cloud

        # Save transformed model and point cloud to local directory
        o3d.io.write_point_cloud(os.path.join(save_dir, "transformed_model.ply"), transformed_model)
        o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud.ply"), point_cloud)

        print("Save ply to", save_dir)


if __name__ == "__main__":
    model_library = ModelLibrary("ours")
    print(model_library.models.keys())
    for key in model_library.models.keys():
        model_library.visualize(key)
