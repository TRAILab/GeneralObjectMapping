"""

A CO3D dataset loader.

"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os

import numpy as np
import open3d as o3d
import torch
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection

from src.dataset.co3d.co3d_helper.co3d_simple import co3d
from src.dataset.co3d.general_dataset import GeneralDataset


def tensor_to_ply(tensor, save_path):
    """
    Converts a torch tensor with shape (N, 3) to an Open3D point cloud and saves it as a PLY file.

    Parameters:
    - tensor: Torch tensor with shape (N, 3) representing points in 3D space.
    - save_path: Full path (including file name) where the PLY file will be saved.
    """
    # Ensure the tensor is on CPU and convert to numpy
    np_points = tensor.cpu().numpy()

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(save_path, pcd)


def co3d_cam_to_opencv_cam(frame_data):
    """
    @ frame_data: frame_data from co3d dataset
    """
    camera = frame_data.camera

    # if box_crop:
    image_size = frame_data.effective_image_size_hw.unsqueeze(0)
    # else:
    #     image_size = frame_data.image_size_hw.unsqueeze(0)

    # cv2.projectPoints(x_world, rvec, tvec, camera_matrix, [])
    R_cw_cv, tvec_cw_cv, camera_matrix_cv = opencv_from_cameras_projection(camera, image_size)

    # choose 1st batch
    R_cw_cv = R_cw_cv[0]
    tvec_cw_cv = tvec_cw_cv[0]
    camera_matrix_cv = camera_matrix_cv[0]

    # change torch to numpy
    R_cw_cv = R_cw_cv.cpu().numpy()
    tvec_cw_cv = tvec_cw_cv.cpu().numpy()
    camera_matrix_cv = camera_matrix_cv.cpu().numpy()

    return R_cw_cv, tvec_cw_cv, camera_matrix_cv


class CO3D(GeneralDataset):
    """
    IO for loading CO3D dataset
    """

    def __init__(self, data_root, category="hydrant", subset_name="manyview_dev_0"):
        """
        Initialize the CO3D dataset.

        Args:
            data_root (str): The root directory of the CO3D dataset.
        """
        super().__init__(data_root)

        self.data_root = data_root

        self.category = category

        self.subset_name = subset_name

        self.load_data()

    def load_data(self):
        """
        Init processing during dataset initialization
        """

        # sequences_dir = os.path.join(self.data_root, self.category)

        # list all folders names under sequences_dir
        # self.scene_names = os.listdir(sequences_dir)

        self.co3d = co3d(self.data_root, self.category, self.subset_name)

    def load_scene_list(self, subset="all"):
        """
        Input:
            - subset: 'all' or 'train' or 'test' [optional]

        Output:
            scene_names
            scene_info

        """

        scene_names = ["106_12648_23157"]

        # construct scene_info
        scene_info = {}

        # Step 1: Get a method to select a subset of scene name from all the names
        scene_info["selected_scene_names"] = scene_names

        scene_info["ins_orders_list_all_scenes"] = [[0]]

        # Step 2: For each scene_name, load its valid frames
        scene_info["scene_ins_frames"] = [range(10)]

        # for each scene_name, load its valid single frames
        scene_info["scene_ins_frames_single"] = [[0]]

        # for each scene_name, for each instance, load information
        scene_info["category_list"] = ["hydrant"]

        return scene_names, scene_info

    def get_frame_num(self, scene_name):
        """
        Get the number of frames in a scene.

        Args:
            scene_name (str): The name of the scene.

        Returns:
            int: The number of frames in the scene.
        """

        # Update: Using official loader to get frame num
        frame_num = self.co3d.get_frame_num_official(scene_name)

        return frame_num

    def load_scene_names(self):
        return self.co3d.sequence_names

    def get_scene_num(self):
        return len(self.load_scene_names())

    def get_frames_by_scene(self, scene_name, N=None):
        """
        Input:
            - N: number of frames to load [optional]
        """

        return self.co3d.get_frame_datas_official(scene_name, N)

    def get_frames_structure_for_inputs(self, scene_name, N=None, resolution=None):
        """
        Step 1: Load original scenes frames from official CO3D dataset loader.

        Step 2: Transform the loaded data into correct sturctures for our inputs.
        """

        frames_co3d = self.get_frames_by_scene(scene_name, N)

        # Construct observation list
        observation_list = []
        for frame in frames_co3d:

            # camera to opencv structure
            R_cw_cv, tvec_cw_cv, camera_matrix_cv = co3d_cam_to_opencv_cam(frame)
            K = camera_matrix_cv

            # Construct a 4x4 matrix T_wc
            T_cw = np.eye(4)
            T_cw[:3, :3] = R_cw_cv
            T_cw[:3, 3] = tvec_cw_cv
            # Construct torch Tensor with float
            T_cw = torch.tensor(T_cw, dtype=torch.float32)

            # DEBUG See the Whole Scene
            mask_points = True
            # DEBUG See the Whole Scene
            pts_world = self.co3d.unproject_points(frame, mask_points=mask_points)  # (N, 3)

            pts_3d_w = pts_world  # unproject from depth information

            T_wc = T_cw.inverse()

            # transform from camera to world; using torch tensor
            if pts_3d_w is None:
                pts_3d_c = None
            else:
                pts_3d_c = torch.matmul(T_cw[:3, :3], pts_3d_w.T).T + T_cw[:3, 3]

            # map to 0-255 inside rgb images, and using int8
            rgb = frame.image_rgb * 255
            rgb = rgb.to(torch.uint8)
            # C, H, W -> H, W, C
            rgb = rgb.permute(1, 2, 0).cpu().numpy()

            depth_map = frame.depth_map
            # C, H, W -> H, W, C
            depth_map = depth_map.permute(1, 2, 0).cpu().numpy().squeeze()  # Shape: (H, W)

            mask_crop = frame.fg_probability > 0.5
            # C, H, W -> H, W, C
            mask_crop = mask_crop.permute(1, 2, 0).cpu().numpy().squeeze()  # Shape: (H, W)

            # Generate a new image type: rgb_mask_cropped
            rgb_mask_cropped = rgb.copy()
            rgb_mask_cropped[~mask_crop] = 0

            observation = {
                "pts_3d_c": pts_3d_c,  # camera
                "pts_3d_w": pts_3d_w,  # world;
                "rgb": rgb,
                "depth": depth_map,
                "mask": mask_crop,
                "depth_image": depth_map,
                "t_wc": T_wc,
                "t_cw": T_cw,
                "sub_id": frame.frame_number.item(),
                # 'obs_id': frame.frame_id, # for visualization as name
                "K": K,
                "frame": frame,
                "rgb_mask_cropped": rgb_mask_cropped,
                "category": self.category,
            }

            observation_list.append(observation)

        return observation_list


if __name__ == "__main__":
    dataset_root = "dataset/co3d_single"

    dataset = CO3D(dataset_root)

    # load frame
    scene_names = dataset.load_scene_names()

    # frames = dataset.get_frames_by_scene(scene_names[0], N=10)

    frames_structure = dataset.get_frames_structure_for_inputs(scene_names[0], N=10)

    os.makedirs("./output/debug/co3d_pts_coordinate", exist_ok=True)
    frame_ids = [0, 1]
    for frame_id in frame_ids:
        frame = frames_structure[frame_id]

        # Save
        tensor_to_ply(frame["pts_3d_c"], f"./output/debug/co3d_pts_coordinate/pts_c_{frame_id}.ply")
        tensor_to_ply(frame["pts_3d_w"], f"./output/debug/co3d_pts_coordinate/pts_w_{frame_id}.ply")

    print("done")
