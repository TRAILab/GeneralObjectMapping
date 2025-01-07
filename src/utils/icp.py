import numpy as np
import open3d as o3d

from src.data_association.association_utils import Object, Observation
from src.data_association.init_pose_estimator import PoseEstimator

"""
Utils IO
"""


def construct_object_from_frame(frame, category):
    """
    Need attributes:

        name_class
        obj_id

        point_cloud
        pcd_center

        bbox_length, bbox_height, bbox_width

    """

    frame.masks = np.stack([frame.mask])

    # label_id = name_to_coco_id(category)   # some categories have no corresponding label
    label_id = 0
    frame.labels = [label_id]  # TODO: put chair label here

    frame.scores = [1.0]

    # idx is to select the bbox from a group of bboxes
    idx = 0
    obs = Observation(frame, idx)

    obj_id = 0
    object = Object(obj_id, obs)

    object.name_class = category

    return object


def estimate_bbox_size(point_cloud):
    """
    Estimate the bbox size of the object.

    Input:
        - point_cloud: (N, 3)
    """

    # Use Open3D Pointcloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(point_cloud)

    bbox_3d = pcd_o3d.get_axis_aligned_bounding_box()

    bbox_dim = bbox_3d.get_max_bound() - bbox_3d.get_min_bound()

    bbox_length = bbox_dim[0]  # along x axis
    bbox_height = bbox_dim[1]  # along y axis
    bbox_width = bbox_dim[2]  # along z axis

    # Note: We identify Length/Height/Width information according to the object's orientation in the world
    # This is not a good practice;
    # We should treat the three dims as the same, since we have no prior assumption about the world's coordinate orientation.
    return bbox_length, bbox_height, bbox_width


class ICPMatcher:
    def __init__(self, source="ours"):

        # load a model
        self.estimator = PoseEstimator(source=source)

    def match_points(self, pts, category, rotation_keep_yaw=False):
        """
        Update A simple version:

        Take input pointcloud, directly match a mesh with given category to this pointcloud.

        @ rotation_keep_yaw: close for CO3D to get complete ICP rotation.
        """

        thresh = 0

        pts_np = pts
        if not isinstance(pts_np, np.ndarray):
            pts_np = pts_np.cpu().numpy()
        pcd_center = pts_np.mean(0)
        bbox_length, bbox_height, bbox_width = estimate_bbox_size(pts_np)

        # get an open3d point
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts_np)

        result = self.estimator.estimate_direct(
            category,
            bbox_length,
            bbox_height,
            bbox_width,
            point_cloud,
            pcd_center,
            rot_axis="y",
            thresh=thresh,
            obj_id=-1,
            rotation_keep_yaw=rotation_keep_yaw,
        )

        success = result["success"]
        # except:
        #     success = False

        if success:
            t_world_object_init = result["estimated_pose"]
            s_world_object_init = result["estimated_scale"]  # (3,1)

            # diag
            mat_s_world_object_init = np.diag(s_world_object_init)

            ts_world_object_init = t_world_object_init
            ts_world_object_init[0:3, 0:3] = (
                ts_world_object_init[0:3, 0:3] @ mat_s_world_object_init
            )

        else:
            ts_world_object_init = None

            print("! Fail to init a pose for this object.")

        return ts_world_object_init

    def match(self, det, pts, category):
        # Step1: construct an object
        frame = det.frame
        obj = construct_object_from_frame(frame, category)

        try:
            thresh = 0

            # Note z axis is the Up axis of ply meshes of object models
            success = self.estimator.estimate(obj, rot_axis="y", thresh=thresh)
        except:
            success = False

        if success:
            t_world_object_init = obj.estimated_pose
            s_world_object_init = obj.estimated_scale  # (3,1)

            # diag
            mat_s_world_object_init = np.diag(s_world_object_init)

            ts_world_object_init = t_world_object_init
            ts_world_object_init[0:3, 0:3] = (
                ts_world_object_init[0:3, 0:3] @ mat_s_world_object_init
            )

        else:
            ts_world_object_init = None

            print("! Fail to init a pose for this object.")

        return ts_world_object_init
