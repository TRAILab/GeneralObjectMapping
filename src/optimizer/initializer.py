import numpy as np
import open3d as o3d
from open3d import pipelines as o3d_p


def get_unique_rotations():
    # all possible ones
    unique_rotations = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                rotation = (i, j, k)
                unique_rotations.append(rotation)

    return unique_rotations


def get_four_rotations():
    # four rotataions rotating around Z axis for 0,90,180,270 degs
    rotations = []
    for k in range(4):
        if k == 0:
            continue  # skip the first one since it has been contained

        rotations.append((0, 0, k))

    return rotations


def rotate_pose(pose, rotation):
    """
    pose: SE(3), 4x4

    rotation: (i, j, k), 3x1; number of 90 degree rotations along each axis
    """
    i, j, k = rotation

    # get a rotation matrix, which rotates around X axis for angle i*pi/2
    # use a library
    from scipy.spatial.transform import Rotation as R

    r = R.from_euler("x", i * np.pi / 2, degrees=False)
    rot_mat = r.as_matrix()

    # get a rotation matrix, which rotates around Y axis for angle j*pi/2
    r = R.from_euler("y", j * np.pi / 2, degrees=False)
    rot_mat = np.matmul(rot_mat, r.as_matrix())

    # get a rotation matrix, which rotates around Z axis for angle k*pi/2
    r = R.from_euler("z", k * np.pi / 2, degrees=False)
    rot_mat = np.matmul(rot_mat, r.as_matrix())

    # apply rotation; note pose is 4x4, rot_mat is 3x3
    pose_rot = np.matmul(pose[:3, :3], rot_mat)
    pose_rot = np.concatenate((pose_rot, pose[:3, 3:]), axis=1)
    pose_rot = np.concatenate((pose_rot, np.array([[0, 0, 0, 1]])), axis=0)

    return pose_rot


def init_object_pose_from_camera(
    input, method="icp", debug_mode=False, rotation_search_method="all"
):
    """
    Method 1: init from GT + a coarse deepsdf norm transform
            note this method has a problem for Shap-E: the rotation is not considered.

            @input: 'det'

    Method 2: Global ICP registration between observation and Shap-E

            @input: input = {
                'pts_obj': pts_obj,
                'pts_observation': pts_3d_c,
                'init_t_co'
            }

            @debug_mode: Also output all the results for debugging.

    Method 3: 'global' registration between observation and Shap-E

    @ output: t_co

    @ rotation_search_method: effective for icp,
        'all': consider all 64 rotations,
        'four': only four rotations around Z axis

    """
    if method == "gt_norm":
        det = input["det"]

        if "t_cam_deepsdf" in det:
            t_cam_obj_init = (
                det.t_cam_deepsdf
            )  # TODO: note the init deepsdf norm transform is coarse.
        else:
            t_cam_obj_init = det.T_cam_obj

        t_co = t_cam_obj_init
        # t_oc = np.linalg.inv(t_cam_obj_init)

    elif method == "icp":
        raise NotImplementedError

    elif method == "global":
        # global_registration
        pts_obj = input["pts_obj"]
        pts_3d_c = input["pts_observation"]

        t_co = global_registration(pts_obj, pts_3d_c)

    # copy the data and return
    t_co = np.copy(t_co)
    return t_co


def global_registration(pts1: np.array, pts2: np.array, voxel_size=0.05):
    """
    @ return T: 4x4 transformation matrix, pts2 = T * pts1
    """
    # Convert numpy arrays to open3d point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    # Downsample the point clouds
    pcd1_down = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd1_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd2_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Apply RANSAC based global registration
    # Compute FPFH features
    fpfh1 = o3d_p.registration.compute_fpfh_feature(
        pcd1_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30)
    )
    fpfh2 = o3d_p.registration.compute_fpfh_feature(
        pcd2_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30)
    )

    # Apply RANSAC based global registration
    distance_threshold = voxel_size * 1.5

    result_ransac = o3d_p.registration.registration_ransac_based_on_feature_matching(
        pcd1_down,
        pcd2_down,
        fpfh1,
        fpfh2,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d_p.registration.TransformationEstimationPointToPoint(True),
        ransac_n=4,
        checkers=[
            o3d_p.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d_p.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d_p.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    # Get the transformation matrix
    T = result_ransac.transformation

    return T
