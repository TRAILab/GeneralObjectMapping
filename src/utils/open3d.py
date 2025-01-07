"""
Utils for Open3D visualization
"""

import os

import open3d as o3d


def set_view_mat(vis, mat):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()

    # world to eye
    T = mat

    # T = np.array([[np.cos(yaw), -np.sin(yaw), 0., 0.],
    #               [np.sin(yaw), np.cos(yaw)*np.cos(theta), -np.sin(theta), 0.],
    #               [0., np.sin(theta), np.cos(theta), dist],
    #               [0., 0., 0., 1.]])

    # debug: output current camera pose with convert_to_pinhole_camera_parameters
    # print('before:', cam.extrinsic)

    # init a random view from a SE(3) for debugging
    # T_random = np.array([[0.9999, 0.0001, 0.0134, -100.],
    #                         [0.0001, 1., 0.0001, 0.],
    #                         [-0.0134, 0.0001, 0.9999, 0.],
    #                         [0., 0., 0., 1.]])
    # cam.extrinsic = T_random

    cam.extrinsic = T

    # print('after:', cam.extrinsic)

    vis_ctr.convert_from_pinhole_camera_parameters(cam)


def load_open3d_view(vis, view_file_name="./view_file.json"):
    load_view_point = os.path.isfile(view_file_name)
    ctr = vis.get_view_control()
    # ctr.rotate(10.0, 0.0)
    if load_view_point:
        param = o3d.io.read_pinhole_camera_parameters(view_file_name)
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        print("Fail to load view from:", view_file_name)
