"""

Testing initializer by loading two examples of chairs and tables
Visualize the result in 3D.

"""

import os

# add dir .., insert to 0
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy


def get_diag_length(pts):

    # get the min and max of x, y, z
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    min_z = np.min(pts[:, 2])
    max_z = np.max(pts[:, 2])

    # calculate the diagnal length
    diag_len = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)

    return diag_len


def test_simple_init():
    category = "chair"

    # output/debug/initializer/chair(table)/pts_obj.npy
    pts_obj = np.load("output/debug/initializer/{}/pts_obj.npy".format(category))

    # pts_3d_c.npy
    pts_3d_c = np.load("output/debug/initializer/{}/pts_3d_c.npy".format(category))

    from optimizer.initializer import init_object_pose_from_camera

    # Step one: get an initialization for translation and scale
    # translation: center diff between two point cloud
    # scale: diag length of pts_3d_c / diag length of pts_obj
    t_cam_obj_init = get_scale_and_translation(pts_obj, pts_3d_c)

    input = {"pts_obj": pts_obj, "pts_observation": pts_3d_c, "init_t_co": t_cam_obj_init}
    method_init_pose = "icp"

    debug_mode = True

    if not debug_mode:
        t_co = init_object_pose_from_camera(input, method=method_init_pose)

        # debug: visualize before rotation refinements
        # t_co = t_cam_obj_init

        # visualize
        visualize_results(pts_obj, pts_3d_c, t_co)

    else:
        """
        Debug mode: see all the results of this initilaization icp matching
        """
        result = init_object_pose_from_camera(input, method=method_init_pose, debug_mode=True)

        # choose the best one
        fitness_list = [r[0] for r in result]
        # rank from high to low
        rank = np.argsort(fitness_list)[::-1]

        # visualize the init, and converge, and corresponding score for each case
        for id, re in enumerate(result):
            # result.append((fitness, inlier_rmse, trans_icp, init_t_co))
            fitness = re[0]
            inlier_rmse = re[1]
            trans_icp = re[2]
            init_t_co = re[3]

            # find the place in the rank
            rk_place = np.where(rank == id)[0][0]

            # visualize init
            visualize_results(
                pts_obj,
                pts_3d_c,
                init_t_co,
                save_name="output/debug/initializer/{}/init_rk{}_id{}_fit{:.4f}_rmse{:.4f}.png".format(
                    category, rk_place, id, fitness, inlier_rmse
                ),
                show_init=False,
            )

            visualize_results(
                pts_obj,
                pts_3d_c,
                trans_icp,
                save_name="output/debug/initializer/{}/icp_rk{}_id{}_fit{:.4f}_rmse{:.4f}.png".format(
                    category, rk_place, id, fitness, inlier_rmse
                ),
                show_init=False,
            )


def get_scale_and_translation(pts_obj, pts_3d_c):
    # scale
    s_1 = get_diag_length(pts_3d_c)
    s_2 = get_diag_length(pts_obj)
    s = s_1 / s_2

    # deep copy
    pts_obj_scaled = copy.deepcopy(pts_obj) * s

    # translation
    # get the center of pts_obj and pts_3d_c
    center_1 = np.mean(pts_3d_c, axis=0)
    center_2 = np.mean(pts_obj_scaled, axis=0)

    t = center_1 - center_2

    # combine it as a transformation matrix, 4x4
    t_co = np.eye(4)
    t_co[:3, 3] = t
    # add scale
    t_co[:3, :3] = t_co[:3, :3] * s

    return t_co


def test_global_registration():
    """
    Test method: GLobal registration with scale manually adjustment
    """
    category = "table"

    # output/debug/initializer/chair(table)/pts_obj.npy
    pts_obj = np.load("output/debug/initializer/{}/pts_obj.npy".format(category))

    # pts_3d_c.npy
    pts_3d_c = np.load("output/debug/initializer/{}/pts_3d_c.npy".format(category))

    """
    An extra step to align the scale of pts_obj, to pts_3d_c
    """
    # scale pts_obj with s, so that the scale of pts_obj is close to pts_3d_c

    # calculate the diagnal length of pts_3d_c
    s_1 = get_diag_length(pts_3d_c)
    s_2 = get_diag_length(pts_obj)

    s = s_1 / s_2

    pts_obj = pts_obj * s

    # calculate the diagnal length of pts_obj

    # match
    # import optimizer/initializer.py
    from optimizer.initializer import global_registration

    t_co = global_registration(pts_obj, pts_3d_c)

    # visualize
    visualize_results(pts_obj, pts_3d_c, t_co)


def visualize_results(pts_obj, pts_3d_c, t_co, save_name=None, vis=None, show_init=True):
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    # before matching
    pts_obj_o3d = o3d.geometry.PointCloud()
    pts_obj_o3d.points = o3d.utility.Vector3dVector(pts_obj)
    pts_obj_o3d.paint_uniform_color([0.0, 0.0, 1.0])  # blue

    pts_3d_c_o3d = o3d.geometry.PointCloud()
    pts_3d_c_o3d.points = o3d.utility.Vector3dVector(pts_3d_c)
    pts_3d_c_o3d.paint_uniform_color([1.0, 0.0, 0.0])  # red

    if show_init:
        vis.add_geometry(pts_obj_o3d)

    vis.add_geometry(pts_3d_c_o3d)

    # vis.run()

    # after matching
    # transform pts_obj_o3d
    import copy

    pts_obj_o3d_trans = copy.deepcopy(pts_obj_o3d)
    pts_obj_o3d_trans.transform(t_co)
    pts_obj_o3d_trans.paint_uniform_color([0.0, 1.0, 1.0])  # other

    vis.add_geometry(pts_obj_o3d_trans)

    if save_name is not None:
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(save_name)
    else:
        vis.run()


if __name__ == "__main__":
    # test_global_registration()

    test_simple_init()
