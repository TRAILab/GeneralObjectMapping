"""
Visualization scripts for the result of optimization.
"""

import glob
import os

import imageio
import numpy as np
import open3d as o3d
import torch

from src.utils.data_trans import trimesh_to_open3d
from src.utils.open3d import load_open3d_view, set_view_mat


def visualize_results(history, weights, save_dir="./output/"):
    """
    Draw a matplotlib to show loss, latent norm changing w.r.t iterations.

    @ weights: {
    'w_2d': 1.,
    }
        @ w_2d: the weight of 2D loss in the total loss.
    """

    loss_list = history["loss"]
    latent_list = history["latent"]

    import matplotlib.pyplot as plt

    # plot loss and latent norm in one figure, with two axes
    plt.figure()

    # plot loss with two axes
    axes = plt.gca()
    axes.plot(loss_list, "b")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Loss")

    # plot latent norm with another axes
    axes = axes.twinx()
    latent_norm_list = [torch.norm(latent).item() for latent in latent_list]
    axes.plot(latent_norm_list, "r")
    axes.set_ylabel("Latent Norm")

    # add legend
    plt.legend(["Loss", "Latent Norm"])

    # plt.show()

    # save to output
    plt.savefig(save_dir + "/loss_latent_norm.png")

    # Further plot loss_2d, loss_3d, loss in one figure
    loss_rgb_list = history["loss_rgb"]
    loss_3d_list = history["loss_3d"]
    loss_depth_list = history["loss_depth"]

    w_2d = weights["w_2d"]
    w_depth = weights["w_depth"]

    loss_2d_list_scaled = [loss_2d * w_2d for loss_2d in loss_rgb_list]
    loss_depth_list_scaled = [loss_depth * w_depth for loss_depth in loss_depth_list]

    plt.figure()

    # plot loss with two axes
    axes = plt.gca()
    axes.plot(loss_list, "--", label="Loss")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Loss")

    # plot latent norm with another axes
    axes = axes.twinx()
    axes.plot(loss_2d_list_scaled, "r", label="Loss RGB (scaled)")
    axes.plot(loss_3d_list, "b", label="Loss 3D")
    axes.plot(loss_depth_list_scaled, "y", label="Loss Depth (scaled)")
    if "loss_3d_gt" in history:
        loss_3d_gt_list = history["loss_3d_gt"]
        axes.plot(loss_3d_gt_list, "b--", label="Loss 3D GT")
    if "loss_zero_norm" in history:
        loss_zero_norm_list = history["loss_zero_norm"]
        loss_zero_norm_list_scaled = [
            loss_zero_norm * weights["w_zero_norm"] for loss_zero_norm in loss_zero_norm_list
        ]
        axes.plot(loss_zero_norm_list_scaled, "g--", label="Loss Zero Norm (Scaled)")

    axes.set_ylabel("Loss 2D (scaled), 3D")

    # add legend
    # plt.legend(['Loss', 'Loss 2D', 'Loss 3D'])
    plt.legend()

    # plt.show()

    # save to output
    plt.savefig(save_dir + "/loss_2d_3d.png")

    # loss_regularizer if exist
    if "loss_regularizer" in history and weights["w_regularizer"] is not None:
        loss_regularizer_list = history["loss_regularizer"]
        loss_regularizer_list_scaled = [
            loss_regularizer * weights["w_regularizer"]
            for loss_regularizer in loss_regularizer_list
        ]
        axes.plot(loss_regularizer_list_scaled, "g", label="latent Regularizer (scaled)")
        plt.savefig(save_dir + "/loss_2d_3d_regularizer.png")

    ###############
    # plot lr
    ###############
    lr_list = history["lr"]
    plt.figure()
    axes = plt.gca()
    axes.plot(lr_list, "b")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Learning Rate")
    # plt.show()
    plt.savefig(save_dir + "/lr.png")

    ###############
    # plot latent update
    ###############
    latent_update_list = history["latent_update"]

    # 'latent_update' : {
    #     'diffusion_prior': latent_update_diffusion_prior_list,
    #     'loss': latent_update_loss_list,
    # }

    latent_update_diffusion_prior_list = latent_update_list["diffusion_prior"]
    latent_update_loss_list = latent_update_list["loss"]

    plt.figure()
    axes = plt.gca()
    axes.plot(latent_update_diffusion_prior_list, "b", label="Diffusion Prior")
    axes.plot(latent_update_loss_list, "r", label="Loss Grad")

    axes.set_xlabel("Iterations")
    axes.set_ylabel("Latent Update")

    plt.legend()
    plt.savefig(save_dir + "/latent_update.png")

    # use another axis to plot the data after 40
    # their x starts from 40
    start_point = round(len(latent_update_diffusion_prior_list) / 3 * 2)
    x_range = np.arange(start_point, len(latent_update_diffusion_prior_list))

    axes = axes.twinx()

    it_num = len(latent_update_diffusion_prior_list)
    if it_num > 0:
        opt_per_diff = int(len(latent_update_loss_list) / it_num)

        axes.plot(x_range, latent_update_diffusion_prior_list[start_point:], "b")
        axes.plot(x_range, latent_update_loss_list[start_point * opt_per_diff :: opt_per_diff], "r")

        plt.legend()
        plt.savefig(save_dir + "/latent_update_large.png")

    # another plot with only loss prior
    plt.figure()
    axes = plt.gca()
    axes.plot(latent_update_loss_list, "r", label="Loss Grad")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Latent Update")

    plt.legend()
    plt.savefig(save_dir + "/latent_update_loss.png")

    # add plot for pose and scale update
    if "pose_update" in history:
        pose_update_list = history["pose_update"]["pose"]
        scale_update_list = history["pose_update"]["scale"]

        plt.figure()
        axes = plt.gca()
        axes.plot(pose_update_list, "b", label="Pose Update")
        axes.plot(scale_update_list, "r", label="Scale Update")
        axes.set_xlabel("Iterations")
        axes.set_ylabel("Pose and Scale Update")
        plt.legend()
        plt.savefig(save_dir + "/pose_scale_update.png")


def save_meshes_in_mid_iterations(output, shape_model, save_dir="./output/", device="cuda"):
    """
    A function pared with the optimization process above.
    """

    print("save meshes into: ", save_dir)

    latent_its = output["history"]["latent"]

    steps = len(latent_its)

    # if steps == 300:
    #     save_it_list = [0, 50, 100, 200, 299]
    # elif steps == 64:
    #     save_it_list = [0, 10, 20, 30, 63]
    # else:
    #     # save_it_list = [0, int(steps/4), int(steps/2), int(steps*3/4), steps-1]
    #     save_it_list = [0, int(steps/2), steps-1]

    # generate save_it_list, including the first one, and the last converged one
    save_it_list = [0, steps - 1]
    for it in save_it_list:
        if it < 0 or it >= len(latent_its):
            continue
        latent = latent_its[it]

        # latent to gpu
        latent = latent.to(device)

        mesh = shape_model.get_shape_from_latent(latent)

        # shape_model.save_shape(mesh, save_dir+'mesh_it_'+str(it)+'.ply')

        mesh_o3d = trimesh_to_open3d(mesh)

        # transform to world
        mesh_o3d_w = mesh_o3d.transform(output["history"]["pose_bo"][it].numpy())

        # save to ply using open3d
        o3d.io.write_triangle_mesh(save_dir + "mesh_it_" + str(it) + "_w.ply", mesh_o3d_w)

    return


def visualize_bound_areas(vis, t_wo_cur: torch.Tensor, color=np.array([0, 0, 0])):
    # @ t_wo_cur: torch.Tensor
    # color: 0-1

    # get bounding box
    # Define the 8 vertices of the bounding box in object coordinates
    vertices = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    # Transform the vertices to world coordinates
    vertices = (t_wo_cur.detach().cpu().numpy() @ np.hstack((vertices, np.ones((8, 1)))).T).T[:, :3]

    # Define the 12 edges of the bounding box
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    # Define colors for each line (in this case, red)
    colors = [color for i in range(len(edges))]

    # Create a line set from the vertices and edges
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(line_set)

    # further show the coordinate of the NeRF area
    mesh_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    mesh_coordinate.transform(t_wo_cur.detach().cpu().numpy())
    vis.add_geometry(mesh_coordinate)


def visualize_current_status(
    latent,
    t_wo_cur,
    pts_3d_w,
    shape_model,
    vis=None,
    save_ply=True,
    t_wc_vis=None,
    gt_pts_3d_b=None,
    save_name="im.png",
    view_file="./view.json",
    save_dir=None,
    vis_pts=False,
    t_wo_gt=None,
):
    """
    Function used during optimization step, show current 3D status.
    """
    # latent to mesh
    mesh_o = shape_model.get_shape_from_latent(latent)
    # mesh to open3d
    mesh_o_o3d = o3d.geometry.TriangleMesh()
    mesh_o_o3d.vertices = o3d.utility.Vector3dVector(mesh_o.verts)
    mesh_o_o3d.triangles = o3d.utility.Vector3iVector(mesh_o.faces)
    # add color to meshes: vertex_channels, face_channels
    # merge RGB into 3 dimensions : mesh_o.vertex_channels['R'] ...
    vertex_colors = np.stack([mesh_o.vertex_channels[n] for n in ["R", "G", "B"]]).T
    mesh_o_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # mesh_o_o3d.triangle_normals = o3d.utility.Vector3dVector(mesh_o.face_channels['normal'])

    # compute normal
    mesh_o_o3d.compute_vertex_normals()

    # object coordinate to world
    mesh_w_o3d = mesh_o_o3d.transform(t_wo_cur.detach().cpu().numpy())

    # visualize
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    vis.clear_geometries()

    # note the 3d models are plotted in world coordinates
    mesh_vis = mesh_w_o3d

    # mesh_coordinate_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])

    if gt_pts_3d_b is not None:
        # also plot gt_pts_3d_b
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_pts_3d_b.cpu().numpy())
        vis.add_geometry(gt_pcd)

    # vis.add_geometry(mesh_coordinate_world)

    # points to open3d
    if vis_pts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d_w.cpu().numpy())
        # set color to blue
        pcd.paint_uniform_color([0, 0, 1])
        pcd_vis = pcd

        # Whether to visualize the observed points
        # It will occlude the shapes, so temperally cancel this
        vis.add_geometry(pcd_vis)

    vis.add_geometry(mesh_vis)
    # vis.update_geometry()

    """
    DEBUG: Temperally add a DeepSDF Shape
    """
    debug_vis_deepsdf = False
    if debug_vis_deepsdf:
        deepsdf_dir = "dataset/deepsdf/zero_shape_chair.ply"
        mesh_deepsdf = o3d.io.read_triangle_mesh(deepsdf_dir)
        mesh_deepsdf.compute_vertex_normals()
        mesh_deepsdf.transform(t_wo_cur.detach().cpu().numpy())
        if t_wc_vis is not None:
            mesh_deepsdf.transform(t_cw_vis)
        vis.add_geometry(mesh_deepsdf)

    vis_nerf_area = True
    if vis_nerf_area:
        # transform a bounding box in object coordinate area (-1,1)^3, into w coordinate by t_wo
        # use open3d line to connect vertices
        visualize_bound_areas(vis, t_wo_cur, color=np.array([0, 0, 0]))

    if t_wo_gt is not None:
        t_wo_gt = torch.from_numpy(t_wo_gt)
        # visualize it
        visualize_bound_areas(vis, t_wo_gt, color=np.array([1, 0, 0]))

    def save_view(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("view.json", param)
        print("save to view file.")

    vis.register_key_callback(ord("S"), save_view)

    # Set open3d to see from the camera view
    if t_wc_vis is None:
        # view_file = './view_file_deepsdf.json'
        load_open3d_view(vis, view_file)
    else:

        # plot zero coordinate
        b_plot_coordinate = False
        if b_plot_coordinate:
            mesh_coordinate_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[0, 0, 0]
            )
            # mesh_coordinate.transform(t_wc_vis.cpu().numpy())
            vis.add_geometry(mesh_coordinate_cam)

        # set view from t_wc_vis
        t_cw_vis = np.linalg.inv(t_wc_vis)
        set_view_mat(vis, t_cw_vis)

    vis.poll_events()
    vis.update_renderer()

    # save current view into png file
    # if not exist, create it
    save_dir_vis = save_dir
    os.makedirs(save_dir_vis, exist_ok=True)
    vis.capture_screen_image(os.path.join(save_dir_vis, save_name))

    if save_ply:
        print("save ply files into: ", save_dir_vis)
        # also save current ply into disk
        o3d.io.write_point_cloud(save_dir_vis + "/o3d_pts_3d_vis.ply", pcd_vis)
        o3d.io.write_triangle_mesh(save_dir_vis + "/o3d_mesh_vis.ply", mesh_vis)


def Visualize3DInput(det, gt_mesh, init_latent, shape_model, vis=None):
    """
    Debug function
    """

    t_cam_obj = det.T_cam_obj

    # transform it into GT world

    gt_mesh_cam = gt_mesh.transform(t_cam_obj)
    # compute normal
    gt_mesh_cam.compute_vertex_normals()

    pts_c = det.surface_points

    # visualize in 3d for pts and gt mesh
    pts_c_o3d = o3d.geometry.PointCloud()
    pts_c_o3d.points = o3d.utility.Vector3dVector(pts_c)

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    vis.clear_geometries()

    # Check 1: gt mesh and observation in camera frame
    vis.add_geometry(pts_c_o3d)
    # vis.add_geometry(gt_mesh_cam)

    # Check 2: deepsdf to observation in camera frame

    """
    DEBUG: Temperally add a DeepSDF Shape
    """
    debug_vis_deepsdf = True
    if debug_vis_deepsdf:
        deepsdf_dir = "dataset/deepsdf/zero_shape_chair.ply"
        mesh_deepsdf = o3d.io.read_triangle_mesh(deepsdf_dir)
        mesh_deepsdf.compute_vertex_normals()

        # copy a mesh
        import copy

        mesh_deepsdf_c = copy.deepcopy(mesh_deepsdf)
        mesh_deepsdf_c.transform(t_cam_obj)

        # vis.add_geometry(mesh_deepsdf_c)

        # then tranform with deepsdf trans ...
        t_cam_deepsdf = det.t_cam_deepsdf
        mesh_deepsdf_c_deepsdf = copy.deepcopy(mesh_deepsdf)
        mesh_deepsdf_c_deepsdf.transform(t_cam_deepsdf)
        # vis.add_geometry(mesh_deepsdf_c_deepsdf)

    """
    Get NeRF model and visualize as DeepSDF
    """
    # latent to mesh
    latent = init_latent
    mesh_o = shape_model.get_shape_from_latent(latent)
    # mesh to open3d
    mesh_o_o3d = o3d.geometry.TriangleMesh()
    mesh_o_o3d.vertices = o3d.utility.Vector3dVector(mesh_o.verts)
    mesh_o_o3d.triangles = o3d.utility.Vector3iVector(mesh_o.faces)
    # add color to meshes: vertex_channels, face_channels
    # merge RGB into 3 dimensions : mesh_o.vertex_channels['R'] ...
    vertex_colors = np.stack([mesh_o.vertex_channels[n] for n in ["R", "G", "B"]]).T
    mesh_o_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # mesh_o_o3d.triangle_normals = o3d.utility.Vector3dVector(mesh_o.face_channels['normal'])

    # compute normal
    mesh_o_o3d.compute_vertex_normals()

    # object coordinate to world
    mesh_c_o3d = mesh_o_o3d.transform(t_cam_deepsdf)

    vis.add_geometry(mesh_c_o3d)

    vis.run()

    print("done")


def VisualizeNeRFwithPoseandObsPts(det, init_latent, t_co, shape_model, vis=None):
    pts_c = det.surface_points

    # visualize in 3d for pts and gt mesh
    pts_c_o3d = o3d.geometry.PointCloud()
    pts_c_o3d.points = o3d.utility.Vector3dVector(pts_c)

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    vis.clear_geometries()

    # Check 1: gt mesh and observation in camera frame
    vis.add_geometry(pts_c_o3d)
    """
    Get NeRF model and visualize as DeepSDF
    """
    # latent to mesh
    latent = init_latent
    mesh_o = shape_model.get_shape_from_latent(latent)
    # mesh to open3d
    mesh_o_o3d = o3d.geometry.TriangleMesh()
    mesh_o_o3d.vertices = o3d.utility.Vector3dVector(mesh_o.verts)
    mesh_o_o3d.triangles = o3d.utility.Vector3iVector(mesh_o.faces)
    # add color to meshes: vertex_channels, face_channels
    # merge RGB into 3 dimensions : mesh_o.vertex_channels['R'] ...
    vertex_colors = np.stack([mesh_o.vertex_channels[n] for n in ["R", "G", "B"]]).T
    mesh_o_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # mesh_o_o3d.triangle_normals = o3d.utility.Vector3dVector(mesh_o.face_channels['normal'])

    # compute normal
    mesh_o_o3d.compute_vertex_normals()

    # object coordinate to world
    mesh_c_o3d = mesh_o_o3d.transform(t_co)

    vis.add_geometry(mesh_c_o3d)

    vis.run()

    print("done")


def generate_gif(gif_dir):
    # images under dir to gif
    im_list = glob.glob(gif_dir)
    im_list = sorted(im_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    im_list_data = [imageio.v2.imread(im) for im in im_list]

    frame_save_dir = os.path.dirname(gif_dir)

    gif_path = os.path.join(frame_save_dir, f"im.gif")

    imageio.mimsave(gif_path, im_list_data, duration=5)
    # delete intermediate images
    for im in im_list:
        os.remove(im)


def render_images_for_optimization_process(
    latent_list,
    t_bo_list,
    pts_3d_b,
    shape_model,
    vis,
    t_wc_vis=None,
    vis_jump=10,
    view_file="./view.json",
    save_dir=None,
    t_bo_gt=None,
):
    num_steps = len(latent_list)

    assert len(latent_list) == len(t_bo_list)

    print("Rendering the optimization process with open3d ...")

    # from tqdm import tqdm
    # we get at least 10 frames!
    vis_jump_f = min(max(1, num_steps // 10), vis_jump)
    it_vis_steps = list(range(0, num_steps, vis_jump_f))

    latent = None
    for it in it_vis_steps:
        latent = latent_list[it].to(pts_3d_b.device)
        t_bo_cur = t_bo_list[it].to(pts_3d_b.device)

        visualize_current_status(
            latent,
            t_bo_cur,
            pts_3d_b,
            shape_model,
            vis=vis,
            t_wc_vis=t_wc_vis,
            save_ply=False,
            gt_pts_3d_b=None,
            save_name=f"it_{it}.png",
            view_file=view_file,
            save_dir=save_dir,
            vis_pts=False,
            t_wo_gt=t_bo_gt,
        )

    # A further visualization with vis_pts
    if latent is not None:
        visualize_current_status(
            latent,
            t_bo_cur,
            pts_3d_b,
            shape_model,
            vis=vis,
            t_wc_vis=t_wc_vis,
            save_ply=False,
            gt_pts_3d_b=None,
            save_name=f"wpts.png",
            view_file=view_file,
            save_dir=save_dir,
            vis_pts=True,
            t_wo_gt=t_bo_gt,
        )

    print("Visualization DONE.")
