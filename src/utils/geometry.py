"""
Geometry utils function.

Basic utils function for geometry.

Transform 3D points.

"""

import numpy as np
import open3d as o3d
import torch


def calculate_IoU(bbox1, bbox2):
    """
    IoU of two bounding boxes on image plane.

    @ bbox: (x1, y1, x2, y2)
    """

    # calculate the intersection area
    inter = np.maximum(
        0, np.minimum(bbox1[2], bbox2[2]) - np.maximum(bbox1[0], bbox2[0])
    ) * np.maximum(0, np.minimum(bbox1[3], bbox2[3]) - np.maximum(bbox1[1], bbox2[1]))

    # calculate the union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - inter

    iou = inter / union

    return iou


def calculate_ob_ratio_from_projection(gt_bbox_world, gt_t_world_cam, mask, K):
    """
    Project the GT BBOX area into 2D image,
    calculate the IoU between the bbox of mask, and the projected bbox.

    Args:
        - mask: (H,W) bool, indicate the interested object only
    """

    # project the bbox to 2D
    # bbox: (8,3);  gt_bbox_cam = gt_t_world_cam.inv @ gt_bbox_world
    gt_t_cam_world = np.linalg.inv(gt_t_world_cam)

    # transform to camera frame
    gt_bbox_cam = gt_bbox_world @ gt_t_cam_world[:3, :3].T + gt_t_cam_world[:3, 3]

    # project to image plane, with K
    gt_bbox_2d = gt_bbox_cam @ K.T
    # normalize
    gt_bbox_2d = gt_bbox_2d / gt_bbox_2d[:, 2:3]

    # 8,2 projected points into a smallest bbox
    bbox_proj = np.array(
        [
            gt_bbox_2d[:, 0].min(),
            gt_bbox_2d[:, 1].min(),
            gt_bbox_2d[:, 0].max(),
            gt_bbox_2d[:, 1].max(),
        ]
    )

    # get the bbox of mask
    bbox_ob = np.argwhere(mask)

    # if mask is empty
    if bbox_ob.shape[0] == 0:
        return 0.0

    bbox_ob = np.array(
        [bbox_ob[:, 1].min(), bbox_ob[:, 0].min(), bbox_ob[:, 1].max(), bbox_ob[:, 0].max()]
    )

    # calculate IoU
    iou = calculate_IoU(bbox_proj, bbox_ob)

    # debug = False
    # if debug:
    #     print('iou:', iou)

    #     # plot the process into mask image
    #     import cv2
    #     mask_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #     mask_img[mask] = [255,255,255]

    #     # change type to int
    #     bbox_ob = bbox_ob.astype(np.int32)
    #     bbox_proj = bbox_proj.astype(np.int32)

    #     gt_bbox_2d = gt_bbox_2d.astype(np.int32)

    #     # plot the 8 projected points, and connect to form a 3D bbox in 2D
    #     for i in range(8):
    #         mask_img = cv2.circle(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), 1, (0,0,255), 2)
    #     for i in range(4):
    #         mask_img = cv2.line(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), (gt_bbox_2d[i+4,0], gt_bbox_2d[i+4,1]), (0,0,255), 2)
    #     for i in range(4):
    #         mask_img = cv2.line(mask_img, (gt_bbox_2d[i,0], gt_bbox_2d[i,1]), (gt_bbox_2d[(i+1)%4,0], gt_bbox_2d[(i+1)%4,1]), (0,0,255), 2)
    #         mask_img = cv2.line(mask_img, (gt_bbox_2d[i+4,0], gt_bbox_2d[i+4,1]), (gt_bbox_2d[(i+1)%4+4,0], gt_bbox_2d[(i+1)%4+4,1]), (0,0,255), 2)

    #     # draw bbox
    #     mask_img = cv2.rectangle(mask_img, (bbox_ob[0], bbox_ob[1]), (bbox_ob[2], bbox_ob[3]), (0,255,0), 2)
    #     mask_img = cv2.rectangle(mask_img, (bbox_proj[0], bbox_proj[1]), (bbox_proj[2], bbox_proj[3]), (0,0,255), 2)

    #     # Put text to show iou
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(mask_img, 'iou: {:.3f}'.format(iou), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #     # get a static variable, and +1 each time
    #     if not hasattr(calculate_ob_ratio_from_projection, "counter"):
    #         calculate_ob_ratio_from_projection.counter = 0
    #     else:
    #         calculate_ob_ratio_from_projection.counter += 1

    #     cv2.imwrite(f'./output/debug/mask_img_{calculate_ob_ratio_from_projection.counter}_iou{iou}.png', mask_img)

    return iou


def transform_3d_points(T: torch.Tensor, pts: torch.Tensor):
    """

    Transform points with pts_new = T @ pts

    Input:
        pts: (N, 3)
        T: (4, 4)

    Output:
        pts: (N, 3)

    """
    assert pts.shape[1] == 3
    assert T.shape == (4, 4)

    pts = torch.cat([pts, torch.ones(pts.shape[0], 1, device=pts.device)], dim=1)
    pts = torch.matmul(T, pts.t()).t()
    pts = pts[:, :3] / pts[:, 3:]
    return pts


def skew(w):
    wc = torch.stack(
        (
            torch.tensor(0, dtype=torch.float32).cuda(),
            -w[2],
            w[1],
            w[2],
            torch.tensor(0, dtype=torch.float32).cuda(),
            -w[0],
            -w[1],
            w[0],
            torch.tensor(0, dtype=torch.float32).cuda(),
        )
    ).view(3, 3)
    return wc


def Oplus_se3(T, v, order="left"):
    """
    Update: Only left operation is
        the same to recover the original pose!

    SE3/se3 operations.
    Origin source: reconstruct/loss.py

    Support change of left/right operation.
    * Right: transform w.r.t. local coordinate
    * Left: transform w.r.t. global coordinate
    """
    rho = v[:3]  # translation
    phi = v[3:]  # rotation
    tolerance = 1e-12

    # C = vec2rot(phi)
    ####################################################################
    angle = torch.norm(phi, p=2, dim=0)
    if angle < tolerance:
        # vec2rotSeries
        N = 10
        C = torch.eye(3, dtype=torch.float32).cuda()
        xM = torch.eye(3, dtype=torch.float32).cuda()
        cmPhi = skew(phi)
        for n in range(1, N + 1):
            xM = torch.mm(xM, (cmPhi / n))
            C = C + xM

        from sqrtm import sqrtm

        tmp = sqrtm(torch.mm(torch.transpose(C, 0, 1), C))
        C = torch.mm(C, torch.inverse(tmp))
    else:
        axis_ = phi / angle
        axis = torch.reshape(axis_, (3, 1))
        cp = torch.cos(angle)
        sp = torch.sin(angle)
        I = torch.eye(3, dtype=torch.float32).cuda()
        C = cp * I + (1 - cp) * torch.mm(axis, torch.transpose(axis, 0, 1)) + sp * skew(axis_)
    ####################################################################

    # J = vec2jac(phi)
    ####################################################################
    ph = torch.norm(phi, p=2, dim=0)
    if ph < tolerance:
        # vec2jacSeries
        N = 10
        J = torch.eye(3, dtype=torch.float32).cuda()
        pxn = torch.eye(3, dtype=torch.float32).cuda()
        px = skew(phi)
        for n in range(1, N + 1):
            pxn = torch.mm(pxn, px) / (n + 1)
            J = J + pxn
    else:
        axis_ = phi / ph
        axis = torch.reshape(axis_, (3, 1))
        cph = (1 - torch.cos(ph)) / ph
        sph = torch.sin(ph) / ph
        I = torch.eye(3, dtype=torch.float32).cuda()
        J = sph * I + (1 - sph) * torch.mm(axis, torch.transpose(axis, 0, 1)) + cph * skew(axis_)

    rho_ = torch.reshape(rho, (3, 1))
    trans = torch.mm(J, rho_)
    dT = torch.stack(
        (
            C[0, 0],
            C[0, 1],
            C[0, 2],
            trans[0, 0],
            C[1, 0],
            C[1, 1],
            C[1, 2],
            trans[1, 0],
            C[2, 0],
            C[2, 1],
            C[2, 2],
            trans[2, 0],
            torch.tensor(0, dtype=torch.float32).cuda(),
            torch.tensor(0, dtype=torch.float32).cuda(),
            torch.tensor(0, dtype=torch.float32).cuda(),
            torch.tensor(1, dtype=torch.float32).cuda(),
        )
    ).view(4, 4)

    if order == "right":
        return torch.mm(T, dT)
    elif order == "left":
        return torch.mm(dT, T)
    else:
        raise NotImplementedError


def aggregate_multi_view_points(valid_observations, voxel_size=None):
    """
    Aggregate all points from all views;
    And downsample the points

    @ voxel_size: e.g., 0.01; For CO3D dataset, the coordinate is random. We use scale to normalize it when set to None.
    """
    pts_3d_w_aggregate = []
    for ob in valid_observations:
        pts_3d_w_aggregate.append(ob["pts_3d_w"])

    pts_3d_w_aggregate = torch.cat(pts_3d_w_aggregate, axis=0)

    # use open3d to downsample the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d_w_aggregate)

    if voxel_size is None:
        scale = pcd.get_max_bound() - pcd.get_min_bound()
        voxel_size = np.linalg.norm(scale) / 100.0

    debug = False
    if debug:
        os.makedirs("./output/points", exist_ok=True)
        o3d.io.write_point_cloud("./output/points/pts_3d_w_aggregate.ply", pcd)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # save to ply
    if debug:
        o3d.io.write_point_cloud("./output/points/pts_3d_w_aggregate_downsample.ply", pcd)

    # extract np points
    pts_3d_w_aggregate_downsample = np.asarray(pcd.points)

    return pts_3d_w_aggregate_downsample


def fit_cuboid_to_points(points):
    """
    @ points: (N,3) torch.Tensor

    @ return: (8,3) torch.Tensor
    """
    max_bound = torch.max(points, axis=0).values  # x,y,z
    min_bound = torch.min(points, axis=0).values  # x,y,z

    # max_bound = max_bound.cpu().numpy()
    # min_bound = min_bound.cpu().numpy()

    # get the 8 vertices of the bounding box
    vertices = torch.Tensor(
        [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ]
    ).to(points.device)

    return vertices
