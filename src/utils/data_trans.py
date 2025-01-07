"""
Transform types
"""

import numpy as np
import open3d as o3d


def trimesh_to_open3d(mesh):
    vertices = mesh.verts
    faces = mesh.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # add color to meshes: vertex_channels, face_channels
    # merge RGB into 3 dimensions : mesh_o.vertex_channels['R'] ...
    vertex_colors = np.stack([mesh.vertex_channels[n] for n in ["R", "G", "B"]]).T
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # compute normal
    # o3d_mesh.compute_vertex_normals()

    return o3d_mesh
