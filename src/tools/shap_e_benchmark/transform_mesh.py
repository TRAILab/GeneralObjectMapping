import open3d as o3d

file_name = ...
# text
mesh_dir = ...
# # Load pose
result = ...


import torch

data = torch.load(result)
pose_bo = data["output"]["pose_bo"]

# Open 3d load mesh and transform with pose_bo, then save
mesh = o3d.io.read_triangle_mesh(mesh_dir)
mesh.transform(pose_bo)

save_dir = "./output/shap_e_benchmark"
import os

os.makedirs(save_dir, exist_ok=True)

file_name = os.path.join(save_dir, f"mesh_shap_e_transformed_{file_name}.ply")
# o3d.io.write_point_cloud(file_name, pcd_mesh)

o3d.io.write_triangle_mesh(file_name, mesh)

print("save to", file_name)
