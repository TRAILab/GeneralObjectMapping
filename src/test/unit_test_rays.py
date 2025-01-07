import numpy as np
import open3d as o3d

# Assuming the function is a method of a class named 'ShapeIO'
from shape_model.shape_io import Shap_E

grid_size = 32
shape_model = Shap_E(grid_size=grid_size)

shape_io = shape_model

# Define the mask, transformation matrix, and camera matrix
mask = np.ones((100, 100))
t_cam_obj = np.eye(4).astype(np.float32)
K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]).astype(np.float32)

# Generate the rays
rays = shape_io.generate_rays(mask, t_cam_obj, K)

# Create a 3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the camera center to the visualization
camera_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
camera_center.paint_uniform_color([1, 0, 0])
vis.add_geometry(camera_center)

# Add the image plane to the visualization
image_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
image_plane.translate([-0.5, -0.5, 1])
image_plane.paint_uniform_color([0, 1, 0])
vis.add_geometry(image_plane)

# Add the sampled pixels to the visualization
for i in range(rays.shape[0]):
    pixel = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    pixel.translate([rays[i, 0, 0], rays[i, 0, 1], 1])
    pixel.paint_uniform_color([0, 0, 1])
    vis.add_geometry(pixel)

# Add the rays to the visualization
for i in range(rays.shape[0]):
    points = [np.zeros(3), rays[i, 1, :].numpy()]
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

# Run the visualization
vis.run()
vis.destroy_window()
