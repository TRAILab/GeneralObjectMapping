"""
Given a group of ply names, first use open3d to project into an image (load a view json file),
then arrange those images into a large image.
"""

import argparse
import os

import numpy as np
import open3d as o3d
from PIL import Image


def process_plys(ply_dir, output_image):
    view_file = "view_canonical_mesh.json"

    # Check if the view file exists
    if not os.path.exists(view_file):
        # If not, let the user manually adjust the view
        print(
            "Please manually adjust the view in the Open3D window and press 's' to save the view to 'view.json'."
        )

        # Load the first point cloud
        ply_files = [f for f in os.listdir(ply_dir) if f.endswith(".ply")]
        pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_files[0]))

        # Create a visualizer with key
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=800, height=800)
        vis.add_geometry(pcd)

        # Register a key callback function to save the view when 's' is pressed
        def save_view(vis):
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(view_file, param)
            print("save to", view_file)

        vis.register_key_callback(ord("S"), save_view)

        vis.run()  # Allow the user to manually adjust the view
        vis.destroy_window()

        # The user should have saved the view to 'view.json' by pressing 's'

    # Load the view point
    view = o3d.io.read_pinhole_camera_parameters(view_file)

    # Get a list of all ply files in the directory
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith(".ply")]

    # Create a list to store the images
    images = []

    # Process each ply file
    for ply_file in ply_files:
        # Load the point cloud
        # pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_file))

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(ply_dir, ply_file))

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)

        # Add the point cloud
        vis.add_geometry(mesh)

        # Set the view point
        vis.get_view_control().convert_from_pinhole_camera_parameters(view)

        # Capture the image
        img = vis.capture_screen_float_buffer(True)
        # to numpy
        img = np.asarray(img)

        # Close the visualizer
        vis.destroy_window()

        # Convert the image to a PIL image and append it to the list
        images.append((img * 255).astype(np.uint8))

    # numpy concat all images
    large_image = np.concatenate(images, axis=1)

    # Concatenate the images into a large image
    large_image = Image.fromarray(large_image)

    # Save the large image
    large_image.save(output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ply files into an image.")
    parser.add_argument(
        "--ply_dir",
        help="The directory containing the ply files.",
        default="output/test_shape_latent_space/interpolation/chair_table/",
    )
    parser.add_argument(
        "--output_image", help="The output image file.", default="output/chair_table.png"
    )
    args = parser.parse_args()

    process_plys(args.ply_dir, args.output_image)
