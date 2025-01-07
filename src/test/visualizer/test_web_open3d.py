"""
This file is for web visualization of Open3D data.
"""

import os


def test_web_open3d():
    import open3d as o3d

    o3d.visualization.webrtc_server.enable_webrtc()

    print("running!")


if __name__ == "__main__":
    test_web_open3d()
