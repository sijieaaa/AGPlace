

import open3d
import numpy as np
import open3d as o3d


c = 0.8
# RGB
cmaps = [
    [0, 0, c],
    [0, c, 0],
    [0, c, c],
    [c, 0, 0],
    [c, 0, c],
    [c, c, 0],
    [c, c, c],
]


def viz_lidar_open3d(pointcloud, width=None, height=None):

    pointcloud = np.array(pointcloud)

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])


    # pcd_list = []
    pcd_list = [FOR1]

    if isinstance(pointcloud, np.ndarray):
        pcd = open3d.geometry.PointCloud()
        points = pointcloud
        points = open3d.utility.Vector3dVector(points)
        pcd.points = points
        colors = np.ones([pointcloud.shape[0], 3]) * cmaps[0]
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd)

    elif isinstance(pointcloud, list):
        for i in range(len(pointcloud)): 
            pcd = open3d.geometry.PointCloud()
            points = pointcloud[i]
            points = open3d.utility.Vector3dVector(points)
            pcd.points = points
            colors = np.ones([pointcloud[i].shape[0], 3]) * cmaps[i]
            pcd.colors = open3d.utility.Vector3dVector(colors)
            pcd_list.append(pcd)

    else:
        raise NotImplementedError





    if (width is not None) & (height is not None):
        open3d.visualization.draw_geometries(pcd_list, width=width, height=height)
    else:
        open3d.visualization.draw_geometries(pcd_list)