import os
import open3d as o3d
import numpy as np
import glob

cloud_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/data_odometry_velodyne/dataset/sequences/21_onecar/velodyne"
bb_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/OpenPCDet/tools/bb_pose.txt"
car_odom_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/kiss-icp/python/results/21_onecar_odom/velodyne_poses_tum.txt"

def return_bb_poses(bb_path):
    bb_poses = []
    with open(bb_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            # Convert to float
            line = [float(i) for i in line]
            bb_poses.append(line)
    return bb_poses

print("bb_poses_type", type(return_bb_poses(bb_path)))
# print("bb_points", return_bb_poses(bb_path))
# BB format - [x, y, z, w, h, l, yaw]

def return_car_odom(car_odom_path):
    car_odom = []
    with open(car_odom_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            # Convert to float
            line = [float(i) for i in line]
            car_odom.append(line)
    return car_odom

print("car_odom_type", type(return_car_odom(car_odom_path)[0][0]))
# print("car_odom", return_car_odom(car_odom_path))
# Odom Pose format - [R|t] except last row is [0, 0, 0, 1]

def return_clouds(cloud_path):
    clouds = []
    for file in glob.glob(cloud_path + "/*.bin"):
        cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        clouds.append(cloud)
    return clouds

print("clouds_type", type(return_clouds(cloud_path)[0][0]))
# print("clouds", return_clouds(cloud_path))

def return_clouds_o3d(cloud_path):
    clouds = []
    for file in glob.glob(cloud_path + "/*.bin"):
        cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:, :3]))
        clouds.append(cloud)
    return clouds

print("clouds_o3d_type", type(return_clouds_o3d(cloud_path)[0]))
# print("clouds_o3d", return_clouds_o3d(cloud_path))

def return_clouds_o3d_bb(cloud_path, bb_path):
    clouds = []
    bb_poses = return_bb_poses(bb_path)
    for file in glob.glob(cloud_path + "/*.bin"):
        cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:, :3]))
        clouds.append(cloud)
    return clouds, bb_poses

print("clouds_o3d_bb_type", type(return_clouds_o3d_bb(cloud_path, bb_path)[0][0]))
# print("clouds_o3d_bb", return_clouds_o3d_bb(cloud_path, bb_path))

def return_clouds_o3d_bb_odom(cloud_path, bb_path, car_odom_path):
    clouds = []
    bb_poses = return_bb_poses(bb_path)
    car_odom = return_car_odom(car_odom_path)
    for file in glob.glob(cloud_path + "/*.bin"):
        cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:, :3]))
        clouds.append(cloud)
    return clouds, bb_poses, car_odom

print("clouds_o3d_bb_odom_type", type(return_clouds_o3d_bb_odom(cloud_path, bb_path, car_odom_path)[0][0]))
# print("clouds_o3d_bb_odom", return_clouds_o3d_bb_odom(cloud_path, bb_path, car_odom_path))


def visualize(cloud_path, bb_path):
    clouds = return_clouds_o3d(cloud_path)
    bb_poses = return_bb_poses(bb_path)
    o3d.visualization.draw_geometries([clouds[0]])
    # Add a bounding box to the scene # Create a random oriented bounding box at the origin (0, 0, 0).
    bb = o3d.geometry.OrientedBoundingBox.create_from_points(clouds[0].points)
    bb.color = (1, 0, 0)
    o3d.visualization.draw_geometries([clouds[0], bb])


visualize(cloud_path, bb_path)