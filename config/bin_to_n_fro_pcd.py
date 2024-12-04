# This script loads a kitti dataset converts from .bin to .pcd and saves it in the output path

# SCRIPT ARGUMENTS: Dataset path, output path

import os
import open3d as o3d
import numpy as np
import glob 
# path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/data_odometry_velodyne/dataset/sequences/21_onecar/velodyne"
# out_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/data_odometry_velodyne/dataset/sequences/21_onecar_pcd/velodyne"

path = '/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/custom_dataset/sequence_hatchback/'
out_path = '/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/custom_dataset/sequence_hatchback_bin/'
def bin_to_pcd(path, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = glob.glob(path + "/*.bin")
    for file in files:
        data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        o3d.io.write_point_cloud(output_path + "/" + os.path.basename(file)[:-4] + ".pcd", pcd)
        print("Saved " + os.path.basename(file)[:-4] + ".pcd")


def pcd_to_bin(path, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = glob.glob(path + "/*.pcd")
    for file in files:
        pcd = o3d.io.read_point_cloud(file)
        data = np.asarray(pcd.points)
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
        data = data.astype(np.float32)
        data.tofile(output_path + "/" + os.path.basename(file)[:-4] + ".bin")
        print("Saved " + os.path.basename(file)[:-4] + ".bin")

# bin_to_pcd(path, out_path)

if __name__ == "__main__":
    # bin_to_pcd(path, out_path)
    pcd_to_bin(path, out_path)
    print("Done")

