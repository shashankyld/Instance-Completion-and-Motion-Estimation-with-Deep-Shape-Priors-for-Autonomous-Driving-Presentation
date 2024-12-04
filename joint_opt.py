#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import click
from deep_sdf.deep_sdf.workspace import config_decoder
import numpy as np
import open3d as o3d
import os
from os.path import join, dirname, abspath
from reconstruct.utils import color_table, convert_to_world_frame, convert_to_canonic_space, T_general
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
import yaml


'''    
@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/config.yaml'))
'''

# 2D and 3D detection and data association
def joint_optimization(config, point_cloud):
    cfg = yaml.safe_load(open(config))
    DeepSDF_DIR = cfg['deepsdf_dir']
    decoder = config_decoder(DeepSDF_DIR)
    optimizer = Optimizer(decoder, cfg)

    start = get_time()

    # Load point cloud from cropped_car.pcd
    # car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/DEEP_SDF/cropped_car_canonical.pcd")
    car = point_cloud
    print("car", car)
    # Object to Canonical Frame Transformation
    T_constant_parity = T_general( -np.pi/2,0, np.pi/2, 1/2, [0, 0, 0]) ## CAR FRAME EAST, UP, BACK - XYZ FRAME
    car.transform(T_constant_parity)
    # Rotate point cloud by 20 degrees around z-axis
    car.transform(T_general(0, 0,0, 1, [0, 0, 0]))  ## CAR FRAME EAST, UP, BACK - XYZ FRAME - Pitch|Yaw|-ve Roll
    # car.transform(rot_g_world)
    # o3d.visualization.draw_geometries([car])
    car_points = np.asarray(car.points)
    # car_points = car_points.astype(np.float32)
    print("car_points.shape", car_points.shape)
    # Optimization
    obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), car_points)
    end = get_time()

    print("code and pose optimized for this point cloud, time elapsed: %f seconds", end - start)
    
    # CAR POINTS AFTER TRANSFORMATION
    correction_transform = obj.t_cam_obj
    print("correction_transform", correction_transform)
    car_points_after_correction = np.matmul(car_points, correction_transform[:3, :3].T) + correction_transform[:3, 3]
    print("car_points_after_correction.shape", car_points_after_correction.shape)



    
    # Add SOURCE LiDAR point cloud
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(car_points)
    green_color = np.full((car_points.shape[0], 3), color_table[1]) # Green COLOR
    scene_pcd.colors = o3d.utility.Vector3dVector(green_color)

    # Add optimized transform cloud
    car_points_after_correction_pcd = o3d.geometry.PointCloud()
    car_points_after_correction_pcd.points = o3d.utility.Vector3dVector(car_points_after_correction)
    red_color = np.full((car_points_after_correction.shape[0], 3), color_table[0]) # Red COLOR
    car_points_after_correction_pcd.colors = o3d.utility.Vector3dVector(red_color)



    # Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)

  
    mesh = mesh_extractor.extract_mesh_from_code(obj.code)
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    mesh_o3d.compute_vertex_normals()

    # Add OUTPUT LiDAR point cloud
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    # obj_pcd.transform(T_constant_parity)
    # Blue color
    blue_color = np.full((mesh.vertices.shape[0], 3), color_table[2]) # Blue COLOR
    obj_pcd.colors = o3d.utility.Vector3dVector(blue_color)

    return scene_pcd, obj_pcd, car_points_after_correction_pcd, mesh_o3d, correction_transform

if __name__ == "__main__":
    joint_optimization("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/DEEP_SDF/configs/config.yaml")