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
import copy

    
@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/config.yaml'))

# 2D and 3D detection and data association
def main(config):
    cfg = yaml.safe_load(open(config))
    DeepSDF_DIR = cfg['deepsdf_dir']
    decoder = config_decoder(DeepSDF_DIR)
    optimizer = Optimizer(decoder, cfg)


    start = get_time()

    # Load point cloud from cropped_car.pcd
    # car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/DEEP_SDF/cropped_car_canonical.pcd")
    # car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/DataSets/custom_dataset/canonical_car.pcd")
    # car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/config/car_in_object_frame.pcd")
    car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd_new/35.pcd")

    
    ## APPLYING TRANSFORMATION INITIALLY AS NOICE
    # car.transform(T_general(np.deg2rad(0), np.deg2rad(0),np.deg2rad(20), 1, [0, 0, 0]))  ## CAR FRAME EAST, UP, BACK - XYZ FRAME - Roll|Pitch|Yaw|SCALE|TRANSLATION

    car_canonical = copy.deepcopy(car)
    print("car", car)
    # Object to Canonical Frame Transformation
    T_constant_parity = T_general( -np.pi/2,0, np.pi/2, 1/2, [0, 0, 0]) ## CAR FRAME EAST, UP, BACK - XYZ FRAME
    car_canonical.transform(T_constant_parity)
    # Rotate point cloud by 20 degrees around z-axis
    
    # car.transform(rot_g_world)
    # o3d.visualization.draw_geometries([car])
    car_canonical_points = np.asarray(car_canonical.points)
    # car_points = car_points.astype(np.float32)
    print("car_points.shape", car_canonical_points.shape)
    # Optimization
    obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), car_canonical_points)
    end = get_time()
    
    print("code and pose optimized for this point cloud, time elapsed: %f seconds", end - start)
    
    # CORRECTION FACTOR TRANSFORMATION
    correction_transform = obj.t_cam_obj
    print("correction in transform", obj.t_cam_obj)

    # Calculate scale change in optimization 
    canonical_scale = np.linalg.det(correction_transform[:3, :3])**(1/3)



    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()
    
    # Visualize

    # 1. Add Coordinated Frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    # 2. Original car in object frame
    vis.add_geometry(car)
    # 3. Car in canonical frame
    # vis.add_geometry(car_canonical)

    
    
    # Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    mesh = mesh_extractor.extract_mesh_from_code(obj.code)
    mesh_original = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    mesh_original.compute_vertex_normals()
    # Change size of mesh by scale factor from optimization
    # mesh_o3d.scale(1/canonical_scale, center=mesh_o3d.get_center())
    mesh_optimized = copy.deepcopy(mesh_original)
    mesh_optimized.transform(np.linalg.inv(correction_transform))
    
    # 4. Car Mesh in Canonical Frame (before optimization) 
    # vis.add_geometry(mesh_original)

    # temperary: mesh_unoptimized_in_object_frame
    mesh_unoptimized_in_object_frame = copy.deepcopy(mesh_original)
    mesh_unoptimized_in_object_frame.transform(np.linalg.inv(T_constant_parity))
    vis.add_geometry(mesh_unoptimized_in_object_frame)

    # 5. Car Mesh in Object Frame (after optimization)
    # vis.add_geometry(mesh_optimized)

    mesh_in_object_frame = copy.deepcopy(mesh_optimized)
    # APPLY INVERSE TRANSFORMATION TO GET MESH IN OBJECT FRAME WITH CONSTANT PARITY
    mesh_in_object_frame.transform(np.linalg.inv(T_constant_parity))
    # 6. Car Mesh in Object Frame (after optimization)
    vis.add_geometry(mesh_in_object_frame)


    # Add OUTPUT LiDAR point cloud
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    # obj_pcd.transform(T_constant_parity)
    # Blue color
    blue_color = np.full((mesh.vertices.shape[0], 3), color_table[2]) # Blue COLOR
    obj_pcd.colors = o3d.utility.Vector3dVector(blue_color)
    # vis.add_geometry(obj_pcd)
        
        
    # Transform mesh from object to world coordinate
    # mesh_o3d.transform(obj.t_cam_obj)
    # obj_pcd.transform(obj.t_cam_obj)
        
    # mesh_o3d.transform(rot_g_world)
    # obj_pcd.transform(rot_g_world)
        
    # vis.add_geometry(mesh_o3d)
    # print("AFTER obj.t_cam_obj", obj.t_cam_obj)
    
    # ### LOADING EXTERNAL MESHES 
    # ext_mesh =    o3d.io.read_triangle_mesh("/home/shashank/.gazebo/models/hatchback/meshes/hatchback.obj")
    # ext_mesh.transform(T_general(0, 0,0, 1/100, [3, 0, 0]))
    # vis.add_geometry(ext_mesh)
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()
    


# python3 DEEP_SDF/canonical_space.py
'''
SHOW POINTS IN CANONICAL SPACE.
'''


if __name__ == "__main__":
    # pcd_folder_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd_new/"
    # pcd_files = os.listdir(pcd_folder_path)
    # pcd_files.sort()
    # for pcd_file in pcd_files:
    #     print("pcd_file", pcd_file)
    #     main(pcd_file = pcd_folder_path + pcd_file)
    main()