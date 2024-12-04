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
import time
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
    car = o3d.io.read_point_cloud("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/config/hatch_back_in_object_frame.pcd")

    ## APPLYING TRANSFORMATION INITIALLY AS NOICE
    car.transform(T_general(np.deg2rad(0), np.deg2rad(0),np.deg2rad(15), 1, [0, 0, 0]))  ## CAR FRAME EAST, UP, BACK - XYZ FRAME - Roll|Pitch|Yaw|SCALE|TRANSLATION

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
    obj = optimizer.reconstruct_object_list(np.eye(4, dtype="float32"), car_canonical_points)
    end = get_time()
    obj_codes = obj.code_list
    obj_poses = obj.pose_list
    print("obj_codes", obj_codes)
    print("code and pose optimized for this point cloud, time elapsed: %f seconds", end - start)
    
    
    # CORRECTION FACTOR TRANSFORMATION
    correction_transform = obj.t_cam_obj
    print(obj.t_cam_obj)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()
    vis.add_geometry(car)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)    
    mesh_o3d = o3d.geometry.TriangleMesh()
    vis.add_geometry(mesh_o3d)
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)

    for i in range(len(obj_codes)):
        mesh = mesh_extractor.extract_mesh_from_code(obj_codes[i])
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()
        

        correction_transform = obj.pose_list[i]
        # mesh_optimized = copy.deepcopy(mesh_o3d)
        mesh_o3d.transform(np.linalg.inv(correction_transform))

        # mesh_in_object_frame = copy.deepcopy(mesh_optimized)
        # APPLY INVERSE TRANSFORMATION TO GET MESH IN OBJECT FRAME WITH CONSTANT PARITY
        mesh_o3d.transform(np.linalg.inv(T_constant_parity))

        # Write to mesh
        # o3d.io.write_triangle_mesh("/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/optimized_meshes/car1/mesh_optimized" +"_" +str(i) + ".ply", mesh_optimized )


        # Visualize mesh 
        vis.update_geometry(mesh_o3d)
        # vis.add_geometry(car)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.5)
        vis.run()
    vis.destroy_window()


    


# python3 DEEP_SDF/canonical_space.py
'''
SHOW POINTS IN CANONICAL SPACE.
'''


if __name__ == "__main__":
    main()