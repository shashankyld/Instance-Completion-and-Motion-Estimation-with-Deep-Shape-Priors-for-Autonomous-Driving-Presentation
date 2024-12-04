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
from reconstruct.utils import color_table, convert_to_world_frame, convert_to_canonic_space
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
import yaml


    
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

    # start reconstruction
    objects_recon = []
    start = get_time()
    
    id = 209
    detections = np.load(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy', allow_pickle='TRUE').item()
    
    # start reconstruction
    objects_recon = {}
    
    # Ugly initialization
    point_w_frame = np.array([[0, 0, 0]])
    point_c_frame = np.array([[0, 0, 0]])
    
    _, rott = convert_to_world_frame(point_w_frame)
    
    start = get_time()
        
    for frame_id, det in detections.items():
        point_w_frame = np.concatenate((point_w_frame, det.PCD))
        
        c_point = convert_to_canonic_space(det.pts_obj_global)
        point_c_frame = np.concatenate((point_c_frame, c_point))

    # Ugly initialization
    point_w_frame = point_w_frame[1:]
    point_c_frame = point_c_frame[1:]

    point_s_w_frame, rot_g_world =  convert_to_world_frame(point_c_frame)
        
    # Optimization
    obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), point_c_frame)
    objects_recon[frame_id] = obj

    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()
    
    # Add SOURCE LiDAR point cloud
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(point_s_w_frame)
    green_color = np.full((point_s_w_frame.shape[0], 3), color_table[1]) # Green COLOR
    scene_pcd.colors = o3d.utility.Vector3dVector(green_color)
    vis.add_geometry(scene_pcd)
    
    # Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)

    for frame_id, obj in objects_recon.items():
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()

        # Add OUTPUT LiDAR point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        red_color = np.full((mesh.vertices.shape[0], 3), color_table[0]) # Red COLOR
        obj_pcd.colors = o3d.utility.Vector3dVector(red_color)
        vis.add_geometry(obj_pcd)
        
        
        # Transform mesh from object to world coordinate
        # mesh_o3d.transform(obj.t_cam_obj)
        # obj_pcd.transform(obj.t_cam_obj)
        
        mesh_o3d.transform(rot_g_world)
        obj_pcd.transform(rot_g_world)
        
        vis.add_geometry(mesh_o3d)
        print("AFTER obj.t_cam_obj", obj.t_cam_obj)


    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()

# python3 DEEP_SDF/canonical_space.py
'''
SHOW POINTS IN CANONICAL SPACE.
'''

if __name__ == "__main__":
    main()