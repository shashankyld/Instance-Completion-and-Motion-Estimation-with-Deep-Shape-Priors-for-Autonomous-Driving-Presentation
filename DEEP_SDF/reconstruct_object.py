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
import open3d as o3d
import numpy as np
import os
from os.path import join, dirname, abspath
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
from reconstruct.utils import color_table, write_mesh_to_ply, convert_to_world_frame, convert_to_canonic_space
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
    # instance_id_list = [0, 1, 72, 209, 373, 512, 551, 555]
    id = 551
    detections = np.load(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy', allow_pickle='TRUE').item()

    # start reconstruction
    objects_recon = {}
    
    # Ugly initialization
    g_point = np.array([[0, 0, 0]])
    
    # Mesh
    save_dir = cfg['save_mesh_dir']
    
    _, rott = convert_to_world_frame(g_point)
    
    start = get_time()

    g_pose = {}
    s_pose = {}
    
    for frame_id, det in detections.items():
        if det.pts_obj_global.shape[0] > 200:
            c_all_points = convert_to_canonic_space(det.pts_obj_global)
            obj = optimizer.reconstruct_object(np.eye(4) , c_all_points)
            # g pose is global pose in world frame
            # s pose is global pose in senosr frame
            obj.g_pose = det.g_pose @ rott @ obj.t_cam_obj
            obj.s_pose = det.s_pose @ rott @ obj.t_cam_obj
            
            g_pose[frame_id] = obj.g_pose
            s_pose[frame_id] = obj.s_pose

            objects_recon[frame_id] = obj
            g_point = np.concatenate((g_point, det.PCD))
    
    np.save(f'results/deep_sdf/pose/g_pose_{id}.npy', np.array(g_pose, dtype=object), allow_pickle=True)
    np.save(f'results/deep_sdf/pose/s_pose_{id}.npy', np.array(s_pose, dtype=object), allow_pickle=True)
    
    g_point = g_point[1:]
    
    
    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # # Add Source LiDAR point cloud
    # c_source_pcd = o3d.geometry.PointCloud()
    # c_source_pcd.points = o3d.utility.Vector3dVector(pts_obj_global)
    # green_color = np.full((pts_obj_global.shape[0], 3), color_table[1]) # Green COLOR
    # c_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
    # vis.add_geometry(c_source_pcd)
    
    # Add Source LiDAR point cloud - global
    g_source_pcd = o3d.geometry.PointCloud()
    g_source_pcd.points = o3d.utility.Vector3dVector(g_point)
    green_color = np.full((g_point.shape[0], 3), color_table[2]) # BLUE COLOR
    g_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
    vis.add_geometry(g_source_pcd)
    
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for frame_id, obj in objects_recon.items():
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])

        # Transform mesh from object to world coordinate
        mesh_o3d.transform(obj.g_pose)
        vis.add_geometry(mesh_o3d)
        
        write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}", "%d.ply" % frame_id))
    
    print("FINISHED")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()