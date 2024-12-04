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
from reconstruct.optimizer_accumulated import Optimizer, MeshExtractor
from reconstruct.utils import color_table, write_mesh_to_ply, convert_to_world_frame, convert_to_canonic_space, text_3d
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
    id = 209
    detections = np.load(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}.npy', allow_pickle='TRUE').item()

    # start reconstruction
    objects_recon = {}
    
    # Ugly initialization
    acc_points = {}
    g_point_0 = np.array([[0, 0, 0]])
    
    # Mesh
    save_dir = cfg['save_mesh_dir']
    
    _, rott = convert_to_world_frame(g_point_0)
    
    start = get_time()

    g_pose = {}
    s_pose = {}

    is_first_f = False
    tmp_pts_canonical_space = []
    
    ######################################## Velocity ########################################
    velocity = {}
    prev_x = -np.inf
    prev_y = -np.inf 
    prev_z = -np.inf
    prev_frame_id = -np.inf
    ######################################## Velocity ########################################
    
    load = False
    if load == False:
        for frame_id, det in detections.items():

            if det.pts_obj_global.shape[0] > 200:

                if is_first_f:
                    pts_canonical_space = convert_to_canonic_space(det.pts_obj_global)
                    
                    # Transform with optimization transformation
                    tmp = tmp_pts_canonical_space[0]
                    for a in tmp_pts_canonical_space:
                        tmp = np.concatenate((tmp, a))

                    accumulated_pts_canonical_space = np.concatenate((tmp, pts_canonical_space))
                    acc_points[frame_id] = accumulated_pts_canonical_space
                else: 
                    pts_canonical_space = convert_to_canonic_space(det.pts_obj_global)
                    accumulated_pts_canonical_space = pts_canonical_space
                    acc_points[frame_id] = accumulated_pts_canonical_space
                    is_first_f = True
                    
                obj = optimizer.reconstruct_object(np.eye(4) , accumulated_pts_canonical_space)
                
                if len(tmp_pts_canonical_space) > 3:
                    tmp_pts_canonical_space.pop(0)
                    
                # Optimized Pose
                pts_canonical_space_homo = np.hstack((pts_canonical_space, np.ones((pts_canonical_space.shape[0], 1))))
                pts_canonical_space_homo_op = (obj.t_cam_obj @ pts_canonical_space_homo.T).T
                pts_canonical_space = pts_canonical_space_homo_op[:, :3]
                tmp_pts_canonical_space.append(pts_canonical_space)
                
                
                obj.g_pose = det.g_pose @ rott @ obj.t_cam_obj
                obj.s_pose = det.s_pose @ rott @ obj.t_cam_obj
                
                g_pose[frame_id] = obj.g_pose
                s_pose[frame_id] = obj.s_pose
                
                # Velocity
                x = obj.g_pose[0, 3]
                y = obj.g_pose[1, 3]
                z = obj.g_pose[2, 3]
                
                if prev_frame_id == -np.inf:
                    v_x = 0
                    v_y = 0
                    v_z = 0
                else:
                    num_frames = frame_id - prev_frame_id
                    time = 0.1 * (num_frames) # 10 Hz, validate this number LATER
                    v_x = (x - prev_x) / time
                    v_y = (y - prev_y) / time
                    v_z = (z - prev_z) / time

                v = np.linalg.norm(np.array([v_x, v_y, v_z]))
                # print("prev_x, prev_y, prev_z", prev_x, prev_y, prev_z)
                # print("x, y, z", x, y, z)
                # print("v_x, v_y, v_z", v_x, v_y, v_z, v)
                velocity[frame_id] = {'v': v, 'v_x': v_x, 'v_y': v_y, 'v_z': v_z}
                # print("velocity[frame_id]", velocity[frame_id])
                prev_x = x
                prev_y = y
                prev_z = z
                prev_frame_id = frame_id
                # Velocity
                
                objects_recon[frame_id] = obj
                
                accumulated_pts_g_space, _ = convert_to_world_frame(accumulated_pts_canonical_space)
                
                accumulated_pts_g_space_homo = np.hstack((accumulated_pts_g_space, np.ones((accumulated_pts_g_space.shape[0], 1))))
                accumulated_pts_g_space_homo = (det.g_pose @ accumulated_pts_g_space_homo.T).T
                accumulated_pts_g_space = accumulated_pts_g_space_homo[:, :3]

                g_point_0 = np.concatenate((g_point_0, accumulated_pts_g_space))
        
        # np.save(f'results/instance_association/PointCloud_KITTI21_Obj_ID_{id}-accumulated.npy', np.array(acc_points, dtype=object), allow_pickle=True)

        # np.save(f'results/deep_sdf/obj/{id}_accumulated.npy', np.array(objects_recon, dtype=object), allow_pickle=True)
        # np.save(f'results/deep_sdf/pose/v_{id}_accumulated.npy', np.array(velocity, dtype=object), allow_pickle=True)
        np.save(f'results/deep_sdf/pose/g_pose_{id}_accumulated.npy', np.array(g_pose, dtype=object), allow_pickle=True)
        np.save(f'results/deep_sdf/pose/s_pose_{id}_accumulated.npy', np.array(s_pose, dtype=object), allow_pickle=True)
        

        g_point_0 = g_point_0[1:]
        # np.save(f'results/deep_sdf/pcl_show/{id}_accumulated.npy', np.array(g_point_0, dtype=object), allow_pickle=True)
        
        end = get_time()
        print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))
    # else:
    #     objects_recon = np.load(f'results/deep_sdf/obj/{id}_accumulated.npy', allow_pickle='TRUE').item()
    #     velocity = np.load(f'results/deep_sdf/pose/v_{id}_accumulated.npy', allow_pickle='TRUE').item()
    #     g_pose = np.load(f'results/deep_sdf/pose/g_pose_{id}_accumulated.npy', allow_pickle='TRUE').item()
    #     g_point_0 = np.load(f'results/deep_sdf/pcl_show/{id}_accumulated.npy')
        
    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add Source LiDAR point cloud
    c_source_pcd = o3d.geometry.PointCloud()
    c_source_pcd.points = o3d.utility.Vector3dVector(g_point_0)
    BLUE_color = np.full((g_point_0.shape[0], 3), color_table[2]) # BLUE COLOR
    c_source_pcd.colors = o3d.utility.Vector3dVector(BLUE_color)
    vis.add_geometry(c_source_pcd)
    
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    # for (frame_id, obj)in objects_recon.items():
    for (frame_id, obj), (_, v) in zip(objects_recon.items(), velocity.items()):
        
        # Add Source LiDAR point cloud - global
        # g_source_pcd = o3d.geometry.PointCloud()
        # g_source_pcd.points = o3d.utility.Vector3dVector(g_point_obj)
        # green_color = np.full((g_point_obj.shape[0], 3), color_table[1]) # GREEN COLOR
        # g_source_pcd.colors = o3d.utility.Vector3dVector(green_color)
        # # g_source_pcd.transform(obj.g_pose)
        # vis.add_geometry(g_source_pcd)
        
        ######################################## Velocity ########################################
        v_v = v['v']
        v_x = v['v_x']
        v_y = v['v_y']
        v_z = v['v_z']
        
        print("v", v)
        print("obj.g_pose", obj.g_pose)
        
        x = obj.g_pose[0, 3]
        y = obj.g_pose[1, 3]
        z = obj.g_pose[2, 3] + 2
        
        text_points = text_3d(str(round(v_v, 2)), (x, y, z), (v_x/v_v, v_y/v_v, 0))
        vis.add_geometry(text_points)
        ######################################## Velocity ########################################
        
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])

        # Transform mesh from object to world coordinate
        mesh_o3d.transform(obj.g_pose)
        vis.add_geometry(mesh_o3d)
        
        # write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}-accumulated-text", "%d.ply" % frame_id))
        o3d.io.write_point_cloud(f"{save_dir}/{id}-accumulated-text/{frame_id}.pcd", text_points)
        write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(f"{save_dir}/{id}-accumulated", "%d.ply" % frame_id))
        
    
    print("FINISHED")
    # COMMENT THIS LINE, result in easy visualization.
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame) 
    
    vis.run()
    vis.destroy_window()

# python3 DEEP_SDF/reconstruct_object_accumulated.py

if __name__ == "__main__":
    main()
    
    
# https://github.com/isl-org/Open3D/issues/2
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d