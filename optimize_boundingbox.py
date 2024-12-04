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

############# CREATING A DATACLASS FOR BOUNDING BOXES #############
###################################################################
from dataclasses import dataclass

@dataclass
class BoundingBox_with_pose_n_dimensions:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rot: float


           
def translate_boxes_to_open3d_instance(bbox, crop=False):
    """
          4 -------- 6
         /|         /|
        5 -------- 3 .
        | |        | |
        . 7 -------- 1
        |/         |/
        2 -------- 0
    https://github.com/open-mmlab/OpenPCDet/blob/master/tools/visual_utils/open3d_vis_utils.py
    """
    center = [bbox.x, bbox.y, bbox.z] #  + bbox.height / 2 ???? Think about this
    lwh = [bbox.length, bbox.width, bbox.height]
    # Create a Rotation Matrix from Yaw angle that comes from bbox.rot
    yaw = bbox.rot
    bbox.rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    if not crop:
        box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)
    else:
        lwh = [bbox.length, bbox.width, bbox.height * 0.9]
        box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def tranform_oriented_bounding_box_with_T(box, T):
    """
    Input: bbox - OrientedBoundingBox object
              T - Transformation matrix 
    Output: bbox - OrientedBoundingBox object
    """
    # Since, general transformation matrix is 4*4, and oriented box doesnt support general transforms, 
    # We apply series of simple transforms to the box

    # Translate 
    box.translate(T[:3, 3])
    # Scale
    scaling_factor = np.linalg.det(T[:3, :3]) ** (1/3)
    print("scale factor", scaling_factor)
    print("scale extent before scaling:", box.extent) 
    box.extent = box.extent * scaling_factor
    print("scale extent after scaling:", box.extent)
    # Rotate
    box.rotate(T[:3, :3])
    return box







    
###################################################################
###################################################################


@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/config.yaml'))


# 2D and 3D detection and data association
def main(config):
    """
    Input - Point cloud of +the cropped car in the object frame
    Output - Visualization of different frames(Optional), return the final bouding box 
                in the object frame (Detection frame) -->> Correction factor

    Think about including bounding boxes in visualization and transformatiosn, 
    think about generating bounding boxes for the shape completed car 
    as it would be the correct size of the car atleast in the canonical frame
    """
    
    # Load deepsdf decoder and optimizer
    cfg = yaml.safe_load(open(config))
    DeepSDF_DIR = cfg['deepsdf_dir']
    decoder = config_decoder(DeepSDF_DIR)
    optimizer = Optimizer(decoder, cfg)
    
    ''' 
    Essentially, we are trying to get the T_bb_obj_wrt_detection_frame
    '''
    T_bb_obj_wrt_detection_frame = np.eye(4, dtype="float32")

    # Start timer
    start = get_time()
    car_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd_new/70.pcd"
    print("This is for the car in frame: ", os.path.basename(car_path).split(".")[0])
    file_name = os.path.basename(car_path).split(".")[0]
    # 1. Load point cloud of the cropped car in the object frame
    car = o3d.io.read_point_cloud(car_path)
    car.paint_uniform_color(color_table[0]) # Red color
    
    # (Optional) Add random transformation to the car point cloud 
    car.transform(T_general(np.deg2rad(0), np.deg2rad(0),np.deg2rad(0), 1, [0, 0.5, 0]))  ## CAR FRAME EAST, UP, BACK - XYZ FRAME - Roll|Pitch|Yaw|SCALE|TRANSLATION

    # 2. Loading a copy of the point cloud - which finally will be 
    # transformed to canonical frame by a series of transformations 
    # They consider T_constant_parity, scale down to canonical, and optimization for sdf loss
    car_canonical = copy.deepcopy(car)
    '''
    XYZ in Open3D are RGB colors respectively
    
    Bounding Boxes by Detection Networks has a frame, in this project, we call it the Object frame,
    this expects the car to be facing in the X direction, Left as Y, and Up as Z
    
    DeepSDF's Canonical frame expects the car to be facing in the -Z direction, Left as -X, and Up as Y
    '''
    # 3. Applying Constant Parity Transformation 
    T_constant_parity = T_general( -np.pi/2,0, np.pi/2, 1, [0, 0, 0]) 
    car_canonical.transform(T_constant_parity)
    print("T_bb_obj_wrt_detection_frame before T_constant_parity", "\n",  T_bb_obj_wrt_detection_frame)
    T_bb_obj_wrt_detection_frame = T_constant_parity @ T_bb_obj_wrt_detection_frame
    print("T_bb_obj_wrt_detection_frame after T_constant_parity", "\n",  T_bb_obj_wrt_detection_frame)


    '''
    DeepSDF is trained to represent objects/cars in a canonical frame of a certain dimension.
    This size is smaller than the actual size of the cars in the dataset.
    So, we need to scale down the car to the canonical size
    '''
    # 4. Scale down to canonical size
    T_canonical_scale = T_general(0, 0, 0, 1/2, [0, 0, 0])
    car_canonical.transform(T_canonical_scale) 
    car_canonical.paint_uniform_color(color_table[2]) # Blue color
    print("T_bb_obj_wrt_detection_frame before T_canonical_scale", "\n",  T_bb_obj_wrt_detection_frame)
    T_bb_obj_wrt_detection_frame = T_canonical_scale @ T_bb_obj_wrt_detection_frame
    print("T_bb_obj_wrt_detection_frame after T_canonical_scale", "\n",  T_bb_obj_wrt_detection_frame)

    # 5. (Optional) Visualize car, car_canonical and axis
    open3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    print("Type of open3d_axis", type(open3d_axis))
    print("#######################################")
    print("Current window shows the car in the object frame, car_canonical in the canonical frame, and the axis in the object frame")
    o3d.visualization.draw_geometries([open3d_axis, car, car_canonical])

    
    # 6. Using surface points of the car, optimize for sdf loss
    ''' 
    This optimization is done in the canonical frame, We initialize the point clouds relative 
    transformation wrt to canonical frame to the identity matrix(i.e. I 4*4). But, the optimization will change 
    this transformation to the optimal transformation that minimizes the sdf loss.
    This effects both pose and scale of the car surface points. 
    '''
    car_canonical_points = np.asarray(car_canonical.points) # PointCloud object to numpy array
    print("# of points on car surface", car_canonical_points.shape)
    # Optimization
    obj = optimizer.reconstruct_object(np.eye(4, dtype="float32"), car_canonical_points)
    end = get_time()
    
    print("code and pose optimized for this point cloud, time elapsed: %f seconds", end - start)
    
    # 7. Extracting correction factor - transformation
    ''' 
    Essentially this is the transformation that is applied to the car surface points so that the 
    points all lie on the surface of the DeepSDF's car mesh.

    '''
    ''' 
    Ignore the terminology of t_cam_obj, this optimizer is reused from the DSP-SLAM project.
    In the context of this project, this transformation is from the deepsdf's canonical frame(Default frame of the DeepSDF)  
    to the object's canonical frame(Created by transformations) 
    
    # CONFUSING? IT IS! Just the error transformation that describes the alignment of the points on the car surface 
    to the implicit surface of the car

    A. DeepSDF's canonical frame is the frame in which the car is facing in the -Z direction, Left as -X, and Up as Y
    B. Object's canonical frame is the frame in which the car is approximately facing in the -Z direction, Left as -X, and Up as Y
    
    This is because, detection error propagates through the series of transformations, 
    that we apply to the car to get it to the canonical frame.
    '''
    correction_transform = obj.t_cam_obj  # Puts the car_canonical points on the surface of the car mesh
    print("correction in transform", obj.t_cam_obj)

    
    # 8. Calculate scale change in optimization 
    ''' 
    This will be useful for us when we are taking the bounding box from 
    canonical frame to the object frame and finally to the world frame
    '''
    canonical_scale = np.linalg.det(correction_transform[:3, :3])**(1/3)
    print("canonical scale", canonical_scale)
    correction_transform_without_scale = correction_transform / canonical_scale
    inv_coversion_transform_without_scale = np.linalg.inv(correction_transform_without_scale)
    T_bb_obj_wrt_detection_frame = inv_coversion_transform_without_scale @ T_bb_obj_wrt_detection_frame
    print("T_bb_obj_wrt_detection_frame after correction_transform_without_scale", "\n",  T_bb_obj_wrt_detection_frame)
    T_bb_obj_wrt_detection_frame = np.linalg.inv(T_canonical_scale) @ T_bb_obj_wrt_detection_frame
    print("T_bb_obj_wrt_detection_frame after T_canonical_scale", "\n",  T_bb_obj_wrt_detection_frame)
    T_bb_obj_wrt_detection_frame = np.linalg.inv(T_constant_parity) @ T_bb_obj_wrt_detection_frame
    print("T_bb_obj_wrt_detection_frame after T_constant_parity", "\n",  T_bb_obj_wrt_detection_frame)

    # 9. Create a new point cloud object with the optimized points in the canonical frame
    ''' 
    Optimized points are obtained by transforming car_canonical points with the correction transform.
    '''
    car_canonical_optimized_points = copy.deepcopy(car_canonical)
    car_canonical_optimized_points.transform(correction_transform)
    car_canonical_optimized_points.paint_uniform_color(color_table[1]) # Green color

    # 9.(Optional) Visualize results
    print("#######################################")
    print("Current window shows the car in canonical frame before and after optimization with Open3D axis")
    print("Green is the optimized car in canonical frame")
    print("Blue is the car in canonical frame before optimization")
    print("Note: correction transform tranforms the points to fit the shape of the car in the canonical frame")
    o3d.visualization.draw_geometries([open3d_axis, car_canonical, car_canonical_optimized_points])    
    
    # 10. Create mesh extractor
    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    mesh = mesh_extractor.extract_mesh_from_code(obj.code)
    mesh_deepsdf = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    # mesh_original.compute_vertex_normals() # Without this, the mesh will appear black
    print("#######################################")
    print("Current window shows the car points in canonical frame before and after optimization, deepsdf mesh with Open3D axis")
    print("Green is the optimized car in canonical frame")
    print("Blue is the car in canonical frame before optimization")
    o3d.visualization.draw_geometries([open3d_axis, car_canonical, car_canonical_optimized_points, mesh_deepsdf])  
    ''' 
    mesh_deepsdf is the mesh of the car in the canonical frame, and of the size mainained by the DeepSDF network.
    # We can create a bounding box around this mesh, and transform it to the object frame and then to the world frame.
    # Before working on the bounding box, lets try to take the mesh along the same transformations as the point cloud.
    '''  
    # 11. Transform mesh from DeepSDF's canonical frame to object's canonical frame
    mesh_canonical_inaccurate_frame = copy.deepcopy(mesh_deepsdf) # This is the mesh in the canonical frame, but of the size of the pointcloud before optimization
    mesh_canonical_inaccurate_frame.transform(np.linalg.inv(correction_transform))
    
    # (Optional) Visualize results
    print("#######################################")
    print("Current window shows the car points in canonical frame before and after optimization, canonical mesh with Open3D axis")
    print("Green is the optimized car in canonical frame")
    print("Blue is the car in canonical frame before optimization")
    print("Canonical mesh shown here fits the point cloud before optimization")
    o3d.visualization.draw_geometries([open3d_axis, car_canonical, car_canonical_optimized_points, mesh_canonical_inaccurate_frame])

    # 12. Transform mesh from object's canonical frame to object frame
    mesh_object_frame = copy.deepcopy(mesh_canonical_inaccurate_frame)
    mesh_object_frame.transform(np.linalg.inv(T_canonical_scale)) # scale back to real car size
    mesh_object_frame.transform(np.linalg.inv(T_constant_parity)) 
    '''
    Transform to object frame by attitude change. This is only because of the differences in convetions used by DeepSDF and the detection networks
    '''
    # (Optional) Visualize results
    print("#######################################")
    print("Current window shows the car points in object frame, points in both canonical frames, object mesh with Open3D axis")
    print("Green is the optimized car in canonical frame")
    print("Blue is the car in canonical frame before optimization")
    print("Red is the car in object frame")
    print("Object mesh shown here fits the point cloud before optimization")
    o3d.visualization.draw_geometries([open3d_axis, car, car_canonical, car_canonical_optimized_points, mesh_object_frame])


    # 13. Creating a point cloud of the car surface of the real car in the object frame
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(mesh_object_frame.vertices)
    obj_pcd.paint_uniform_color(color_table[5]) 
    obj_pcd_npy = np.asarray(obj_pcd.points)
    deepsdf_pcd = o3d.geometry.PointCloud()
    deepsdf_pcd.points = o3d.utility.Vector3dVector(mesh_deepsdf.vertices)
    deepsdf_pcd.paint_uniform_color(color_table[6])
    
    # (Optional) Visualize results
    print("#######################################")
    print("Current window shows the car points in object frame, object mesh and a completed pcd of the object mesh with Open3D axis")
    print("Red is the car in object frame")
    o3d.visualization.draw_geometries([open3d_axis, car, mesh_object_frame, obj_pcd]) 


    # (Optional) Visualize results (Bouding box of the car mesh in object frame)
    bb_object_frame = mesh_object_frame.get_oriented_bounding_box()
    print("#######################################")
    print("Current window shows the car points in object frame, object mesh and a bbox of the object mesh with Open3D axis")
    print("bbox_object_frame: ", bb_object_frame)
    # print("bbox extents: ", bb_object_frame.extent)
    l_car, w_car, h_car = bb_object_frame.extent
    print("length, breadth, height of the bb in object frame: ", l_car, w_car, h_car)
    o3d.visualization.draw_geometries([open3d_axis, car, mesh_object_frame, bb_object_frame]) 
    

    # (Optional) Visualize results: car, mesh_object frame, boudingbox, new_axis with T_bb_obj_wrt_detection_frame applied
    bb_obj_axis = copy.deepcopy(open3d_axis)
    bb_obj_axis = bb_obj_axis.transform(T_bb_obj_wrt_detection_frame)
    print("#######################################")
    print("Current window shows the car in the object frame, mesh to fit it, bouding box around it, and the axis in the object frame + Axis of bouding Box")
    o3d.visualization.draw_geometries([open3d_axis, car, mesh_object_frame, bb_obj_axis, bb_object_frame ])

    # 14. Getting the bounding box of the car aligned with the car correctly, unlike detection networks inaccuracies
    ''' 
    MAYBE DELETE THIS

    If one tries to directly compute a axis aligned bounding box of the car in the object frame, its not going to work.
    Because in object frame the point cloud maynot be aligned with the axis. 
    So, we need to compute axis aligned bounding box in the canonical frame and then transform it to the object frame.
    This is okay becuase the axis aligned bb is bound align with shape of the car in the canonical frame.
    '''

    '''
    MAYBE DELETE THIS

    bb_deepsdf_canonical = mesh_deepsdf.get_axis_aligned_bounding_box() # This is the bounding box in the canonical frame aligned with canonical frame
    '''

    # 15. Calculating dimensions of the car in canonical frame
    aabb = mesh_deepsdf.get_axis_aligned_bounding_box()
    print("aa_bb canonical: ", type(aabb))
    min_extent = np.asarray(aabb.get_min_bound())
    max_extent = np.asarray(aabb.get_max_bound())
    ''' 
    AxisAlignedBoundingBox: min: (-0.436963, -0.343641, -0.955658), max: (0.402493, 0.293094, 0.952079)

    The above are the dimensions of the bounding box in the canonical frame.
    [0] - breadth, [1] - height, [2] - length

    '''
    b, h, l = max_extent - min_extent 

    ## Creating a bounding box object with the dimensions of the car in canonical frame
    bb_with_deepsdf_size_and_open3d_pose = BoundingBox_with_pose_n_dimensions(0, 0, 0, l, b, h, 0) # Creating a bouding axis at open3d axis
    _, bb_with_deepsdf_size_and_open3d_pose = translate_boxes_to_open3d_instance(bb_with_deepsdf_size_and_open3d_pose) # Return lineset and bounding box

    print("length, breadth, height of the bb in canonical frame: ", l, b, h) # Exanple: 1.9077366590499878 0.8394559621810913 0.6367345750331879
    
    # 15. Create an axis and bouding box object for the bounding box in deepsdf's canonical frame
    bb_deepsdf_canonical_axis = copy.deepcopy(open3d_axis)
    bb_deepsdf_canonical_axis = bb_deepsdf_canonical_axis.transform(T_constant_parity)
    print("Is it possible to get the origin of an axis?", print(bb_deepsdf_canonical_axis))
    bb_with_deepsdf_size_and_deepsdf_pose = copy.deepcopy(bb_with_deepsdf_size_and_open3d_pose)
    bb_with_deepsdf_size_and_deepsdf_pose = tranform_oriented_bounding_box_with_T(bb_with_deepsdf_size_and_deepsdf_pose, T_constant_parity)
    

    ## Creating fake bounding box with double the size of bounding box in canonical frame
    T_double_size = T_general(0, 0, 0,2, [0, 0, 0])
    print("T_double_size", T_double_size)

    # 16. Creating Object Canonical PCD
    obj_canonical_pcd = copy.deepcopy(deepsdf_pcd)
    obj_canonical_pcd.transform(np.linalg.inv(correction_transform))

    # (Optional) Visualize results
    o3d.visualization.draw_geometries([open3d_axis, deepsdf_pcd, bb_with_deepsdf_size_and_deepsdf_pose])

    # 16. Transform bounding box from canonical frame to object's canonical frame(Basically, the object's canonical frame is the bounding box in canonical frame aligned to car point cloud before optimization)
    bb_object_canonical_axis = copy.deepcopy(bb_deepsdf_canonical_axis)
    bb_object_canonical_axis = bb_object_canonical_axis.transform(np.linalg.inv(correction_transform))


    # 17. Transform bounding box from object's canonical frame to object frame by constant parity and constant scale factor
    bb_object_axis = copy.deepcopy(bb_object_canonical_axis)
    bb_object_axis = bb_object_axis.transform(np.linalg.inv(T_canonical_scale))
    bb_object_axis = bb_object_axis.transform(np.linalg.inv(T_constant_parity))

    
    ''' 
    # Retiring the bouding box creation by transformation and only using dimensions and axis 
    # to track the bounding box from deepsdf's canonical frame to object's canonical frame and then to object frame

    bb_in_object_canonical_frame_with_size_before_optimization = copy.deepcopy(bb_with_deepsdf_size_and_deepsdf_pose)
    bb_in_object_canonical_frame_with_size_before_optimization = tranform_oriented_bounding_box_with_T(bb_in_object_canonical_frame_with_size_before_optimization, np.linalg.inv(correction_transform))
    '''
    
    inv_coversion_transform = np.linalg.inv(correction_transform)
    print("inv_coversion_transform scale", np.linalg.det(inv_coversion_transform[:3, :3])**(1/3))

    # (Optional) Visualize results
    # o3d.visualization.draw_geometries([open3d_axis, deepsdf_pcd, bb_with_deepsdf_size_and_deepsdf_pose, car_canonical, bb_in_object_canonical_frame_with_size_before_optimization])
    o3d.visualization.draw_geometries([bb_deepsdf_canonical_axis, deepsdf_pcd,  bb_with_deepsdf_size_and_deepsdf_pose])

    # (Optional) Visualize results
    o3d.visualization.draw_geometries([bb_object_canonical_axis, car_canonical, bb_deepsdf_canonical_axis, obj_canonical_pcd])

    ''' 
    There is spme issue with the bounding boxes, so I will only track the pose and the scale of the car. 
    '''
    # (Optional) Visualize results : Only original point cloud and the mesh that fits it
    o3d.visualization.draw_geometries([open3d_axis, car, mesh_object_frame])
    

    ''' 
    Its somehow hard to track the bounding box from canonical frame to object frame. 
    So, I will only track the transformation matrix and the dimensions of the car.
    '''
    T_car_points_in_object_frame = np.eye(4, dtype="float32")
    # apply scale and parity transform
    T_car_points_in_object_canoncial_frame = T_canonical_scale @ T_constant_parity @ T_car_points_in_object_frame
    # apply inv_correction transform
    T_car_points_in_object_canoncial_frame_optimized = inv_coversion_transform @ T_car_points_in_object_canoncial_frame
    # apply parity and scale but inverse
    T_car_points_in_object_frame_optimized = np.linalg.inv(T_canonical_scale) @ np.linalg.inv(T_constant_parity) @ T_car_points_in_object_canoncial_frame_optimized
    # Remove scaling from the transformation matrix : T_car_points_in_object_frame_optimized
    overall_scale = np.linalg.det(T_car_points_in_object_frame_optimized[:3, :3])**(1/3)
    T_car_points_in_object_frame_optimized_without_scale = T_car_points_in_object_frame_optimized / overall_scale


    # Print these transformations
    print("T_car_points_in_object_frame", T_car_points_in_object_frame)
    print("T_car_points_in_object_canoncial_frame", T_car_points_in_object_canoncial_frame)
    print("T_car_points_in_object_canoncial_frame_optimized", T_car_points_in_object_canoncial_frame_optimized)
    print("T_car_points_in_object_frame_optimized", T_car_points_in_object_frame_optimized)



    ''' 
    Creating dimension of the bounding box in each frame by scaling the dimensions using scale factor
    1. DeepSDF's canonical frame
    2. Object's canonical frame
    3. Object frame
    '''
    # Object Canonical Frame
    l_object_canoncial_frame = l * 1/canonical_scale
    b_object_canoncial_frame = b * 1/canonical_scale
    h_object_canoncial_frame = h * 1/canonical_scale

    
    # Object Frame
    l_object_frame = l_object_canoncial_frame * 2
    b_object_frame = b_object_canoncial_frame * 2
    h_object_frame = h_object_canoncial_frame * 2

    # Print these dimensions
    print("dimensions of the bounding box in each frame")
    print("DeepSDF's canonical frame", l, b, h)
    print("Object's canonical frame", l_object_canoncial_frame, b_object_canoncial_frame, h_object_canoncial_frame)
    print("Object frame", l_object_frame, b_object_frame, h_object_frame)


    # 16. Create a bounding box object with the dimensions of the car in object frame
    ## TODO




    # 17. Transform bounding box from object's canonical frame to object frame
    bb_object_axis = copy.deepcopy(open3d_axis)
    bb_object_axis = bb_object_axis.transform(T_car_points_in_object_frame_optimized_without_scale)
    # Change size to match length_object_frame
    
    # (Optional) Visualize results : original car, open3d_axis, 
    o3d.visualization.draw_geometries([car, bb_object_axis, obj_pcd])

    # Creating a bounding box at T_car_points_in_object_frame_optimized and with dimensions l_object_frame, b_object_frame, h_object_frame
    translation_vector = T_car_points_in_object_frame_optimized_without_scale[:3, 3]
    rotation_matrix = T_car_points_in_object_frame_optimized_without_scale[:3, :3]
    dimensions = np.array([l_object_frame, b_object_frame, h_object_frame])
    # Create an oriented bounding box from the rotation matrix, translation vector, and dimensions
    oriented_bounding_box_object_frame_optimized = mesh_object_frame.get_oriented_bounding_box()
    # (Optional) Visualize results : original car, open3d_axis, bounding box optimized
    # o3d.visualization.draw_geometries([car, open3d_axis, bb_object_axis, obj_pcd, oriented_bounding_box_object_frame_optimized])
    #Print all the details of the oriented bounding box 
    print("oriented_bounding_box_object_frame_optimized", oriented_bounding_box_object_frame_optimized)

    # Things to save: Correction Factor - T, Dimensions of the car in object frame, mesh_object_frame
    bb_corrections_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd_new_bouding_box_corrections/" + file_name + ".npy"
    # Save the correction factor 
    saving_dict = {"T_car_points_in_object_frame_optimized": T_car_points_in_object_frame_optimized_without_scale, "dimensions": dimensions, "mesh_object_frame_pcd": obj_pcd_npy}
    np.save(bb_corrections_path, saving_dict)

    # 18. Load detection_bb_pose (x y z l w h yaw(degree)) and apply transformation to it using 
    # T_car_points_in_object_frame_optimized and dimensions and save it in a txt file for 
    # evaluation in the next step with gt which is x y z l w h yaw(radian)) - So save in radians finally
    detection_bb_pose_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd/000/det/" + file_name + ".txt"
    detection = np.loadtxt(detection_bb_pose_path)
    print("detection_before_optimization: ", "\n",detection[0], detection[1], detection[2], detection[3], detection[4], detection[5], np.deg2rad(detection[6]))
    T_detection = T_general(0,0,np.deg2rad(detection[6]), 1, [detection[0], detection[1], detection[2]])
    # print("T_detection", T_detection)
    T_detection_optimized = T_car_points_in_object_frame_optimized_without_scale @ T_detection
    detection_optimized_l = l_object_frame
    detection_optimized_w = b_object_frame
    detection_optimized_h = h_object_frame
    # yaw from transformation matrix : T_detection_optimized
    yaw_detection_optimized_radians = np.arctan2(T_detection_optimized[1, 0], T_detection_optimized[0, 0]) 

    detection_optimized = np.array([T_detection_optimized[0, 3], T_detection_optimized[1, 3], T_detection_optimized[2, 3], 
                                    detection_optimized_l, detection_optimized_w, detection_optimized_h, 
                                    yaw_detection_optimized_radians])
    print("detection_optimized: ", "\n",detection_optimized[0], detection_optimized[1], detection_optimized[2], detection_optimized[3], detection_optimized[4], detection_optimized[5], detection_optimized[6]   )
    detection_optimized_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd_new_bouding_box_after_corrections/" + file_name + ".txt"
    np.savetxt(detection_optimized_path, detection_optimized, fmt='%f')

    # 19. Load GroundTruth and print it
    gt_bb_pose_path = "/home/shashank/Documents/UniBonn/Sem2/MSR_P04/from_scratch/results/Argoverse/pcd/000/gt000016/" + file_name + ".txt"
    gt = np.loadtxt(gt_bb_pose_path)
    print("gt: ", "\n",gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6])

    # Looking at Transformation Matrices
    print("T_gt: ", "\n",T_general(0,0,gt[6], 1, [gt[0], gt[1], gt[2]]))
    print("T_detection: ", "\n",T_general(0,0,detection[6], 1, [detection[0], detection[1], detection[2]]))
    print("T_correction: ", "\n",T_car_points_in_object_frame_optimized_without_scale)
    print("T_detection_optimized: ", "\n",T_detection_optimized)




if __name__ == "__main__":
    main()
    

'''
Logic to transform bounding box from canonical frame to object frame and then to world frame
'''
#    # Convert to Oriented Bounding Box because we need to rotate it to the object frame
#     bb_deepsdf_canonical = bb_deepsdf_canonical.get_oriented_bounding_box()
#     # bb_deepsdf_canonical = bb_deepsdf_canonical.rotate(np.linalg.inv(T_constant_parity)[:3, :3]) # REMOVE THIS
#     bb_object_canonical = bb_deepsdf_canonical.translate(np.linalg.inv(correction_transform)[:3, 3])  # Translate to object frame

#     # Scale to object frame
#     scaling_factor = np.linalg.det(correction_transform[:3, :3]) ** (1/3)  # Taking cube root of determinant
#     bb_object_canonical_extent_scaled = bb_object_canonical.extent * scaling_factor
#     bb_object_canonical.extent = np.copy(bb_object_canonical_extent_scaled)
#     bb_object_canonical.rotate(np.linalg.inv(correction_transform)[:3, :3])

#     # Rotate to object frame using T_canonical_scale (assuming this is a valid transformation)
#     scaling_factor = np.linalg.det(T_canonical_scale[:3, :3]) ** (1/3)
#     bb_object_canonical_extent_scaled = bb_object_canonical.extent * 1/scaling_factor
#     bb_object_canonical.extent = np.copy(bb_object_canonical_extent_scaled)
#     bb_object_canonical.rotate(np.linalg.inv(T_canonical_scale)[:3, :3])


#     bb_object = o3d.geometry.OrientedBoundingBox()                                
#     bb_object = bb_object_canonical.rotate(np.linalg.inv(T_constant_parity)[:3, :3]) # This should be aligned with real point cloud of the car and also be of the correct size

