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

import math
import numpy as np
import torch
from reconstruct.utils import ForceKeyErrorDict, create_voxel_grid, convert_sdf_voxels_to_mesh
from reconstruct.loss import compute_sdf_loss
from reconstruct.loss_utils import decode_sdf, get_robust_res, exp_se3, exp_sim3, get_time


class Optimizer(object):
    def __init__(self, decoder, configs):
        self.decoder = decoder
        optim_cfg = configs['optimizer']
        self.k1 = optim_cfg['joint_optim']['k1']
        self.k2 = optim_cfg['joint_optim']['k2'] # SDF regularization
        self.k3 = optim_cfg['joint_optim']['k3'] # Latent code regularization
        self.k4 = optim_cfg['joint_optim']['k4'] # Rotation regularization
        self.b1 = optim_cfg['joint_optim']['b1'] #
        self.b2 = optim_cfg['joint_optim']['b2'] # Huber Norm for SDF
        self.lr = optim_cfg['joint_optim']['learning_rate']
        self.s_damp = optim_cfg['joint_optim']['scale_damping']
        self.num_iterations_joint_optim = optim_cfg['joint_optim']['num_iterations']
        self.code_len = optim_cfg['code_len']
        self.num_depth_samples = optim_cfg['num_depth_samples']

    def reconstruct_object(self, t_cam_obj, pts, code=None):
        """
        :param t_cam_obj: object pose, object-to-camera transformation
        :param pts: surface points, under camera coordinate (M, 3)
        :param rays: sampled ray directions (N, 3)
        :param depth: depth values (K,) only contain foreground pixels, K = M for KITTI
        :return: optimized opject pose and shape, saved as a dict
        """
        # Always start from zero code
        if code is None:
            # latent_vector = torch.zeros(self.code_len)
            latent_vector = torch.zeros(self.code_len).cuda()
        else:
            # latent_vector = torch.from_numpy(code[:self.code_len])
            latent_vector = torch.from_numpy(code[:self.code_len]).cuda()

        # Initial Pose Estimate
        t_cam_obj = torch.from_numpy(t_cam_obj).to(dtype=torch.float32)
        # t_obj_cam = torch.inverse(t_cam_obj)
        print("t_cam_obj initializing with I", t_cam_obj)
        
        # surface points within Omega_s
        # pts_surface = torch.from_numpy(pts).to(dtype=torch.float32)
        pts_surface = torch.from_numpy(pts).cuda().float()

        start = get_time()
        loss = 0.
        for e in range(self.num_iterations_joint_optim):
            # get depth range and sample points along the rays
            # t_cam_obj = torch.inverse(t_obj_cam)

            # 1. Compute SDF (3D) loss
            sdf_rst = compute_sdf_loss(self.decoder, pts_surface, t_cam_obj, latent_vector)
            if sdf_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_sdf, de_dc_sdf, res_sdf = sdf_rst
            robust_res_sdf, sdf_loss, _ = get_robust_res(res_sdf, self.b2)
            if math.isnan(sdf_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            loss = self.k2 * sdf_loss
            z = latent_vector.cpu()

            # Compute Jacobian and Hessia
            pose_dim = 7

            J_sdf = torch.cat([de_dsim3_sdf, de_dc_sdf], dim=-1)
            H_sdf = self.k2 * torch.bmm(J_sdf.transpose(-2, -1), J_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]
            b_sdf = -self.k2 * torch.bmm(J_sdf.transpose(-2, -1), robust_res_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]

            H = H_sdf
            H[pose_dim:pose_dim + self.code_len, pose_dim:pose_dim + self.code_len] += self.k3 * torch.eye(self.code_len)
            b = b_sdf
            b[pose_dim:pose_dim + self.code_len] -= self.k3 * z

            # add a small damping to the pose part
            H[:pose_dim, :pose_dim] += 1e0 * torch.eye(pose_dim)
            H[pose_dim-1, pose_dim-1] += self.s_damp  # add a large damping for scale
            # solve for the update vector
            dx = torch.mv(torch.inverse(H), b)
            delta_p = dx[:pose_dim]

            delta_c = dx[pose_dim:pose_dim + self.code_len]
            delta_t = exp_sim3(self.lr * delta_p)
            t_cam_obj = torch.mm(delta_t, t_cam_obj)
            # latent_vector += self.lr * delta_c
            latent_vector += self.lr * delta_c.cuda()

            print("Object joint optimization: Iter %d, loss: %f, sdf loss: %f, " % (e, loss, sdf_loss))
            # print("Latent Code: ", latent_vector.cpu().numpy())
        end = get_time()
        print("Reconstruction takes %f seconds" % (end - start))
        # t_cam_obj = torch.inverse(t_cam_obj)
        print("t_cam_obj adjustment needed", t_cam_obj)
        return ForceKeyErrorDict(t_cam_obj=t_cam_obj.numpy(),
                                 code=latent_vector.cpu().numpy(),
                                 is_good=True, loss=loss)
    
    def reconstruct_object_list(self, t_cam_obj, pts, code=None):
        """
        :param t_cam_obj: object pose, object-to-camera transformation
        :param pts: surface points, under camera coordinate (M, 3)
        :param rays: sampled ray directions (N, 3)
        :param depth: depth values (K,) only contain foreground pixels, K = M for KITTI
        :return: optimized opject pose and shape, saved as a dict
        """
        # Always start from zero code
        if code is None:
            # latent_vector = torch.zeros(self.code_len)
            latent_vector = torch.zeros(self.code_len).cuda()
        else:
            # latent_vector = torch.from_numpy(code[:self.code_len])
            latent_vector = torch.from_numpy(code[:self.code_len]).cuda()

        pose_list = []
        code_list = []
        # Initial Pose Estimate
        t_cam_obj = torch.from_numpy(t_cam_obj).to(dtype=torch.float32)
        pose_list.append(t_cam_obj.numpy())
        code_list.append(latent_vector.cpu().numpy())
        # t_obj_cam = torch.inverse(t_cam_obj)
        print("t_cam_obj initializing with I", t_cam_obj)
        
        # surface points within Omega_s
        # pts_surface = torch.from_numpy(pts).to(dtype=torch.float32)
        pts_surface = torch.from_numpy(pts).cuda().float()

        start = get_time()
        loss = 0.
        for e in range(self.num_iterations_joint_optim):
            # get depth range and sample points along the rays
            # t_cam_obj = torch.inverse(t_obj_cam)

            # 1. Compute SDF (3D) loss
            sdf_rst = compute_sdf_loss(self.decoder, pts_surface, t_cam_obj, latent_vector)
            if sdf_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_sdf, de_dc_sdf, res_sdf = sdf_rst
            robust_res_sdf, sdf_loss, _ = get_robust_res(res_sdf, self.b2)
            if math.isnan(sdf_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            loss = self.k2 * sdf_loss
            z = latent_vector.cpu()

            # Compute Jacobian and Hessia
            pose_dim = 7

            J_sdf = torch.cat([de_dsim3_sdf, de_dc_sdf], dim=-1)
            H_sdf = self.k2 * torch.bmm(J_sdf.transpose(-2, -1), J_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]
            b_sdf = -self.k2 * torch.bmm(J_sdf.transpose(-2, -1), robust_res_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]

            H = H_sdf
            H[pose_dim:pose_dim + self.code_len, pose_dim:pose_dim + self.code_len] += self.k3 * torch.eye(self.code_len)
            b = b_sdf
            b[pose_dim:pose_dim + self.code_len] -= self.k3 * z

            # add a small damping to the pose part
            H[:pose_dim, :pose_dim] += 1e0 * torch.eye(pose_dim)
            H[pose_dim-1, pose_dim-1] += self.s_damp  # add a large damping for scale
            # solve for the update vector
            dx = torch.mv(torch.inverse(H), b)
            delta_p = dx[:pose_dim]

            delta_c = dx[pose_dim:pose_dim + self.code_len]
            delta_t = exp_sim3(self.lr * delta_p)
            t_cam_obj = torch.mm(delta_t, t_cam_obj)
            # latent_vector += self.lr * delta_c
            latent_vector += self.lr * delta_c.cuda()

            print("Object joint optimization: Iter %d, loss: %f, sdf loss: %f, " % (e, loss, sdf_loss))
            # print("Latent Code: ", latent_vector.cpu().numpy())
            
            # Append to list
            pose_list.append(t_cam_obj.numpy())
            code_list.append(latent_vector.cpu().numpy())

        end = get_time()
        print("Reconstruction takes %f seconds" % (end - start))
        # t_cam_obj = torch.inverse(t_cam_obj)
        print("t_cam_obj adjustment needed", t_cam_obj)
        return ForceKeyErrorDict(t_cam_obj=t_cam_obj.numpy(),
                                 code=latent_vector.cpu().numpy(), pose_list=pose_list, code_list=code_list,
                                 is_good=True, loss=loss)

class MeshExtractor(object):
    def __init__(self, decoder, code_len=64, voxels_dim=64):
        self.decoder = decoder
        self.code_len = code_len
        self.voxels_dim = voxels_dim
        with torch.no_grad():
            # self.voxel_points = create_voxel_grid(vol_dim=self.voxels_dim)
            self.voxel_points = create_voxel_grid(vol_dim=self.voxels_dim).cuda()

    def extract_mesh_from_code(self, code):
        start = get_time()
        # latent_vector = torch.from_numpy(code[:self.code_len])
        latent_vector = torch.from_numpy(code[:self.code_len]).cuda()
        sdf_tensor = decode_sdf(self.decoder, latent_vector, self.voxel_points)
        vertices, faces = convert_sdf_voxels_to_mesh(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim))
        vertices = vertices.astype("float32")
        faces = faces.astype("int32")
        end = get_time()
        print("Extract mesh takes %f seconds" % (end - start))
        return ForceKeyErrorDict(vertices=vertices, faces=faces)