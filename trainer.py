""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import logging
import open3d as o3d

import model
import utils


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class TrainerDeterministicPointNetLK:
    def __init__(self, args):
        # PointNet
        self.dim_k = args.dim_k
        # LK
        self.device = args.device
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        # network
        self.embedding = args.embedding
        self.filename = args.outfile
        
    def create_features(self):
        if self.embedding == 'pointnet':
            ptnet = model.Pointnet_Features(dim_k=self.dim_k)
        return ptnet.float()

    def create_from_pointnet_features(self, ptnet):
        return model.DeterministicPointNetLK(ptnet, self.device)

    def create_model(self):
        ptnet = self.create_features()
        return self.create_from_pointnet_features(ptnet)

    def train_one_epoch(self, ptnetlk, trainloader, optimizer, device, epoch, mode):
        ptnetlk.float()
        ptnetlk.train()
        vloss = 0.0
        gloss = 0.0
        batches = 0

        for i, data in enumerate(trainloader):
            loss, loss_pose = self.compute_loss(ptnetlk, data, device, epoch, mode)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            vloss += (loss.item())
            gloss += (loss_pose.item())
            batches += 1

        ave_vloss = float(vloss) / batches
        ave_loss_pose = float(gloss) / batches
        
        return ave_vloss, ave_loss_pose

    def eval_one_epoch(self, ptnetlk, testloader, device, epoch, mode):
        ptnetlk.eval()
        vloss = 0.0
        gloss = 0.0
        batches = 0

        for _, data in enumerate(testloader):
            loss, loss_pose = self.compute_loss(ptnetlk, data, device, epoch, mode)

            vloss += (loss.item())
            gloss += (loss_pose.item())
            batches += 1

        ave_vloss = float(vloss)/batches
        ave_loss_pose = float(gloss)/batches
        
        return ave_vloss, ave_loss_pose

    def test_one_epoch(self, ptnetlk, testloader, device, mode, data_type='synthetic', vis=False):
        ptnetlk.eval()
        rotations_gt = []
        translation_gt = []
        rotations_ab = []
        translation_ab = []
        
        for i, data in tqdm.tqdm(enumerate(testloader), total=len(testloader), ncols=73, leave=False):
            # if voxelization: VxNx3, Vx3, 1x4x4
            if data_type == 'real':
                if vis:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose, p0, p1 = data
                    p0 = p0.reshape(-1, p0.shape[2], p0.shape[3]).float().to(device)
                    p1 = p1.reshape(-1, p1.shape[2], p1.shape[3]).float().to(device)
                else:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
                voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
                voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
                voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
                voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
                gt_pose = gt_pose.float().to(device)
            else:
                p0, p1, gt_pose = data
            
            if vis:
                start_idx = 0
                end_idx = self.max_iter + 1
            else:
                start_idx = self.max_iter
                end_idx = self.max_iter + 1 
                
            for j in range(start_idx, end_idx):
                if data_type == 'real':
                    _ = model.DeterministicPointNetLK.do_forward(ptnetlk, voxel_features_p0, voxel_coords_p0,
                                voxel_features_p1, voxel_coords_p1, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)
                else:
                    _ = model.DeterministicPointNetLK.do_forward(ptnetlk, p0, None,
                                p1, None, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)

                estimated_pose = ptnetlk.g

                ig_gt = gt_pose.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                g_hat = estimated_pose.cpu().contiguous().view(-1, 4, 4).detach() # --> [1, 4, 4], p1->p0 (S->T)

                if vis == 'True':
                    # ANCHOR for visualization
                    p0_ = p0[0]
                    p1_ = p1[0]
                    p0_hat = utils.transform(estimated_pose, p1_.unsqueeze(0)).transpose(1,2)[0]   # Nx3
                    
                    pcd0 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(p0_hat.detach().cpu().numpy())
                    pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    pcd0.orient_normals_to_align_with_direction()
                    pcd0.paint_uniform_color([123/255, 89/255, 151/255])
                    
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(p0_.cpu().numpy())
                    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    pcd1.orient_normals_to_align_with_direction()
                    pcd1.paint_uniform_color([53/255,141/255,42/255])
                    
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(p1_.cpu().numpy())
                    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    pcd2.orient_normals_to_align_with_direction()
                    pcd2.paint_uniform_color([205/255, 107/255, 0/255])
                    
                    o3d.visualization.draw_geometries([pcd0, pcd1, pcd2])
                    
                # euler representation for ground truth
                tform_gt = ig_gt.squeeze().numpy().transpose()
                R_gt = tform_gt[:3, :3]
                euler_angle = Rotation.from_matrix(R_gt)
                anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
                angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
                rotations_gt.append(angle_gt)
                trans_gt_t = -R_gt.dot(tform_gt[3, :3])
                translation_gt.append(trans_gt_t)
                # euler representation for predicted transformation
                tform_ab = g_hat.squeeze().numpy()
                R_ab = tform_ab[:3, :3]
                euler_angle = Rotation.from_matrix(R_ab)
                anglez_ab, angley_ab, anglex_ab = euler_angle.as_euler('zyx')
                angle_ab = np.array([anglex_ab, angley_ab, anglez_ab])
                rotations_ab.append(angle_ab)
                trans_ab = tform_ab[:3, 3]
                translation_ab.append(trans_ab)

                dg = g_hat.bmm(ig_gt)   # if correct, dg == identity matrix.
                dx = utils.log(dg)   # --> [1, 6] (if corerct, dx == zero vector)
                dn = dx.norm(p=2, dim=1)   # --> [1]
                dm = dn.mean()
                
                LOGGER.info('test, %d/%d, %f', i, len(testloader), dm)

        utils.test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, self.filename)
        
        return 

    def compute_loss(self, ptnetlk, data, device, mode, data_type='synthetic'):    
        # 1. non-voxelization
        if data_type == 'synthetic':
            p0, p1, gt_pose = data
            p0 = p0.to(self.device)
            p1 = p1.to(self.device)
            gt_pose = gt_pose.to(device)
            r = model.DeterministicPointNetLK.do_forward(ptnetlk, p0, None,
                                p1, None, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)
        else:
            # 2. voxelization
            voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
            voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
            voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
            voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
            voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
            gt_pose = gt_pose.reshape(-1, gt_pose.shape[2], gt_pose.shape[3]).to(device)
            
            r = model.DeterministicPointNetLK.do_forward(ptnetlk, voxel_features_p0_, voxel_coords_p0_,
                    voxel_features_p1_, voxel_coords_p1_, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)

        estimated_pose = ptnetlk.g

        loss_pose = model.DeterministicPointNetLK.comp(estimated_pose, gt_pose)
        pr = ptnetlk.prev_r
        if pr is not None:
            loss_r = model.DeterministicPointNetLK.rsq(r - pr)
        else:
            loss_r = model.DeterministicPointNetLK.rsq(r)
        loss = loss_r + loss_pose

        return loss, loss_pose
