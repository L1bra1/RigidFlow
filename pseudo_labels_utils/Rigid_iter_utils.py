import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils


def rigid_transformation_gen(pc1, pc2):
    pc1_mean = torch.mean(pc1, dim=1, keepdim=True)
    pc2_mean = torch.mean(pc2, dim=1, keepdim=True)

    pc1_moved = pc1 - pc1_mean
    pc2_moved = pc2 - pc2_mean

    X = pc1_moved
    Y = pc2_moved

    # Cross-covariance matrix
    H = torch.matmul(X, Y.transpose(1, 0))

    # Update rigid estimate in closed-form
    [U, S, V] = torch.svd(H)
    R = torch.matmul(V, U.transpose(1, 0))
    t = pc2_mean - torch.matmul(R, pc1_mean)
    return R, t


def pseudo_label_gen_per_sample(pc1, nn_pc2, voxel_label, intial_pseudo_gt,
                                max_num_points_in_voxel=600, min_num_points_in_voxel=20):

    num_voxel = torch.max(voxel_label) + 1
    pseudo_gt = torch.zeros_like(intial_pseudo_gt)

    # Updating pseudo labels for each voxel
    for index_voxel in range(num_voxel):
        mask = torch.where(voxel_label == index_voxel)[0]
        num_points_in_voxel = len(mask)

        if num_points_in_voxel > min_num_points_in_voxel:
            points_in_voxel = pc1[:, mask]
            nn_points_in_pc2 = nn_pc2[:, mask]

            if num_points_in_voxel > max_num_points_in_voxel:
                selected_points_in_voxel = points_in_voxel[:, 0:max_num_points_in_voxel]
                selected_nn_points_in_pc2 = nn_points_in_pc2[:, 0:max_num_points_in_voxel]
            else:
                selected_points_in_voxel = points_in_voxel
                selected_nn_points_in_pc2 = nn_points_in_pc2

            # Updating rigid transformation estimate
            [R, t] = rigid_transformation_gen(selected_points_in_voxel, selected_nn_points_in_pc2)

            # Generating pseudo labels for each voxel
            pseudo_gt[:, mask] = torch.matmul(R, points_in_voxel) + t - points_in_voxel
        else:
            pseudo_gt[:, mask] = intial_pseudo_gt[:, mask]


    return pseudo_gt



def nn_search(warped_pc1_t, pc2_t):
    _, idx = pointnet2_utils.knn(1, warped_pc1_t.permute(0, 2, 1).contiguous(), pc2_t.permute(0, 2, 1).contiguous())
    nn_pc2 = pointnet2_utils.grouping_operation(pc2_t.contiguous(), idx)
    nn_pc2 = nn_pc2.squeeze(-1)
    return nn_pc2


def Pseudo_label_gen_module(pc1, flow_pred, voxel_label_1, pc2, iter = 1):
    """
    Pseudo label generation
    ----------
    Input:
        pc1, pc2: Input points position, [B, N, 3]
        flow_pred: Scene flow prediction, [B, N, 3]
        voxel_label_1: Supervoxel label for each point in pc1, [B, N]
        iter: Iteration number
    -------
    Returns:
        pseudo_gt: Pseudo labels, [B, N, 3]
    """

    pc1 = pc1.permute(0, 2, 1).contiguous()
    pc2 = pc2.permute(0, 2, 1).contiguous()
    flow_pred = flow_pred.permute(0, 2, 1).contiguous()
    batch_size = pc1.size(0)

    # Initializing by predicted flow.
    pseudo_gt = flow_pred.clone().detach()

    # Iteratively generate pseudo labels
    for index_reg in range(iter):
        # Updating point mapping by nearest neighbor search
        nn_pc2 = nn_search(pc1 + pseudo_gt, pc2)

        # Updating pseudo labels per sample
        for index in range(batch_size):
            pseudo_gt[index, :, :] = \
                pseudo_label_gen_per_sample(pc1[index, :, :], nn_pc2[index, :, :], voxel_label_1[index, :], pseudo_gt[index, :, :])
    return pseudo_gt.permute(0, 2, 1).contiguous()

