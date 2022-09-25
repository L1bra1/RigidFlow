"""
References:
flownet3d_pytorch: https://github.com/hyangwinter/flownet3d_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import sys
sys.path.append("..")


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.permute(0, 2, 1)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.detach().long(), :]
    return new_points.permute(0, 3, 1, 2).contiguous()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def group_query(nsample, s_xyz, xyz, s_points, idx = None):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    if idx is None:
        idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points(s_xyz.permute(0, 2, 1).contiguous(), idx.int()).permute(0, 2, 3, 1).contiguous()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if s_points is not None:
        grouped_points = index_points(s_points.permute(0, 2, 1).contiguous(), idx.int()).permute(0, 2, 3, 1).contiguous()
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm, idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))

            last_channel = out_channel


    def forward(self, xyz, points, idx = None):
        """
        In this pointnet++ like convolution, we do not downsample input points.
        ----------
        Input:
            xyz: input points position data, [B, C, N]
            points: input point features, [B, D, N]
        Return:
            new_xyz: input points position data, [B, C, N]
            new_points: output point features, [B, D', N]
            idx: index of neighboring points
        """

        xyz_t = xyz.permute(0, 2, 1).contiguous()

        new_xyz = xyz
        new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()

        points_t = points.permute(0, 2, 1).contiguous()
        new_points, grouped_xyz_norm, idx = group_query(self.nsample, xyz_t, new_xyz_t, points_t, idx)

        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.lrelu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]

        return new_xyz, new_points, idx
