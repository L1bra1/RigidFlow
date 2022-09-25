"""
References:
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
"""

import sys, os
import os.path as osp
import numpy as np
import torch.utils.data as data
from pseudo_labels_utils.SV_gen_utils import compute_sv

__all__ = ['FT3D_s_sv']


class FT3D_s_sv(data.Dataset):
    """
    Generate the FlyingThing3D training dataset following HPLFlowNet

    Parameters
    ----------
    train (bool) : If True, creates dataset from training set, otherwise creates from test set.
    num_points (int) : Number of points in point clouds.
    data_root (str) : Path to dataset root directory.
    num_supervoxels (int) : Desired supervoxel number.
    """
    def __init__(self,
                 train,
                 num_points,
                 data_root, num_supervoxels=40, DEPTH_THRESHOLD = 35.0):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.num_points = num_points
        self.samples = self.make_dataset()
        self.DEPTH_THRESHOLD = DEPTH_THRESHOLD
        self.num_supervoxels = num_supervoxels


        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.train:
            pc1_transformed, pc2_transformed, sf_transformed, voxel_list = self.pc_loader(self.samples[index])
            sf_transformed = np.zeros_like(sf_transformed)

        else:
            pc1_transformed, pc2_transformed, sf_transformed, voxel_list = self.pc_loader(self.samples[index])


        return pc1_transformed.astype(np.float32), pc2_transformed.astype(np.float32), \
               sf_transformed.astype(np.float32), self.samples[index], voxel_list.astype(np.int32)


    def make_dataset(self):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print(useful_paths)
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        return useful_paths

    def pc_loader(self, fn):
        pos1 = np.load(os.path.join(fn, 'pc1.npy'))
        pos2 = np.load(os.path.join(fn, 'pc2.npy'))

        # multiply -1 only for subset datasets
        pos1[..., -1] *= -1
        pos2[..., -1] *= -1
        pos1[..., 0] *= -1
        pos2[..., 0] *= -1

        sf = pos2[:, :3] - pos1[:, :3]

        near_mask = np.logical_and(pos1[:, 2] < self.DEPTH_THRESHOLD, pos2[:, 2] < self.DEPTH_THRESHOLD)
        indices = np.where(near_mask)[0]

        if indices.shape[0] >= self.num_points:
            sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
            sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
        else:
            sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
            sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)


        pc1 = pos1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pos2[sampled_indices2]

        # Compute supervoxel
        voxel_list = compute_sv(pc1, self.num_supervoxels)

        return pc1, pc2, sf, voxel_list
