"""
References:
HPLFlowNet: https://github.com/laoreja/HPLFlowNet/blob/master/datasets/kitti.py
"""

import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['KITTI_s_test']


class KITTI_s_test(data.Dataset):
    """
    Generate the KITTI test dataset following HPLFlowNet

    Parameters
    ----------
    train (bool) : If False, creates KITTI test dataset.
    num_points (int) : Number of points in point clouds.
    data_root (str) : Path to dataset root directory.
    """
    def __init__(self,
                 train,
                 num_points,
                 data_root):
        self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        self.train = train
        assert (self.train is False)

        self.num_points = num_points
        self.samples = self.make_dataset()
        self.DEPTH_THRESHOLD = 35.0
        self.remove_ground = True

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        pc1_transformed, pc2_transformed, sf_transformed = self.pc_loader(self.samples[index])

        return pc1_transformed, pc2_transformed, sf_transformed, self.samples[index]


    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, fn):

        pc1 = np.load(osp.join(fn, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(fn, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]


        sf = pc2[:, :3] - pc1[:, :3]

        near_mask = np.logical_and(pc2[:, 2] < self.DEPTH_THRESHOLD, pc1[:, 2] < self.DEPTH_THRESHOLD)
        indices = np.where(near_mask)[0]

        try:
            sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
            sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
        except ValueError:
            sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
            sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)

        pos1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pos2 = pc2[sampled_indices2]

        return pos1, pos2, sf
