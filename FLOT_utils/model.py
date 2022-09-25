"""
Our implemented FLOT model.
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from FLOT_utils.util import PointNetSetAbstraction
from FLOT_utils.ot import sinkhorn


class FLOT(nn.Module):
    def __init__(self, nb_iter):
        super(FLOT,self).__init__()

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.nb_iter = nb_iter

        self.sa1 = PointNetSetAbstraction(nsample=32, in_channel=3, mlp=[32, 32, 32])
        self.sa2 = PointNetSetAbstraction(nsample=32, in_channel=32, mlp=[64, 64, 64])
        self.sa3 = PointNetSetAbstraction(nsample=32, in_channel=64, mlp=[128, 128, 128])

        self.rf1 = PointNetSetAbstraction(nsample=32, in_channel=3, mlp=[32, 32, 32])
        self.rf2 = PointNetSetAbstraction(nsample=32, in_channel=32, mlp=[64, 64, 64])
        self.rf3 = PointNetSetAbstraction(nsample=32, in_channel=64, mlp=[128, 128, 128])


        self.fc = nn.Conv1d(128, 3, kernel_size=1, bias=True)


    def forward(self, pc1, pc2):
        pc1 = pc1.cuda().transpose(2, 1).contiguous()
        pc2 = pc2.cuda().transpose(2, 1).contiguous()

        _, l1_feature1, idx1 = self.sa1(pc1, pc1, idx = None)
        _, l2_feature1, _ = self.sa2(pc1, l1_feature1, idx1)
        _, l3_feature1, _ = self.sa3(pc1, l2_feature1, idx1)
        
        _, l1_feature2, idx2 = self.sa1(pc2, pc2, idx = None)
        _, l2_feature2, _ = self.sa2(pc2, l1_feature2, idx2)
        _, l3_feature2, _ = self.sa3(pc2, l2_feature2, idx2)

        # Optimal transport
        ot_pc1_t = pc1.cuda().transpose(2, 1).contiguous()
        ot_pc2_t = pc2.cuda().transpose(2, 1).contiguous()

        transport = sinkhorn(
            l3_feature1.cuda().transpose(2, 1).contiguous(),
            l3_feature2.cuda().transpose(2, 1).contiguous(),
            ot_pc1_t,
            ot_pc2_t,
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
        )
        row_sum = transport.sum(-1, keepdim=True)

        # Estimate flow with transport plan
        ot_flow_t = (transport @ ot_pc2_t) / (row_sum + 1e-8) - ot_pc1_t
        ot_flow = ot_flow_t.cuda().transpose(2, 1).contiguous()

        # refine
        _, l1_ot_flow, _ = self.rf1(pc1, ot_flow, idx = idx1)
        _, l2_ot_flow, _ = self.rf2(pc1, l1_ot_flow, idx1)
        _, l3_ot_flow, _ = self.rf3(pc1, l2_ot_flow, idx1)

        # predict
        residual_flow = self.fc(l3_ot_flow)
        sf = ot_flow + residual_flow

        sf = sf.cuda().transpose(2, 1).contiguous()
        return sf

