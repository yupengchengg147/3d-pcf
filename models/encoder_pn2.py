import torch.nn as nn
import torch.nn.functional as F

from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class PNT_2(nn.Module):
    def __init__(self, npoint,df=3):
        super(PNT_2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(int(npoint/8), [0.05, 0.1], [16, 32], df, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(int(npoint/16), [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        # self.sa3 = PointNetSetAbstractionMsg(int(npoint/16), [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        # self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        # self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        # self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        # self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        # print("l1 fea shape: ", l1_points.shape)
        # print("l1_xyz shape: ", l1_xyz.shape)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # print("l2 fea shape: ", l2_points.shape)
        # print("l2_xyz shape: ", l2_xyz.shape)

        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # print("l3 fea shape: ", l3_points.shape)
        # print("l3_xyz shape: ", l3_xyz.shape)

        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # print("l4 fea shape: ", l4_points.shape)
        # print("l4_xyz shape: ", l4_xyz.shape)

        # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)

        return l2_xyz, l2_points

        # return x, l4_points

