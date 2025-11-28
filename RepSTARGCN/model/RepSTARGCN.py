from torch import nn
import torch
from model.layers import *
from model.init_transforms import Transforms


class RepSTARGCN(nn.Module):
    def __init__(self, only_tcn, num_classes, num_joint, seg, bias=True, dim=256, adaptive_transform=False,
                 num_joint_ori=25,
                 gcn_type='mid', use_reparameterization=False):
        super(RepSTARGCN, self).__init__()
        self.only_tcn = only_tcn
        self.seg = seg
        if only_tcn:
            self.Ot = Only_tcn(seg, num_joint, bias, dim)
        else:
            self.spa_net = SpatialNet(num_joint, bias, dim, gcn_type, use_reparameterization=use_reparameterization)
            self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joints, m = input.shape
            input = input.view(bs * s, c, step, num_joints, m)
        elif len(input.shape) == 5:
            bs, c, step, num_joints, m = input.shape
            s = 1
            input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joints, step)
        else:
            bs, step, num_joints, c = input.shape
            s = 1
            m = 1
            input = input.permute(0, 3, 2, 1).contiguous()

        angle = input[:, 0:3, :, :]
        dif = torch.cat(
            [torch.zeros([*input.shape[:3], 1], device=input.device), input[:, :, :, 1:] - input[:, :, :, 0:-1]],
            dim=-1)

        if self.only_tcn:
            input = self.Ot(angle, dif)
        else:
            input = self.spa_net(angle, dif)
            input = self.tem_net(input)

        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        output = output.view(bs, m * s, -1).mean(1)

        return output


