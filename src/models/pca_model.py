import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn as nn

from models.salsa import SALSA


class L2Norm(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12)

class PCAModel(nn.Module):
    def __init__(self,num_in_features,num_out_features):
        super(PCAModel, self).__init__()
        self.pca_conv = nn.Conv2d(num_in_features, num_out_features, kernel_size=(1, 1), stride=1, padding=0)
        self.layer = nn.Sequential(*[self.pca_conv, nn.Flatten(), L2Norm()])

    def forward(self,x):
        return self.layer(x)


class CombinedModel(nn.Module):
    def __init__(self, voxel_sz, num_in_features,num_out_features):
        super(CombinedModel, self).__init__()
        self.spherelpr = SALSA(voxel_sz=voxel_sz)
        self.pca_model = PCAModel(num_in_features, num_out_features)

    def forward(self,data):
        coord, xyz, feat, batch = data
        output_feats, output_desc = self.spherelpr(coord, xyz, feat, batch)
        output_desc = self.pca_model(output_desc[...,None][...,None])
        return output_feats, output_desc
        # return output_desc

