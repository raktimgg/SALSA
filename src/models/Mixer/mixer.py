import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class Mixer(nn.Module):
    def __init__(self,
                 in_channels=35000,
                 out_channels=1000,
                 in_d=30,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_d=4,
                 ) -> None:
        super().__init__()

        self.in_d = in_d

        self.out_d = out_d # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=in_d, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.row_proj = nn.Linear(in_d, out_d)
        self.channel_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x = x.unsqueeze(0)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():

    model = Mixer().to('cuda')
    print_nb_params(model)


if __name__ == '__main__':
    main()
