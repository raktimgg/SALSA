import os
import sys
from datetime import datetime
from time import time

import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn

from models.Mixer.mixer import Mixer
from models.SphereFormer.model.unet_spherical_transformer import Semantic
from models.adappool import AdaptivePooling
from utils.misc_utils import read_yaml_config


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


class SALSA(nn.Module):
    def __init__(self,voxel_sz):
        super(SALSA, self).__init__()
        config = read_yaml_config(os.path.join(os.path.dirname(__file__),'../config/model.yaml'))
        self.k = config['aggregator']['tokens']
        feature_dim = config['feat_extractor']['feature_dim']
        patch_size = config['feat_extractor']['patch_size']
        voxel_size = [voxel_sz, voxel_sz, voxel_sz]
        patch_size = np.array([voxel_size[i] * patch_size for i in range(3)]).astype(np.float32)
        window_size = patch_size * 6
        self.do_pe = True
        self.feature_extractor = Semantic(input_c=config['feat_extractor']['input_c'],
            m=config['feat_extractor']['m'],
            classes=feature_dim,
            block_reps=config['feat_extractor']['block_reps'],
            block_residual=True,
            layers=config['feat_extractor']['layers'],
            window_size=window_size,
            window_size_sphere=np.array(config['feat_extractor']['window_size_sphere']),
            quant_size=window_size/24, 
            quant_size_sphere= np.array(config['feat_extractor']['window_size_sphere'])/24,
            rel_query=True,
            rel_key=True,
            rel_value=True,
            drop_path_rate=config['feat_extractor']['drop_path_rate'],
            window_size_scale=config['feat_extractor']['window_size_scale'],
            grad_checkpoint_layers=[],
            sphere_layers=config['feat_extractor']['sphere_layers'],
            a=config['feat_extractor']['a'],
        )

        self.attpool = AdaptivePooling(feature_dim=feature_dim,output_channels=self.k)

        self.descriptor_extractor = Mixer(in_channels=self.k,
                                            out_channels=config['aggregator']['out_channels'],
                                            in_d=feature_dim,
                                            mix_depth=config['aggregator']['mix_depth'],
                                            mlp_ratio=config['aggregator']['mlp_ratio'],
                                            out_d=config['aggregator']['out_d'])



        self.do_pe = True



    def forward(self, coord, xyz, feat, batch, save_attn_weights=False):
        ########################## Feature extractor ########################################
        batch_shape = batch[-1]+1
        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_shape)

        local_features = self.feature_extractor(sinput, xyz, batch)
        #####################################################################################

        #################### Adaptive pooling + Mixer based aggregator #####################
        padded_split_local_features = []
        _, counts = torch.unique(batch, return_counts=True)
        split_local_fetures = torch.split(local_features, list(counts)) # [(N1,16),(N2,16),(N3,16),(N4,16),...]
        for features in split_local_fetures:
            # print(features.shape)
            if save_attn_weights:
                attval, attn_weights = self.attpool(features.unsqueeze(0), return_weights=True)
                self.attn_weights = attn_weights
            else:
                attval = self.attpool(features.unsqueeze(0))
            padded_split_local_features.append(attval.squeeze(0))          ### Attention based pooling
        padded_split_local_features = torch.stack(padded_split_local_features, dim=0)
        global_descriptor = self.descriptor_extractor(padded_split_local_features)
        #####################################################################################


        return split_local_fetures, global_descriptor


if __name__=='__main__':
    import random
    import time
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # seed = 3407
    seed = 1100
    # seed = np.random.randint(10000)
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    model = SALSA(2,voxel_sz=0.5,device='cuda')
    # save_path = '/data/raktim/Projects/LPR/Main/src/checkpoints/Ablation/NewSphereMixerVoxel2/model_6.pth'
    # checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    # model.load_state_dict(checkpoint)
    model.to('cuda')

    coords = torch.IntTensor(np.random.randint(0,100,size=[11000,3])).to('cuda')
    xyz = torch.FloatTensor(np.random.rand(11000,3)).to('cuda')
    feats = torch.FloatTensor(np.random.rand(11000,3)).to('cuda')
    batch_number = torch.IntTensor(np.ones([11000])).to('cuda')
    # print(coords.shape, xyz.shape, feats.shape, batch_number.shape)
    model.eval()

    N = 1000

    with torch.inference_mode():
        # torch.cuda.synchronize()
        start = time.time()
        for i in tqdm.tqdm(range(N)):
            local_features, output_desc = model(coords, xyz, feats, batch_number)
        # torch.cuda.synchronize()
        end = time.time()
    print('Forward pass took {} seconds for {} trials'.format(end - start,N))



