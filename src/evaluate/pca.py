import copy
import gc
import os
import pickle
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from itertools import repeat
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import eigs
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.pca_model import PCAModel
from models.salsa import SALSA
from data.datasets.base_datasets import TrainingTuple, get_pointcloud_loader
from utils.misc_utils import collate_fn
from utils.o3d_utils import make_open3d_point_cloud

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

class SejongSouthbayLoader(Dataset):
    def __init__(self, sequence,sample_size=-1):
        if sequence=='southbay':
            with open('/data/raktim/Datasets/Apollo-Southbay/train_southbay_2_10.pickle', 'rb') as file:
                self.southbay_data_dict = pickle.load(file)
            self.all_file_loc = []

            for i in range(len(self.southbay_data_dict)):
                self.all_file_loc.append(self.southbay_data_dict[i].rel_scan_filepath)

        elif sequence=='sejong':
            with open('/data/raktim/Datasets/Mulran/Sejong/train_Sejong1_Sejong2_2_10.pickle', 'rb') as file:
                self.sejong_data_dict = pickle.load(file)

            self.all_file_loc = []

            for i in range(len(self.sejong_data_dict)):
                self.all_file_loc.append(self.sejong_data_dict[i].rel_scan_filepath)
        elif sequence=='all':
            with open('/data/raktim/Datasets/Apollo-Southbay/train_southbay_2_10.pickle', 'rb') as file:
                self.southbay_data_dict = pickle.load(file)
            with open('/data/raktim/Datasets/Mulran/Sejong/train_Sejong1_Sejong2_2_10.pickle', 'rb') as file:
                self.sejong_data_dict = pickle.load(file)

            self.all_file_loc = []

            for i in range(len(self.southbay_data_dict)):
                self.all_file_loc.append(self.southbay_data_dict[i].rel_scan_filepath)

            for i in range(len(self.sejong_data_dict)):
                self.all_file_loc.append(self.sejong_data_dict[i].rel_scan_filepath)

        if sample_size!=-1:
            self.all_file_loc = random.sample(self.all_file_loc, min(sample_size, len(self.all_file_loc)))
        self.voxel_size = 0.5
        self.mulran_pc_loader = get_pointcloud_loader('mulran')
        self.southbay_pc_loader = get_pointcloud_loader('southbay')

    def data_prepare(self, xyzr, voxel_size=np.array([0.1, 0.1, 0.1])):

        lidar_pc = copy.deepcopy(xyzr)
        coords = np.round(lidar_pc[:, :3] / voxel_size)
        coords_min = coords.min(0, keepdims=1)
        coords -= coords_min
        feats = lidar_pc

        hash_vals, _, uniq_idx = self.sparse_quantize(coords, return_index=True, return_hash=True)
        coord_voxel, feat = coords[uniq_idx], feats[uniq_idx]
        coord = copy.deepcopy(feat[:,:3])

        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat

    def sparse_quantize(self, coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_hash: bool = False) -> List[np.ndarray]:
        if isinstance(voxel_size, (float, int)):
            voxel_size = tuple(repeat(voxel_size, 3))
        assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

        voxel_size = np.array(voxel_size)
        coords = np.floor(coords / voxel_size).astype(np.int32)

        hash_vals, indices, inverse_indices = np.unique(self.ravel_hash(coords),
                                                return_index=True,
                                                return_inverse=True)
        coords = coords[indices]

        if return_hash: outputs = [hash_vals, coords]
        else: outputs = [coords]

        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs[0] if len(outputs) == 1 else outputs

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x -= np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h


    def __len__(self):
        return len(self.all_file_loc)

    def read_pcd_file(self,filename):
        if filename[22]=='A':
            xyzr = self.southbay_pc_loader(filename)
        else:
            xyzr = self.mulran_pc_loader(filename)
        return xyzr

    def __getitem__(self, idx):
        filename = self.all_file_loc[idx]
        xyzr = self.read_pcd_file(filename)
        if len(xyzr)>0:
            coords, xyz, feats = self.data_prepare(xyzr,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))
        else:
            coords = torch.FloatTensor(np.ones([100,3])*(-1))
            xyz = torch.FloatTensor(np.ones([100,3])*(-1))
            feats = torch.FloatTensor(np.ones([100,3])*(-1))
        return coords, xyz, feats


class L2Norm(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12)

def find_pca(x: np.ndarray, num_pcs=None, subtract_mean=True):
    # assumes x = nvectors x ndims
    x = x.T  # matlab code is ndims x nvectors, so transpose

    n_points = x.shape[1]
    n_dims = x.shape[0]

    if num_pcs is None:
        num_pcs = n_dims

    # print('PCA for {} points of dimension {} to PCA dimension {}'.format(n_points, n_dims, num_pcs))

    if subtract_mean:
        # Subtract mean
        # print(x.shape)
        mu = np.mean(x, axis=1)
        x = (x.T - mu).T
    else:
        mu = np.zeros(n_dims)

    assert num_pcs < n_dims

    if n_dims <= n_points:
        do_dual = False
        # x2 = dims * dims
        x2 = np.matmul(x, x.T) / (n_points - 1)
    else:
        do_dual = True
        # x2 = vectors * vectors
        x2 = np.matmul(x.T, x) / (n_points - 1)

    # check_symmetric(x2)

    if num_pcs < x2.shape[0]:
        print('Compute {} eigenvectors'.format(num_pcs))
        lams, u = eigs(x2, num_pcs)
    else:
        print('Compute eigenvectors')
        lams, u = np.linalg.eig(x2)

    lams, u = np.linalg.eig(x2)

    assert np.all(np.isreal(lams)) and np.all(np.isreal(u))
    lams = np.real(lams)
    u = np.real(u)

    sort_indices = np.argsort(lams)[::-1]
    lams = lams[sort_indices]
    u = u[:, sort_indices]

    if do_dual:
        # U = x * ( U * diag(1./sqrt(max(lams,1e-9))) / sqrt(nPoints-1) );
        diag = np.diag(1. / np.sqrt(np.maximum(lams, 1e-9)))
        utimesdiag = np.matmul(u, diag)
        u = np.matmul(x, utimesdiag / np.sqrt(n_points - 1))

    return u, lams, mu



def do_pca(model,num_in_features, num_out_features, device, sequence, sample_size=-1, return_features_pca=False):
    train_dataset = SejongSouthbayLoader(sequence,sample_size)
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=24)
    out_vecs = torch.zeros((len(train_loader.dataset), num_in_features), dtype=torch.float32)
    count = 0
    with torch.inference_mode():
        for i, batch_data in tqdm(enumerate(train_loader),total = len(train_loader)):
            coord, xyz, feat, batch_number = batch_data
            coord, xyz, feat, batch_number = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device)
            local_features, global_descriptor = model(coord, xyz, feat, batch_number)
            out_vecs[count:count+batch_number[-1]+1, :] = global_descriptor.cpu()
            if return_features_pca:
                if i==0:
                    out_feature_vecs = torch.vstack(local_features).cpu()
                else:
                    out_feature_vecs = torch.vstack([out_feature_vecs,torch.vstack(local_features).cpu()])
            count += batch_number[-1]+1
            del coord, xyz, feat, batch_number, batch_data, local_features, global_descriptor
            gc.collect()
            torch.cuda.empty_cache()
    out_vecs = out_vecs.numpy()
    if return_features_pca:
        out_feature_vecs = out_feature_vecs.numpy()
    ####################### Full PCA #############################
    print('===> Compute Full PCA')
    u, lams, mu = find_pca(out_vecs, num_out_features)
    u = u[:, :num_out_features]
    lams = lams[:num_out_features]

    u = np.matmul(u, np.diag(np.divide(1., np.sqrt(lams + 1e-9))))

    utmu = np.matmul(u.T, mu)

    pca_model = PCAModel(num_in_features,num_out_features).to(device)
    pca_model.pca_conv.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1)))
    pca_model.pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu))
    ################################################################

    if return_features_pca:
        num_in_features = 32
        num_out_features = 16
        u, lams, mu = find_pca(out_feature_vecs, num_out_features)
        u = u[:, :num_out_features]
        lams = lams[:num_out_features]

        u = np.matmul(u, np.diag(np.divide(1., np.sqrt(lams + 1e-9))))
        utmu = np.matmul(u.T, mu)

        feature_pca_model = PCAModel(num_in_features,num_out_features).to(device)
        feature_pca_model.pca_conv.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1)))
        feature_pca_model.pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu))
        return pca_model, feature_pca_model
    return pca_model, None

def main():
    return_feature_pca = False
    seed = 1100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = 'cuda'
    model = SALSA(voxel_sz=0.5)
    load_path = os.path.join(os.path.dirname(__file__),'../checkpoints/SALSA/Model/model_26.pth')
    checkpoint = torch.load(load_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    pca,pca_feature = do_pca(model,num_in_features=512,num_out_features=256,device=device,sequence='all',sample_size=-1,return_features_pca=return_feature_pca)
    torch.save(pca.state_dict(),os.path.join(os.path.dirname(__file__),'../checkpoints/SALSA/PCA/pca_model.pth'))
    if return_feature_pca:
        torch.save(pca_feature.state_dict(),os.path.join(os.path.dirname(__file__),'../checkpoints/SALSA/PCA/pca_feature_model.pth'))




if __name__ == '__main__':
    main()
