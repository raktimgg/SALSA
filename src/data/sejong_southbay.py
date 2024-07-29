import copy
import gc
import json
import math
import os
import pickle
import random
from itertools import repeat
from typing import List, Tuple, Union

import faiss
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from data.datasets.base_datasets import TrainingTuple, get_pointcloud_loader
from utils.misc_utils import collate_fn
from utils.o3d_utils import get_matching_indices, make_open3d_point_cloud


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

class SejongSouthbayLoader(Dataset):
    def __init__(self, all_file_loc = None, pcl_transform=None):
        if all_file_loc==None:
            with open('/data/raktim/Datasets/Apollo-Southbay/train_southbay_2_10.pickle', 'rb') as file:
                self.southbay_data_dict = pickle.load(file)
            with open('/data/raktim/Datasets/Mulran/Sejong/train_Sejong1_Sejong2_2_10.pickle', 'rb') as file:
                self.sejong_data_dict = pickle.load(file)


            self.all_file_loc = []
            self.pos_pairs_ind = []
            self.non_negative_pairs_ind = []
            self.transforms = []
            self.is_southbay = []

            ############# For Southbay ##################################
            for i in range(len(self.southbay_data_dict)):
                self.all_file_loc.append(self.southbay_data_dict[i].rel_scan_filepath)
                self.transforms.append(self.southbay_data_dict[i].pose)   ## check the form of transform
                self.pos_pairs_ind.append(self.southbay_data_dict[i].positives.tolist())
                self.non_negative_pairs_ind.append(self.southbay_data_dict[i].non_negatives.tolist())
                self.is_southbay.append(1)
            ############# For Sejong ####################################
            len_southbay = len(self.southbay_data_dict)

            for i in range(len(self.sejong_data_dict)):
                self.all_file_loc.append(self.sejong_data_dict[i].rel_scan_filepath)
                self.transforms.append(self.sejong_data_dict[i].pose)   ## check the form of transform
                pos_pairs = (np.array(self.sejong_data_dict[i].positives)+len_southbay).tolist()
                self.pos_pairs_ind.append(pos_pairs)
                non_neg_pairs = (np.array(self.sejong_data_dict[i].non_negatives) + len_southbay).tolist()
                self.non_negative_pairs_ind.append(non_neg_pairs)
                self.is_southbay.append(0)

            self.transforms = np.array(self.transforms)
        else:
            self.all_file_loc = all_file_loc

        self.voxel_size = 0.5
        self.pcl_transform = pcl_transform
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
        if self.pcl_transform is not None:
            xyzr = self.pcl_transform(xyzr)
        if len(xyzr)>0:
            coords, xyz, feats = self.data_prepare(xyzr,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))
        else:
            coords = torch.FloatTensor(np.ones([100,3])*(-1))
            xyz = torch.FloatTensor(np.ones([100,3])*(-1))
            feats = torch.FloatTensor(np.ones([100,3])*(-1))
        return coords, xyz, feats



class SejongSouthbayTupleLoader(SejongSouthbayLoader):
    def __init__(self,cached_queries = 1000, pcl_transform=None):
        super().__init__(pcl_transform=pcl_transform)
        self.cached_queries = cached_queries
        self.nNeg = 5
        self.margin = 0.1
        self.prev_epoch_neg=(-np.ones((len(self.all_file_loc),self.nNeg),dtype=int)).tolist()

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.all_file_loc) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.all_file_loc))

        arr = np.random.permutation(arr)

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

    def update_subcache(self, net, outputdim):

        # reset triplets
        self.triplets = []

        # take n query indices
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # get corresponding positive indices
        pos_samples = []
        queries = []
        neg_samples = []

        pidxs = []
        all_pidxs = []
        k = 0
        for i in qidxs:
            queries.append(self.all_file_loc[i])

            all_pidxs.extend(self.pos_pairs_ind[i])
            pidx = np.random.choice(self.pos_pairs_ind[i])
            pidxs.append(pidx)
            k = k+1
        pidxs = np.unique(np.array(pidxs))

        for pidx in pidxs:
            pos_samples.append(self.all_file_loc[pidx])

        all_pidxs = np.unique(np.array(all_pidxs)).tolist()

        set1 = set(np.arange(len(self.all_file_loc)).tolist())
        set2 = set(all_pidxs)
        neg_smapling_set = list(set1 - set2)

        nidxs = np.random.choice(neg_smapling_set, self.cached_queries*4, replace=False)

        for nidx in nidxs:
            neg_samples.append(self.all_file_loc[nidx])
        np.set_printoptions(threshold=np.inf)

        # # make dataloaders for query, positive, and negative
        batch_size = 128
        q_dset = SejongSouthbayLoader(all_file_loc=queries)
        q_loader = DataLoader(q_dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

        p_dset = SejongSouthbayLoader(all_file_loc=pos_samples)
        p_loader = DataLoader(p_dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

        n_dset = SejongSouthbayLoader(all_file_loc=neg_samples)
        n_loader = DataLoader(n_dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

        device = 'cuda'
        # calculate their descriptors
        net = net.to(device)
        net.eval()
        with torch.inference_mode():

            # initialize descriptors
            qvecs = np.zeros((len(q_loader.dataset), outputdim), dtype=np.float32)
            pvecs = np.zeros((len(p_loader.dataset), outputdim), dtype=np.float32)
            nvecs = np.zeros((len(n_loader.dataset), outputdim), dtype=np.float32)

            # compute descriptors and mine hard negatives
            print('Mining hard negatives')
            count = 0
            for i, batch_data in tqdm(enumerate(n_loader),total = len(n_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device)
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                nvecs[count:count+batch_number[-1]+1, :] = global_descriptor.cpu().numpy()
                count += batch_number[-1]+1
                del coord, xyz, feat, batch_number, batch_data, local_features, global_descriptor
                gc.collect()
                torch.cuda.empty_cache()

            count = 0
            for i, batch_data in tqdm(enumerate(q_loader),total = len(q_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device)
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                qvecs[count:count+batch_number[-1]+1, :] = global_descriptor.cpu().numpy()
                count += batch_number[-1]+1
                del coord, xyz, feat, batch_number, batch_data, local_features, global_descriptor
                gc.collect()
                torch.cuda.empty_cache()

            count = 0
            for i, batch_data in tqdm(enumerate(p_loader),total = len(p_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device)
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                pvecs[count:count+batch_number[-1]+1, :] = global_descriptor.cpu().numpy()
                count += batch_number[-1]+1
                del coord, xyz, feat, batch_number, batch_data, local_features, global_descriptor
                gc.collect()
                torch.cuda.empty_cache()

        faiss_index = faiss.IndexFlatL2(outputdim)
        faiss_index.add(nvecs)
        dNeg_arr, n_ind_arr = faiss_index.search(qvecs, self.nNeg+1)  # nNeg+1 negatives that are closest to query
        # dNeg_arr - distance matrix
        # n_ind_arr - corresponding indices
        for q in range(len(qidxs)):
            qidx = qidxs[q]
            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pos_pairs_ind[qidx]))
            # cached_pidx: indices of pidxs corresponding to the ones in self.pos_pairs_ind[qidx]
            faiss_index = faiss.IndexFlatL2(outputdim)
            faiss_index.add(pvecs[cached_pidx])
            dPos, p_ind = faiss_index.search(qvecs[q:q+1], 1)
            pidx = pidxs[list(cached_pidx[0])[p_ind.item()]]
            loss = dPos.reshape(-1) - dNeg_arr[q,:].reshape(-1) + self.margin
            violatingNeg = loss>0
            if self.prev_epoch_neg[qidx][0]==-1:
                # if less than nNeg are violating then skip this query
                if np.sum(violatingNeg) <= self.nNeg:
                    continue
                else:
                    # select hardest negatives and update prev_epoch_neg
                    hardest_negIdx = np.argsort(loss)[:self.nNeg]
                    # select the hardest negatives
                    hardestNeg = nidxs[n_ind_arr[q,hardest_negIdx]]
            else:
                #At least n/2 new negatives
                if np.sum(violatingNeg) <= math.ceil(self.nNeg/2):
                    continue
                else:
                    # hardest negatives from a random image pool and previous epoch
                    hardest_negIdx = np.argsort(loss)[:min(self.nNeg,np.sum(violatingNeg))]
                    cached_hardestNeg = nidxs[n_ind_arr[q,hardest_negIdx]]
                    neg_candidates = np.asarray([x for x in cached_hardestNeg if x not in self.prev_epoch_neg[qidx]]+self.prev_epoch_neg[qidx])
                    hardestNeg= neg_candidates[random.sample(range(len(neg_candidates)),self.nNeg)]
            self.prev_epoch_neg[qidx]= np.copy(hardestNeg).tolist()

            # transform back to original index (back to original idx domain)
            q_loc = self.all_file_loc[qidx]
            p_loc = self.all_file_loc[pidx]
            n_loc = self.all_file_loc[hardestNeg[0]]

            # package the triplet and target
            triplet_id = [qidx,pidx,hardestNeg[0]]
            triplet = [q_loc, p_loc, n_loc]
            target = [-1, 1, 0]
            self.triplets.append((triplet, triplet_id, target))

    def __len__(self):
        return len(self.triplets)

    def base_2_lidar(self, wTb):
        bTl = np.asarray([-0.999982947984152,  -0.005839838492430,   -0.000005225706031,  1.7042,
                          0.005839838483221,   -0.999982947996283,   0.000001775876813,   -0.0210,
                          -0.000005235987756,  0.000001745329252,    0.999999999984769,  1.8047,
                          0, 0, 0, 1]
                         ).reshape(4, 4)
        return wTb @ bTl

    def get_delta_pose(self, transforms,filename):
        if filename[22]=='A':
            w_T_p1 = transforms[0]
            w_T_p2 = transforms[1]
        else:
            w_T_p1 = self.base_2_lidar(transforms[0])
            w_T_p2 = self.base_2_lidar(transforms[1])

        p1_T_w = np.linalg.inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        return p1_T_p2

    def get_point_tuples(self, q_xyz, p_xyz, q_idx, p_idx,filename):
        q_pcd = make_open3d_point_cloud(q_xyz, color=None)
        p_pcd = make_open3d_point_cloud(p_xyz, color=None)

        matching_search_voxel_size = min(self.voxel_size*1.5, 0.1)

        q_odom = self.transforms[q_idx]
        p_odom = self.transforms[p_idx]
        all_odometry = [q_odom, p_odom]

        delta_T = self.get_delta_pose(all_odometry,filename)
        p_pcd.transform(delta_T)

        reg = o3d.pipelines.registration.registration_icp(
            p_pcd, q_pcd, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        p_pcd.transform(reg.transformation)

        pos_pairs = get_matching_indices(
            q_pcd, p_pcd, matching_search_voxel_size)
        # assert pos_pairs.ndim == 2, f"No pos_pairs for {query_id} in drive id: {drive_id}"

        return pos_pairs

    def __getitem__(self,idx):
        anchor_filename, pos_filename, neg_filename = self.triplets[idx][0]
        anchor_idx, pos_idx, neg_idx = self.triplets[idx][1]
        labels = self.triplets[idx][1]

        anchor_xyzr = self.read_pcd_file(anchor_filename)
        anchor_coords, anchor_xyz, anchor_feats = self.data_prepare(anchor_xyzr,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))

        pos_xyzr = self.read_pcd_file(pos_filename)
        pos_coords, pos_xyz, pos_feats = self.data_prepare(pos_xyzr,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))

        point_pos_pairs = self.get_point_tuples(anchor_xyz, pos_xyz, anchor_idx, pos_idx, anchor_filename)

        neg_xyzr = self.read_pcd_file(neg_filename)
        neg_coords, neg_xyz, neg_feats = self.data_prepare(neg_xyzr,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))

        return anchor_coords, anchor_xyz, anchor_feats, pos_coords, pos_xyz, pos_feats, neg_coords, neg_xyz, neg_feats, labels, point_pos_pairs



