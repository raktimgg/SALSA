import csv
import random

import numpy as np
import torch
import yaml


def collate_fn(batch):
    coord, xyz, feat = list(zip(*batch))
    offset, count = [], 0

    new_coord, new_xyz, new_feat = [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
    offset_ = torch.IntTensor(offset[:k]).clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch_number = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
    coords,xyz,feat = torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k])
    return coords,xyz,feat,batch_number


def tuple_collate_fn(batch):
    anchor_coords, anchor_xyz, anchor_feats, pos_coords, pos_xyz, pos_feats, neg_coords, neg_xyz, neg_feats, labels, point_pos_pairs = list(zip(*batch))
    offset, count = [], 0

    new_coord, new_xyz, new_feat, new_label, new_point_pos_pairs = [], [], [], [], []

    coord, xyz, feat = anchor_coords, anchor_xyz, anchor_feats
    for i, item in enumerate(xyz):

        count += item.shape[0]
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(labels[i][0])

    coord, xyz, feat = pos_coords, pos_xyz, pos_feats
    for i, item in enumerate(xyz):

        count += item.shape[0]
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(labels[i][1])


    coord, xyz, feat = neg_coords, neg_xyz, neg_feats
    for i, item in enumerate(xyz):

        count += item.shape[0]
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(labels[i][2])

    if point_pos_pairs!=None:
        for i, item in enumerate(point_pos_pairs):
            # item = np.array(item) + len(new_point_pos_pairs)
            # if i>0:
            #     item1 = np.array(item)[:,0] + new_coord[i-1].shape[0]
            #     item2 = np.array(item)[:,1] + new_coord[i-1].shape[0]
            new_point_pos_pairs.append(item)


    offset_ = torch.IntTensor(offset).clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch_number = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
    coords,xyz,feat,labels = torch.cat(new_coord), torch.cat(new_xyz), torch.cat(new_feat), torch.Tensor(new_label)
    return coords,xyz,feat,batch_number,labels, new_point_pos_pairs


def hashM(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


#####################################################################################
# Load poses
#####################################################################################


def load_poses_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)

    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
        line_f = [float(i) for i in line]
        P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
        transforms.append(P)
        positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray(
        [float(t)/1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts

def read_yaml_config(filename):
    with open(filename, 'r') as stream:
        try:
            # Load the YAML file
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None