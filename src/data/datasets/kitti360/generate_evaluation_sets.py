# Test set for Kitti360 Sequence 09.
# This script is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/kitti/generate_evaluation_sets.py

import argparse
import os
import sys
from typing import List

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets.base_datasets import EvaluationSet, EvaluationTuple, filter_query_elements
from datasets.kitti360.kitti360_raw import Kitti360Sequence

# MAP_TIMERANGE = (0, 170)
MAP_TIMERANGE = (0, 300)

def get_scans(sequence: Kitti360Sequence, min_displacement: float = 0.1, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all point clouds from the sequence (the full sequence or test split only)

    elems = []
    old_pos = None
    count_skipped = 0
    displacements = []

    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.rel_lidar_timestamps[ndx]) or (ts_range[1] < sequence.rel_lidar_timestamps[ndx]):
                continue
        pose = sequence.lidar_poses[ndx]
        # Kitti poses are in camera coordinates system where where y is upper axis dim
        position = pose[[0,1], 3]

        if old_pos is not None:
            displacements.append(np.linalg.norm(old_pos - position))

            if np.linalg.norm(old_pos - position) < min_displacement:
                # Ignore the point cloud if the vehicle didn't move
                count_skipped += 1
                continue
        # print(sequence.rel_scan_filepath)
        item = EvaluationTuple(sequence.rel_lidar_timestamps[ndx], sequence.rel_scan_filepath[ndx], position, pose)
        elems.append(item)
        old_pos = position

    print(f'{count_skipped} clouds skipped due to displacement smaller than {min_displacement}')
    print(f'mean displacement {np.mean(np.array(displacements))}')
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, min_displacement: float = 0.1,
                            dist_threshold: float = 5.) -> EvaluationSet:

    sequence = Kitti360Sequence(dataset_root, map_sequence)

    map_set = get_scans(sequence, min_displacement, MAP_TIMERANGE)
    query_set = get_scans(sequence, min_displacement, (MAP_TIMERANGE[-1], sequence.rel_lidar_timestamps[-1]))
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for KItti dataset')
    # kitti: /mnt/088A6CBB8A6CA742/Datasets/Kitti/dataset/
    # mulran: /mnt/088A6CBB8A6CA742/Datasets/MulRan/
    # apollo:
    parser.add_argument('--dataset_root', type=str, required=False, default='/data/raktim/Datasets/KITTI360/KITTI-360/data_3d_raw')
    parser.add_argument('--min_displacement', type=float, default=3.0)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=5.)

    args = parser.parse_args()

    # Sequences are fixed
    sequence = '09'
    sequence_name = '2013_05_28_drive_00'+ sequence + '_sync'
    print(f'Dataset root: {args.dataset_root}')
    print(f'Kitti sequence: {sequence}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    kitti_eval_set = generate_evaluation_set(args.dataset_root, sequence, min_displacement=args.min_displacement,
                                             dist_threshold=args.dist_threshold)
    file_path_name = os.path.join(os.path.dirname(__file__), f'kitti360_{sequence}_{args.min_displacement}_eval.pickle')
    print(f"Saving evaluation pickle: {file_path_name}")
    kitti_eval_set.save(file_path_name)
