# Test sets for Mulran dataset.
# This file is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/mulran/generate_evaluation_sets.py

import argparse
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from datasets.base_datasets import EvaluationSet, EvaluationTuple, filter_query_elements
from datasets.mulran.mulran_raw import MulranSequence

# # KAIST 02
# MAP_TIMERANGE = (1566535940856033867, 1566536300000000000)
# QUERY_TIMERANGE = (1566536300000000000, 1566536825534173166)
# # Riverside 01:
# # MAP_TIMERANGE = (1564718063503232284, 1564718300000000000)
# # QUERY_TIMERANGE = (1564718300000000000, 1564718603800415528)


def get_scans(sequence: MulranSequence, ts_range: tuple = None) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        if ts_range is not None:
            if (ts_range[0] > sequence.timestamps[ndx]) or (ts_range[1] < sequence.timestamps[ndx]):
                continue
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence_name: str, query_sequence_name: str, min_displacement: float = 0.2,
                            dist_threshold=20) -> EvaluationSet:
    split = 'test'
    map_sequence = MulranSequence(dataset_root, map_sequence_name, split=split, min_displacement=min_displacement)
    query_sequence = MulranSequence(dataset_root, query_sequence_name, split=split, min_displacement=min_displacement)
    print(min_displacement)

    if map_sequence_name == query_sequence_name:
        print('Wrong Wrong Wrong')
        map_set = get_scans(map_sequence, MAP_TIMERANGE)
        query_set = get_scans(query_sequence, QUERY_TIMERANGE)
    else:
        map_set = get_scans(map_sequence)
        query_set = get_scans(query_sequence)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for Mulran dataset')
    parser.add_argument('--dataset_root', type=str, required=False, default='/data/raktim/Datasets/Mulran/Sejong')
    parser.add_argument('--sequence', type=str, required=False, default='Sejong')
    parser.add_argument('--min_displacement', type=float, default=0.2)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=20)
    args = parser.parse_args()

    # Sequences is a list of (map sequence, query sequence)
    sequences = [('Sejong1', 'Sejong2')]
    if args.sequence == 'DCC':
        sequences = [('DCC1', 'DCC2')]
        args.min_displacement = 10.0
        args.dist_threshold = 5


    print(f'Dataset root: {args.dataset_root}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')


    for map_sequence, query_sequence in sequences:
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')

        test_set = generate_evaluation_set(args.dataset_root, map_sequence, query_sequence,
                                           min_displacement=args.min_displacement, dist_threshold=args.dist_threshold)

        pickle_name = f'test_{map_sequence}_{query_sequence}_{args.min_displacement}_{args.dist_threshold}.pickle'
        # file_path_name = os.path.join(args.dataset_root, pickle_name)
        file_path_name = os.path.join(os.path.dirname(__file__), pickle_name)
        print(f"Saving evaluation pickle: {file_path_name}")
        test_set.save(file_path_name)
