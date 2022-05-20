import time

import torch
from torch.utils.data.dataloader import default_collate

from datasets.KITTI360Dataset import KITTI3603DDictPairs
from datasets.KITTIDataset import KITTILoader3DDictPairs
import torch.utils.data


def datasets_concat_kitti(data_dir, sequences_list, transforms, data_type, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        poses_file = data_dir + "/sequences/" + sequence + "/poses.txt"
        if data_type == "3D":
            d = KITTILoader3DDictPairs(data_dir, sequence, poses_file, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti360(data_dir, sequences_list, transforms, data_type, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        if data_type == "3D":
            d = KITTI3603DDictPairs(data_dir, sequence, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def merge_inputs(queries):
    anchors = []
    positives = []
    negatives = []
    anchors_logits = []
    positives_logits = []
    negatives_logits = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'anchor' and key != 'positive' and key != 'negative' and key != 'anchor_logits'
               and key != 'positive_logits' and key != 'negative_logits'}
    for input in queries:
        if 'anchor' in input:
            anchors.append(input['anchor'])
        if 'positive' in input:
            positives.append(input['positive'])
        if 'negative' in input:
            negatives.append(input['negative'])
        if 'anchor_logits' in input:
            anchors_logits.append(input['anchor_logits'])
        if 'positive_logits' in input:
            positives_logits.append(input['positive_logits'])
        if 'negative_logits' in input:
            negatives_logits.append(input['negative_logits'])

    if 'anchor' in input:
        returns['anchor'] = anchors
    if 'positive' in input:
        returns['positive'] = positives
    if 'negative' in input:
        returns['negative'] = negatives
    if 'anchor_logits' in input:
        returns['anchor_logits'] = anchors_logits
    if 'positive_logits' in input:
        returns['positive_logits'] = positives_logits
    if 'negative_logits' in input:
        returns['negative_logits'] = negatives_logits
    return returns


class Timer(object):
    """A simple timer."""

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0

    @property
    def avg(self):
        if self.calls > 0:
            return self.total_time / self.calls
        else:
            return 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff

