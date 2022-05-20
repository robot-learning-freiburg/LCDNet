import argparse
import os
import pickle
import time
from collections import OrderedDict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTIDataset import KITTILoader3DPoses
from evaluation.plot_PR_curve import compute_PR, compute_AP, compute_PR_pairs
from models.get_models import get_model
from utils.data import merge_inputs
from datetime import datetime

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_input(model, samples, exp_cfg, device):
    anchor_list = []
    for point_cloud in samples:
        anchor_i = point_cloud

        anchor_list.append(model.backbone.prepare_input(anchor_i))
        del anchor_i

    model_in = KittiDataset.collate_batch(anchor_list)
    for key, val in model_in.items():
        if not isinstance(val, np.ndarray):
            continue
        model_in[key] = torch.from_numpy(val).float().to(device)
    return model_in


class SamplePairs(Sampler):

    def __init__(self, data_source, pairs):
        super(SamplePairs, self).__init__(data_source)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return [self.pairs[i, 0] for i in range(len(self.pairs))]


class BatchSamplePairs(BatchSampler):

    def __init__(self, data_source, pairs, batch_size):
        # super(BatchSamplePairs, self).__init__(batch_size, True)
        self.data_source = data_source
        self.pairs = pairs
        self.batch_size = batch_size
        self.count = 0

    def __len__(self):
        return 2*len(self.pairs)

    def __iter__(self):
        self.count = 0
        while 2*self.count + self.batch_size < 2*len(self.pairs):
            current_batch = []
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch
            self.count += self.batch_size//2
        if 2*self.count < 2*len(self.pairs):
            diff = 2*len(self.pairs)-2*self.count
            current_batch = []
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch



def main_process(gpu, weights_path, args):
    global EPOCH

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    saved_params = torch.load(weights_path, map_location='cpu')

    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 6
    exp_cfg['loop_file'] = 'loop_GT_4m'
    exp_cfg['head'] = 'UOTHead'

    validation_sequences = [args.validation_sequence]
    if args.dataset == 'kitti':
        validation_dataset = KITTILoader3DPoses(args.root_folder, validation_sequences[0],
                                                os.path.join(args.root_folder, 'sequences', validation_sequences[0],'poses.txt'),
                                                train=False,
                                                loop_file=exp_cfg['loop_file'],
                                                remove_random_angle=args.remove_random_angle,
                                                without_ground=args.without_ground)
    elif args.dataset == 'kitti360':
        validation_dataset = KITTI3603DPoses(args.root_folder, validation_sequences[0],
                                             train=False, loop_file='loop_GT_4m_noneg',
                                             remove_random_angle=args.remove_random_angle,
                                             without_ground=args.without_ground)
    else:
        raise argparse.ArgumentTypeError("Unknown dataset")

    ValidationLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                            batch_size=exp_cfg['batch_size'],
                                            num_workers=2,
                                            shuffle=False,
                                            collate_fn=merge_inputs,
                                            pin_memory=True)

    model = get_model(exp_cfg, is_training=False)
    renamed_dict = OrderedDict()
    for key in saved_params['state_dict']:
        if not key.startswith('module'):
            renamed_dict = saved_params['state_dict']
            break
        else:
            renamed_dict[key[7:]] = saved_params['state_dict'][key]

    # Convert shape from old OpenPCDet
    if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != model.state_dict()['backbone.backbone.conv_input.0.weight'].shape:
        for key in renamed_dict:
            if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                if len(renamed_dict[key].shape) == 5:
                    renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)

    res = model.load_state_dict(renamed_dict, strict=True)

    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model.train()
    model = model.to(device)

    map_tree_poses = KDTree(np.stack(validation_dataset.poses)[:, :3, 3])

    emb_list_map = []
    rot_errors = []
    transl_errors = []
    time_descriptors = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    for batch_idx, sample in enumerate(tqdm(ValidationLoader)):

        model.eval()
        time1 = time.time()
        with torch.no_grad():

            anchor_list = []
            for i in range(len(sample['anchor'])):
                anchor = sample['anchor'][i].to(device)

                anchor_i = anchor

                anchor_list.append(model.backbone.prepare_input(anchor_i))
                del anchor_i

            model_in = KittiDataset.collate_batch(anchor_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

            emb = batch_dict['out_embedding']
            emb_list_map.append(emb)

        time2 = time.time()
        time_descriptors.append(time2-time1)

    emb_list_map = torch.cat(emb_list_map).cpu().numpy()

    emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
    pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)

    poses = np.stack(validation_dataset.poses)
    _, _, precision_ours_fp, recall_ours_fp = compute_PR(pair_dist, poses, map_tree_poses)
    ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
    print(weights_path)
    print(validation_sequences)
    print("Protocol 1 - Average Precision", ap_ours_fp)
    
    marker = 'x'
    markevery = 0.03
    plt.clf()
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    plt.plot(recall_ours_fp, precision_ours_fp, label='LCDNet (Protocol 1)', marker=marker, markevery=markevery)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall [%]")
    plt.ylabel("Precision [%]")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.show()
    # fig.savefig(f'./precision_recall_curve.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='./checkpoints')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--remove_random_angle', type=int, default=-1)
    parser.add_argument('--validation_sequence', type=str, default='08')
    parser.add_argument('--without_ground', action='store_true', default=False,
                        help='Use preprocessed point clouds with ground plane removed')
    args = parser.parse_args()

    main_process(0, args.weights_path, args)
