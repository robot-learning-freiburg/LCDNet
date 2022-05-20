import argparse
import torch
from torch.utils.data import Dataset
import pykitti
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np


class KITTILoader3DPosesOnlyLoopPositives(Dataset):

    def __init__(self, dir, sequence, poses, positive_range=5., negative_range=25., hard_range=None):
        super(KITTILoader3DPosesOnlyLoopPositives, self).__init__()

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.dir = dir
        self.sequence = sequence
        self.data = pykitti.odometry(dir, sequence)
        poses2 = []
        T_cam_velo = np.array(self.data.calib.T_cam0_velo)
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = np.zeros((4, 4))
                pose[0, 0:4] = np.array(x[0:4])
                pose[1, 0:4] = np.array(x[4:8])
                pose[2, 0:4] = np.array(x[8:12])
                pose[3, 3] = 1.0
                pose = np.linalg.inv(T_cam_velo) @ (pose @ T_cam_velo)
                poses2.append(pose)
        self.poses = np.stack(poses2)
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return len(self.data.timestamps)

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]

        anchor_pose = torch.tensor([x, y, z])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range)
        min_range = max(0, idx-50)
        max_range = min(idx+50, len(self.data.timestamps))
        positive_idxs = list(set(indices[0]) - set(range(min_range, max_range)))
        positive_idxs.sort()
        num_loop = len(positive_idxs)

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.negative_range)
        indices = set(indices[0])
        negative_idxs = set(range(len(self.data.timestamps))) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            pass

        return num_loop, positive_idxs, negative_idxs, hard_idxs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI', help='dataset directory')
    args = parser.parse_args()

    base_dir = args.root_folder
    for sequence in ["00", "03", "04", "05", "06", "07", "08", "09"]:
        poses_file = base_dir + "/sequences/" + sequence + "/poses.txt"

        dataset = KITTILoader3DPosesOnlyLoopPositives(base_dir, sequence, poses_file, 4, 10, [6, 10])
        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'sequences', sequence, 'loop_GT_4m.pickle')

        for i in range(len(dataset)):

            sample, pos, neg, hard = dataset[i]
            if sample > 0.:
                sample_dict = {}
                sample_dict['idx'] = i
                sample_dict['positive_idxs'] = pos
                sample_dict['negative_idxs'] = neg
                sample_dict['hard_idxs'] = hard
                lc_gt.append(sample_dict)
        with open(lc_gt_file, 'wb') as f:
            pickle.dump(lc_gt, f)
        print(f'Sequence {sequence} done')
