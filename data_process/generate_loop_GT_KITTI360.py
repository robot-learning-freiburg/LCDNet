import argparse
import torch
from torch.utils.data import Dataset
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np


class KITTI360(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, positive_range=5., negative_range=25., hard_range=None):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.dir = dir
        self.sequence = sequence
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = []
        poses = os.path.join(dir, 'data_poses', sequence, 'cam0_to_world.txt')
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                self.frames_with_gt.append(int(x[0]))
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                poses2.append(pose.float().numpy())
        self.frames_with_gt = np.array(self.frames_with_gt, dtype=np.int16)
        poses2 = np.stack(poses2)
        self.poses = poses2
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]

        anchor_pose = torch.tensor([x, y, z])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range)
        min_range = max(0, idx-50)
        max_range = min(idx+50, self.poses.shape[0])
        positive_idxs = list(set(indices[0]) - set(range(min_range, max_range)))
        positive_idxs.sort()
        num_loop = len(positive_idxs)
        if num_loop > 0:
            positive_idxs = list(self.frames_with_gt[np.array(positive_idxs)])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.negative_range)
        indices = set(indices[0])
        negative_idxs = set(range(self.poses.shape[0])) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            hard_idxs = list(self.frames_with_gt[np.array(list(hard_idxs))])
            pass

        return num_loop, positive_idxs,\
               list(self.frames_with_gt[np.array(negative_idxs)]),\
               hard_idxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI-360', help='dataset directory')
    args = parser.parse_args()

    base_dir = args.root_folder
    for sequence in ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                     "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0006_sync",
                     "2013_05_28_drive_0007_sync", "2013_05_28_drive_0009_sync", "2013_05_28_drive_0010_sync"]:
        dataset = KITTI360(base_dir, sequence, 4, 10, [6, 10])
        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'data_poses', sequence, 'loop_GT_4m.pickle')

        for i in range(len(dataset)):
            sample, pos, neg, hard = dataset[i]
            if sample > 0.:
                idx = dataset.frames_with_gt[i]
                sample_dict = {}
                sample_dict['idx'] = idx
                sample_dict['positive_idxs'] = pos
                sample_dict['negative_idxs'] = neg
                sample_dict['hard_idxs'] = hard
                lc_gt.append(sample_dict)
        with open(lc_gt_file, 'wb') as f:
            pickle.dump(lc_gt, f)
        print(f'Sequence {sequence} done')
