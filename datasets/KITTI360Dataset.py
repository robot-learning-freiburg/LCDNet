import h5py
import torch
from torch.utils.data import Dataset

import os, os.path
import numpy as np
import random
import pickle

import utils.rotation_conversion as RT


def get_velo(idx, dir, sequence, jitter=False, remove_random_angle=-1, without_ground=False):
    if without_ground:
        velo_path = os.path.join(dir, 'data_3d_raw', sequence,
                                 'velodyne_no_ground', f'{idx:010d}.npy')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
    else:
        velo_path = os.path.join(dir, 'data_3d_raw', sequence,
                                 'velodyne_points', 'data', f'{idx:010d}.bin')
        scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    if remove_random_angle > 0:
        azi = np.arctan2(scan[..., 1], scan[..., 0])
        cols = 2084 * (np.pi - azi) / (2 * np.pi)
        cols = np.minimum(cols, 2084 - 1)
        cols = np.int32(cols)
        start_idx = np.random.randint(0, 2084)
        end_idx = start_idx + (remove_random_angle / (360.0/2084))
        end_idx = int(end_idx % 2084)
        remove_idxs = cols > start_idx
        remove_idxs = remove_idxs & (cols < end_idx)
        scan = scan[np.logical_not(remove_idxs)]

    return scan


class KITTI3603DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, train=True, loop_file='loop_GT',
                 jitter=False, remove_random_angle=-1, without_ground=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """
        super(KITTI3603DPoses, self).__init__()

        self.dir = dir
        self.sequence = sequence
        self.jitter = jitter
        self.remove_random_angle = remove_random_angle
        self.without_ground = without_ground
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
        self.poses = poses2
        self.train = train

        gt_file = os.path.join(dir, 'data_poses', sequence, f'{loop_file}.pickle')
        self.loop_gt = []
        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            for elem in temp:
                temp_dict = {'idx': elem['idx'], 'positive_idxs': elem['positive_idxs']}
                self.loop_gt.append(temp_dict)
            del temp
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.frames_with_gt)

    def __getitem__(self, idx):
        frame_idx = self.frames_with_gt[idx]

        anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence,
                                                self.jitter, self.remove_random_angle, self.without_ground))

        if self.train:
            x = self.poses[idx][0, 3]
            y = self.poses[idx][1, 3]
            z = self.poses[idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])
            negative_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = frame_idx
            negative_idx = frame_idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[frame_idx][0, 3]
                possible_match_pose[1] = self.poses[frame_idx][1, 3]
                possible_match_pose[2] = self.poses[frame_idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 4 and frame_idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 10 and frame_idx == negative_idx:
                    negative_idx = i
                    cont += 1

            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence,
                                                     self.jitter, self.remove_random_angle, self.without_ground))
            negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence,
                                                     self.jitter, self.remove_random_angle, self.without_ground))

            sample = {'anchor': anchor_pcd,
                      'positive': positive_pcd,
                      'negative': negative_pcd}
        else:
            sample = {'anchor': anchor_pcd}

        return sample


class KITTI3603DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, loop_file='loop_GT', jitter=False, without_ground=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTI3603DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.without_ground = without_ground
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = {}
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
                poses2[int(x[0])] = pose.float().numpy()
        self.poses = poses2
        gt_file = os.path.join(dir, 'data_poses', sequence, f'{loop_file}.pickle')
        self.loop_gt = []
        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            for elem in temp:
                temp_dict = {'idx': elem['idx'], 'positive_idxs': elem['positive_idxs']}
                self.loop_gt.append(temp_dict)
            del temp
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.jitter, self.without_ground))

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])
        positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.jitter, self.without_ground))

        if positive_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        positive_pose = self.poses[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        r_anch = anchor_pose
        r_pos = positive_pose
        r_anch = RT.npto_XYZRPY(r_anch)[3:]
        r_pos = RT.npto_XYZRPY(r_pos)[3:]

        anchor_rot_torch = torch.tensor(r_anch.copy(), dtype=torch.float32)
        positive_rot_torch = torch.tensor(r_pos.copy(), dtype=torch.float32)

        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  'sequence': self.sequence_int,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx
                  }

        return sample
