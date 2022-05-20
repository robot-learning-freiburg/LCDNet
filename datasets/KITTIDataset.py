import h5py
import torch
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset
import os, os.path
import numpy as np
import random
import pickle

import utils.rotation_conversion as RT


def get_velo(idx, dir, sequence, jitter=False, remove_random_angle=-1, without_ground=False):
    if without_ground:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}',
                                 'velodyne_no_ground', f'{idx:06d}.h5')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
    else:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}', 'velodyne', f'{idx:06d}.bin')
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


class KITTILoader3DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, train=True, loop_file='loop_GT',
                 jitter=False, remove_random_angle=-1, without_ground=False):
        """

        :param dir: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: semantic-KITTI ground truth poses file
        """

        self.dir = dir
        self.sequence = sequence
        self.jitter = jitter
        self.remove_random_angle = remove_random_angle
        self.without_ground = without_ground
        data = read_calib_file(os.path.join(dir, 'sequences', sequence, 'calib.txt'))
        cam0_to_velo = np.reshape(data['Tr'], (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[0:4])
                pose[1, 0:4] = torch.tensor(x[4:8])
                pose[2, 0:4] = torch.tensor(x[8:12])
                pose[3, 3] = 1.0
                pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
                poses2.append(pose.float().numpy())
        self.poses = poses2
        self.train = train

        gt_file = os.path.join(dir, 'sequences', sequence, f'{loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        anchor_pcd = torch.from_numpy(get_velo(idx, self.dir, self.sequence, self.jitter,
                                               self.remove_random_angle, self.without_ground))

        if self.train:
            x = self.poses[idx][0, 3]
            y = self.poses[idx][1, 3]
            z = self.poses[idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = idx
            negative_idx = idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[idx][0, 3]
                possible_match_pose[1] = self.poses[idx][1, 3]
                possible_match_pose[2] = self.poses[idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 4 and idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 10 and idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
                    negative_idx = i
                    cont += 1

            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.jitter,
                                                     self.remove_random_angle, self.without_ground))
            negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.jitter,
                                                     self.remove_random_angle, self.without_ground))

            sample = {'anchor': anchor_pcd,
                      'positive': positive_pcd,
                      'negative': negative_pcd}
        else:
            sample = {'anchor': anchor_pcd}

        return sample


class KITTILoader3DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, loop_file='loop_GT', jitter=False, without_ground=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTILoader3DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = int(sequence)
        self.without_ground = without_ground
        data = read_calib_file(os.path.join(dir, 'sequences', sequence, 'calib.txt'))
        cam0_to_velo = np.reshape(data['Tr'], (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[0:4])
                pose[1, 0:4] = torch.tensor(x[4:8])
                pose[2, 0:4] = torch.tensor(x[8:12])
                pose[3, 3] = 1.0
                pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
                poses2.append(pose.float().numpy())
        self.poses = poses2
        gt_file = os.path.join(dir, 'sequences', sequence, f'{loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.jitter, self.without_ground))

        #Random permute points
        random_permute = torch.randperm(anchor_pcd.shape[0])
        anchor_pcd = anchor_pcd[random_permute]

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])

        positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.jitter, self.without_ground))

        #Random permute points
        random_permute = torch.randperm(positive_pcd.shape[0])
        positive_pcd = positive_pcd[random_permute]


        if positive_idx >= len(self.poses):
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
                  'sequence': self.sequence,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx
                  }

        return sample
