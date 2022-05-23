import argparse
import os
from collections import OrderedDict

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from scipy.spatial import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTIDataset import KITTILoader3DPoses
from models.get_models import get_model
from utils.data import merge_inputs, Timer
from datetime import datetime
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT

import open3d as o3d
if hasattr(o3d, 'pipelines'):
    reg_module = o3d.pipelines.registration
else:
    reg_module = o3d.registration

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_database_embs(model, sample, exp_cfg, device):
    model.eval()
    margin = exp_cfg['margin']

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

        batch_dict = model(model_in, metric_head=False)
        anchor_out = batch_dict['out_embedding']


    if exp_cfg['norm_embeddings']:
        anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
    return anchor_out


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
        self.pairs = pairs
        self.batch_size = batch_size
        self.count = 0

    def __len__(self):
        tot = 2*len(self.pairs)
        ret = (tot + self.batch_size - 1) // self.batch_size
        return ret

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
    rank = gpu

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    saved_params = torch.load(weights_path, map_location='cpu')
    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 2

    exp_cfg['loop_file'] = 'loop_GT_4m'
    exp_cfg['head'] = 'UOTHead'
    exp_cfg['sinkhorn_type'] = 'unbalanced'

    current_date = datetime.now()

    exp_cfg['test_sequence'] = args.validation_sequence
    sequences_validation = [exp_cfg['test_sequence']]
    exp_cfg['sinkhorn_iter'] = 5

    if args.dataset == 'kitti':
        dataset_for_recall = KITTILoader3DPoses(args.root_folder, sequences_validation[0],
                                                os.path.join(args.root_folder, 'sequences', sequences_validation[0],'poses.txt'),
                                                train=False, loop_file=exp_cfg['loop_file'],
                                                remove_random_angle=args.remove_random_angle,
                                                without_ground=args.without_ground)
    else:
        dataset_for_recall = KITTI3603DPoses(args.root_folder, sequences_validation[0],
                                             train=False, loop_file='loop_GT_4m_noneg',
                                             remove_random_angle=args.remove_random_angle,
                                             without_ground=args.without_ground)

    test_pair_idxs = []
    index = faiss.IndexFlatL2(3)
    poses = np.stack(dataset_for_recall.poses).copy()
    index.add(poses[:50, :3, 3].copy())
    num_frames_with_loop = 0
    num_frames_with_reverse_loop = 0
    for i in tqdm(range(100, len(dataset_for_recall.poses))):
        current_pose = poses[i:i+1, :3, 3].copy()
        index.add(poses[i-50:i-49, :3, 3].copy())
        lims, D, I = index.range_search(current_pose, 4.**2)
        for j in range(lims[0], lims[1]):
            if j == 0:
                num_frames_with_loop += 1
                yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
                yaw_diff = yaw_diff % (2 * np.pi)
                if 0.79 <= yaw_diff <= 5.5:
                    num_frames_with_reverse_loop += 1

            test_pair_idxs.append([I[j], i])
    test_pair_idxs = np.array(test_pair_idxs)

    batch_sampler = BatchSamplePairs(dataset_for_recall, test_pair_idxs, exp_cfg['batch_size'])
    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               # batch_size=exp_cfg['batch_size'],
                                               num_workers=2,
                                               # sampler=sampler,
                                               batch_sampler=batch_sampler,
                                               # worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    model = get_model(exp_cfg)

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

    model.load_state_dict(renamed_dict, strict=True)

    model = model.to(device)

    rot_errors = []
    transl_errors = []
    yaw_error = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    # Testing
    if exp_cfg['weight_rot'] > 0:
        current_frame = 0
        yaw_preds = torch.zeros((len(dataset_for_recall.poses), len(dataset_for_recall.poses)))
        transl_errors = []
        for batch_idx, sample in enumerate(tqdm(RecallLoader)):

            model.eval()
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

                torch.cuda.synchronize()
                batch_dict = model(model_in, metric_head=True)
                torch.cuda.synchronize()
                pred_transl = []
                yaw = batch_dict['out_rotation']

                if not args.ransac:
                    transformation = batch_dict['transformation']
                    homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                    transformation = torch.cat((transformation, homogeneous), dim=1)
                    transformation = transformation.inverse()
                    for i in range(batch_dict['batch_size'] // 2):
                        yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(transformation[i])[-1].item()
                        pred_transl.append(transformation[i][:3, 3].detach().cpu())
                elif args.ransac:
                    coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
                    feats = batch_dict['point_features'].squeeze(-1)
                    for i in range(batch_dict['batch_size'] // 2):
                        coords1 = coords[i]
                        coords2 = coords[i + batch_dict['batch_size'] // 2]
                        feat1 = feats[i]
                        feat2 = feats[i + batch_dict['batch_size'] // 2]
                        pcd1 = o3d.geometry.PointCloud()
                        pcd1.points = o3d.utility.Vector3dVector(coords1[:, 1:].cpu().numpy())
                        pcd2 = o3d.geometry.PointCloud()
                        pcd2.points = o3d.utility.Vector3dVector(coords2[:, 1:].cpu().numpy())
                        pcd1_feat = reg_module.Feature()
                        pcd1_feat.data = feat1.permute(0, 1).cpu().numpy()
                        pcd2_feat = reg_module.Feature()
                        pcd2_feat.data = feat2.permute(0, 1).cpu().numpy()

                        torch.cuda.synchronize()
                        try:
                            result = reg_module.registration_ransac_based_on_feature_matching(
                                pcd2, pcd1, pcd2_feat, pcd1_feat, True,
                                0.6,
                                reg_module.TransformationEstimationPointToPoint(False),
                                3, [],
                                reg_module.RANSACConvergenceCriteria(5000))
                        except:
                            result = reg_module.registration_ransac_based_on_feature_matching(
                                pcd2, pcd1, pcd2_feat, pcd1_feat,
                                0.6,
                                reg_module.TransformationEstimationPointToPoint(False),
                                3, [],
                                reg_module.RANSACConvergenceCriteria(5000))

                        transformation = torch.tensor(result.transformation.copy())
                        if args.icp:
                            p1 = o3d.geometry.PointCloud()
                            p1.points = o3d.utility.Vector3dVector(sample['anchor'][i][:, :3].cpu().numpy())
                            p2 = o3d.geometry.PointCloud()
                            p2.points = o3d.utility.Vector3dVector(
                                sample['anchor'][i + batch_dict['batch_size'] // 2][:, :3].cpu().numpy())
                            result2 = reg_module.registration_icp(
                                        p2, p1, 0.1, result.transformation,
                                        reg_module.TransformationEstimationPointToPoint())
                            transformation = torch.tensor(result2.transformation.copy())
                        yaw_preds[test_pair_idxs[current_frame + i, 0], test_pair_idxs[current_frame + i, 1]] = \
                            mat2xyzrpy(transformation)[-1].item()
                        pred_transl.append(transformation[:3, 3].detach().cpu())

                for i in range(batch_dict['batch_size'] // 2):
                    pose1 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 0]]
                    pose2 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 1]]
                    delta_pose = np.linalg.inv(pose1) @ pose2
                    transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl[i]
                    transl_errors.append(transl_error.norm())

                    yaw_pred = yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]]
                    yaw_pred = yaw_pred % (2 * np.pi)
                    delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                    delta_yaw = delta_yaw % (2 * np.pi)
                    diff_yaw = abs(delta_yaw - yaw_pred)
                    diff_yaw = diff_yaw % (2 * np.pi)
                    diff_yaw = (diff_yaw * 180) / np.pi
                    if diff_yaw > 180.:
                        diff_yaw = 360 - diff_yaw
                    yaw_error.append(diff_yaw)

                current_frame += batch_dict['batch_size'] // 2


    print(weights_path)
    print(exp_cfg['test_sequence'])

    transl_errors = np.array(transl_errors)
    yaw_error = np.array(yaw_error)
    print("Mean rotation error: ", yaw_error.mean())
    print("Median rotation error: ", np.median(yaw_error))
    print("STD rotation error: ", yaw_error.std())
    print("Mean translation error: ", transl_errors.mean())
    print("Median translation error: ", np.median(transl_errors))
    print("STD translation error: ", transl_errors.std())
    # save_dict = {'rot': yaw_error, 'transl': transl_errors}
    # save_path = f'./results_for_paper/lcdnet00+08_{exp_cfg["test_sequence"]}'
    if '360' in weights_path:
        save_path = f'./results_for_paper/lcdnet++_{exp_cfg["test_sequence"]}'
    else:
        save_path = f'./results_for_paper/lcdnet00+08_{exp_cfg["test_sequence"]}'
    if args.remove_random_angle > 0:
        save_path = save_path + f'_remove{args.remove_random_angle}'
    if args.icp:
        save_path = save_path+'_icp'
    elif args.ransac:
        save_path = save_path+'_ransac'
    if args.teaser:
        save_path = save_path + '_teaser'
    # print("Saving to ", save_path)
    # with open(f'{save_path}.pickle', 'wb') as f:
    #     pickle.dump(save_dict, f)
    valid = yaw_error <= 5.
    valid = valid & (np.array(transl_errors) <= 2.)
    succ_rate = valid.sum() / valid.shape[0]
    rte_suc = transl_errors[valid].mean()
    rre_suc = yaw_error[valid].mean()
    print(f"Success Rate: {succ_rate}, RTE: {rte_suc}, RRE: {rre_suc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--ransac', action='store_true', default=False)
    parser.add_argument('--teaser', action='store_true', default=False)
    parser.add_argument('--icp', action='store_true', default=False)
    parser.add_argument('--remove_random_angle', type=int, default=-1)
    parser.add_argument('--validation_sequence', type=str, default='08')
    parser.add_argument('--without_ground', action='store_true', default=False,
                        help='Use preprocessed point clouds with ground plane removed')
    args = parser.parse_args()

    main_process(0, args.weights_path, args)
