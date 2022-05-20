import argparse
import os
import time
from functools import partial
from shutil import copy2

import yaml
import wandb
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random
from torch.nn.parallel import DistributedDataParallel

from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DPoses
from datasets.KITTIDataset import KITTILoader3DPoses, KITTILoader3DDictPairs
from loss import TripletLoss, sinkhorn_matches_loss, pose_loss

from models.get_models import get_model
from triple_selector import hardest_negative_selector, random_negative_selector, \
    semihard_negative_selector
from utils.data import datasets_concat_kitti, merge_inputs, datasets_concat_kitti360
from evaluate_kitti import evaluate_model_with_emb
from datetime import datetime

from utils.geometry import get_rt_matrix, mat2xyzrpy
from utils.tools import _pairwise_distance
from pytorch_metric_learning import distances

import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, optimizer, sample, loss_fn, exp_cfg, device, mode='pairs'):
    if True:
        model.train()

        optimizer.zero_grad()

        if 'sequence' in sample:
            neg_mask = sample['sequence'].view(1,-1) != sample['sequence'].view(-1, 1)
        else:
            neg_mask = torch.zeros((sample['anchor_pose'].shape[0], sample['anchor_pose'].shape[0]),
                                   dtype=torch.bool)

        pair_dist = _pairwise_distance(sample['anchor_pose'])
        neg_mask = ((pair_dist > exp_cfg['negative_distance']) | neg_mask)
        neg_mask = neg_mask.repeat(2, 2).to(device)

        anchor_transl = sample['anchor_pose'].to(device)
        positive_transl = sample['positive_pose'].to(device)
        anchor_rot = sample['anchor_rot'].to(device)
        positive_rot = sample['positive_rot'].to(device)

        anchor_list = []
        positive_list = []

        delta_pose = []
        for i in range(anchor_transl.shape[0]):
            anchor = sample['anchor'][i].to(device)
            positive = sample['positive'][i].to(device)

            anchor_i = anchor
            positive_i = positive
            anchor_transl_i = anchor_transl[i]
            anchor_rot_i = anchor_rot[i]
            positive_transl_i = positive_transl[i]
            positive_rot_i = positive_rot[i]

            anchor_i_reflectance = anchor_i[:, 3].clone()
            positive_i_reflectance = positive_i[:, 3].clone()
            anchor_i[:, 3] = 1.
            positive_i[:, 3] = 1.

            rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
            rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')

            if exp_cfg['point_cloud_augmentation']:

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (np.pi / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3. - 1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_anch_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                anchor_i = rt_anch_augm.inverse() @ anchor_i.T
                anchor_i = anchor_i.T
                anchor_i[:, 3] = anchor_i_reflectance.clone()

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (3.141592 / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3.-1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_pos_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                positive_i = rt_pos_augm.inverse() @ positive_i.T
                positive_i = positive_i.T
                positive_i[:, 3] = positive_i_reflectance.clone()

                rt_anch_concat = rt_anchor @ rt_anch_augm
                rt_pos_concat = rt_positive @ rt_pos_augm

                rt_anchor2positive = rt_anch_concat.inverse() @ rt_pos_concat
                ext = mat2xyzrpy(rt_anchor2positive)

            else:
                raise NotImplementedError()

            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
            positive_list.append(model.module.backbone.prepare_input(positive_i))
            del anchor_i, positive_i
            delta_pose.append(rt_anchor2positive.unsqueeze(0))

        delta_pose = torch.cat(delta_pose)

        model_in = KittiDataset.collate_batch(anchor_list + positive_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)

        metric_head = True
        compute_embeddings = True
        compute_transl = True
        compute_rotation = True
        batch_dict = model(model_in, metric_head, compute_embeddings,
                           compute_transl, compute_rotation, mode=mode)

        model_out = batch_dict['out_embedding']

        # Translation loss
        total_loss = 0.

        loss_transl = torch.tensor([0.], device=device)

        if exp_cfg['weight_rot'] > 0.:
            if exp_cfg['sinkhorn_aux_loss']:
                aux_loss = sinkhorn_matches_loss(batch_dict, delta_pose, mode=mode)
            else:
                aux_loss = torch.tensor([0.], device=device)
            loss_rot = pose_loss(batch_dict, delta_pose, mode=mode)
            if exp_cfg['sinkhorn_type'] == 'rpm':
                inlier_loss = (1 - batch_dict['transport'].sum(dim=1)).mean()
                inlier_loss += (1 - batch_dict['transport'].sum(dim=2)).mean()
                loss_rot += 0.01 * inlier_loss

            total_loss = total_loss + exp_cfg['weight_rot']*(loss_rot + 0.05*aux_loss)
        else:
            loss_rot = torch.tensor([0.], device=device)

        if exp_cfg['weight_metric_learning'] > 0.:
            if exp_cfg['norm_embeddings']:
                model_out = model_out / model_out.norm(dim=1, keepdim=True)

            pos_mask = torch.zeros((model_out.shape[0], model_out.shape[0]), device=device)

            batch_size = (model_out.shape[0]//2)
            for i in range(batch_size):
                pos_mask[i, i+batch_size] = 1
                pos_mask[i+batch_size, i] = 1

            loss_metric_learning = loss_fn(model_out, pos_mask, neg_mask) * exp_cfg['weight_metric_learning']
            total_loss = total_loss + loss_metric_learning

        total_loss.backward()
        optimizer.step()

        return total_loss, loss_rot, loss_transl


def test(model, sample, exp_cfg, device):
    model.eval()

    with torch.no_grad():
        anchor_transl = sample['anchor_pose'].to(device)
        positive_transl = sample['positive_pose'].to(device)
        anchor_rot = sample['anchor_rot'].to(device)
        positive_rot = sample['positive_rot'].to(device)

        anchor_list = []
        positive_list = []
        delta_transl_list = []
        delta_rot_list = []
        delta_pose_list = []
        for i in range(anchor_transl.shape[0]):
            anchor = sample['anchor'][i].to(device)
            positive = sample['positive'][i].to(device)

            anchor_i = anchor
            positive_i = positive

            anchor_transl_i = anchor_transl[i]
            anchor_rot_i = anchor_rot[i]
            positive_transl_i = positive_transl[i]
            positive_rot_i = positive_rot[i]

            rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
            rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')
            rt_anchor2positive = rt_anchor.inverse() @ rt_positive
            ext = mat2xyzrpy(rt_anchor2positive)
            delta_transl_i = ext[0:3]
            delta_rot_i = ext[3:]
            delta_transl_list.append(delta_transl_i.unsqueeze(0))
            delta_rot_list.append(delta_rot_i.unsqueeze(0))
            delta_pose_list.append(rt_anchor2positive.unsqueeze(0))

            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
            positive_list.append(model.module.backbone.prepare_input(positive_i))
            del anchor_i, positive_i

        delta_rot = torch.cat(delta_rot_list)
        delta_pose_list = torch.cat(delta_pose_list)

        model_in = KittiDataset.collate_batch(anchor_list + positive_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)

        batch_dict = model(model_in, metric_head=True)
        anchor_out = batch_dict['out_embedding']

        diff_yaws = delta_rot[:, 2] % (2*np.pi)

        transformation = batch_dict['transformation']
        homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
        transformation = torch.cat((transformation, homogeneous), dim=1)
        transformation = transformation.inverse()
        final_yaws = torch.zeros(transformation.shape[0], device=transformation.device,
                                 dtype=transformation.dtype)
        for i in range(transformation.shape[0]):
            final_yaws[i] = mat2xyzrpy(transformation[i])[-1]
        yaw = final_yaws
        transl_comps_error = (transformation[:,:3,3] - delta_pose_list[:,:3,3]).norm(dim=1).mean()

        yaw = yaw % (2*np.pi)
        yaw_error_deg = torch.abs(diff_yaws - yaw)
        yaw_error_deg[yaw_error_deg>np.pi] = 2*np.pi - yaw_error_deg[yaw_error_deg>np.pi]
        yaw_error_deg = yaw_error_deg.mean() * 180 / np.pi

    if exp_cfg['norm_embeddings']:
        anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
    anchor_out = anchor_out[:anchor_transl.shape[0]]
    return anchor_out, transl_comps_error, yaw_error_deg


def get_database_embs(model, sample, exp_cfg, device):
    model.eval()

    with torch.no_grad():
        anchor_list = []
        for i in range(len(sample['anchor'])):
            anchor = sample['anchor'][i].to(device)

            anchor_i = anchor
            anchor_list.append(model.module.backbone.prepare_input(anchor_i))
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


def main_process(gpu, exp_cfg, common_seed, world_size, args):
    global EPOCH
    rank = gpu

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    local_seed = (common_seed + common_seed ** gpu) ** 2
    local_seed = local_seed % (2**32 - 1)
    np.random.seed(common_seed)
    torch.random.manual_seed(common_seed)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    print(f"Process {rank}, seed {common_seed}")

    current_date = datetime.now()
    dt_string = current_date.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_folder = current_date.strftime("%d-%m-%Y_%H-%M-%S")

    exp_cfg['effective_batch_size'] = args.batch_size * world_size
    if args.wandb and rank == 0:
        wandb.init(project="deep_lcd", name=dt_string, config=exp_cfg)

    if args.dataset == 'kitti':
        sequences_training = ["05", "06", "07", "09"]
        sequences_validation = ["00", "08"]
    elif args.dataset == 'kitti360':
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0004_sync",
                              "2013_05_28_drive_0005_sync", "2013_05_28_drive_0006_sync"]
        sequences_validation = ["2013_05_28_drive_0002_sync", "2013_05_28_drive_0009_sync"]
    else:
        raise TypeError("Dataset should be either 'kitti' or 'kitti360'")

    data_transform = None
    # sequences_training = ["09"]
    # sequences_validation = ["09"]

    if args.dataset == 'kitti':
        training_dataset, dataset_list_train = datasets_concat_kitti(args.root_folder,
                                                                     sequences_training,
                                                                     data_transform,
                                                                     "3D",
                                                                     loop_file=exp_cfg['loop_file'],
                                                                     jitter=exp_cfg['point_cloud_jitter'],
                                                                     without_ground=args.without_ground)
        validation_dataset = KITTILoader3DDictPairs(args.root_folder, sequences_validation[0],
                                                    os.path.join(args.root_folder, 'sequences', sequences_validation[0], 'poses.txt'),
                                                    loop_file=exp_cfg['loop_file'], without_ground=args.without_ground)
        dataset_for_recall = KITTILoader3DPoses(args.root_folder, sequences_validation[0],
                                                os.path.join(args.root_folder, 'sequences', sequences_validation[0], 'poses.txt'),
                                                train=False, loop_file=exp_cfg['loop_file'], without_ground=args.without_ground)
    elif args.dataset == 'kitti360':
        training_dataset, dataset_list_train = datasets_concat_kitti360(args.root_folder,
                                                                        sequences_training,
                                                                        data_transform,
                                                                        "3D",
                                                                        loop_file=exp_cfg['loop_file'],
                                                                        jitter=exp_cfg['point_cloud_jitter'],
                                                                        without_ground=args.without_ground)
        validation_dataset = KITTI3603DDictPairs(args.root_folder, sequences_validation[0],
                                                 loop_file=exp_cfg['loop_file'], without_ground=args.without_ground)
        dataset_for_recall = KITTI3603DPoses(args.root_folder, sequences_validation[0],
                                             train=False, loop_file=exp_cfg['loop_file'], without_ground=args.without_ground)

    dataset_list_valid = [dataset_for_recall]

    train_indices = list(range(len(training_dataset)))
    np.random.shuffle(train_indices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed,
    )
    recall_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_for_recall,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed,
        shuffle=False
    )

    if exp_cfg['loss_type'].startswith('triplet'):
        neg_selector = random_negative_selector
        if 'hardest' in exp_cfg['loss_type']:
            neg_selector = hardest_negative_selector
        if 'semihard' in exp_cfg['loss_type']:
            neg_selector = semihard_negative_selector
        loss_fn = TripletLoss(exp_cfg['margin'], neg_selector, distances.LpDistance())
    else:
        raise NotImplementedError(f"Loss {exp_cfg['loss_type']} not implemented")

    positive_distance = 4.
    negative_distance = 10.

    if rank == 0:
        print("Positive distance: ", positive_distance)
    exp_cfg['negative_distance'] = negative_distance

    final_dest = ''
    if rank == 0:
        if not os.path.exists(args.checkpoints_dest):
            try:
                os.mkdir(args.checkpoints_dest)
            except:
                raise TypeError('Folder for saving checkpoints does not exist!')
        final_dest = args.checkpoints_dest + '/' + dt_string_folder
        os.mkdir(final_dest)
        if args.wandb:
            wandb.save(f'{final_dest}/best_model_so_far.tar')
            copy2('wandb_config.yaml', f'{final_dest}/wandb_config.yaml')
            wandb.save(f'{final_dest}/wandb_config.yaml')

    model = get_model(exp_cfg)
    if args.weights is not None:
        print('Loading pre-trained params...')
        saved_params = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(saved_params['state_dict'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank)

    start_full_time = time.time()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=exp_cfg['learning_rate'], betas=(exp_cfg['beta1'], exp_cfg['beta2']),
                           eps=exp_cfg['eps'], weight_decay=exp_cfg['weight_decay'], amsgrad=False)

    starting_epoch = 1
    scheduler_epoch = -1
    if args.resume:
        optimizer.load_state_dict(saved_params['optimizer'])
        starting_epoch = saved_params['epoch']
        scheduler_epoch = saved_params['epoch']

    if exp_cfg['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5,
                                                         last_epoch=scheduler_epoch)
    else:
        raise RuntimeError("Unknown Scheduler")

    best_rot_error = 1000
    max_recall = 0.
    max_auc = 0.
    old_saved_file = None
    old_saved_file_recall = None
    old_saved_file_auc = None

    np.random.seed(local_seed)
    torch.random.manual_seed(local_seed)

    for epoch in range(starting_epoch, exp_cfg['epochs'] + 1):
        dist.barrier()

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        recall_sampler.set_epoch(epoch)
        EPOCH = epoch

        init_fn = partial(_init_fn, epoch=epoch, seed=local_seed)
        TrainLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                  sampler=train_sampler,
                                                  batch_size=args.batch_size,
                                                  num_workers=2,
                                                  worker_init_fn=init_fn,
                                                  collate_fn=merge_inputs,
                                                  pin_memory=True,
                                                  drop_last=True)

        TestLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                 sampler=val_sampler,
                                                 batch_size=args.batch_size,
                                                 num_workers=2,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 pin_memory=True)

        RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                                   sampler=recall_sampler,
                                                   batch_size=args.batch_size,
                                                   num_workers=2,
                                                   worker_init_fn=init_fn,
                                                   collate_fn=merge_inputs,
                                                   pin_memory=True)

        if epoch > starting_epoch:
            if exp_cfg['scheduler'] == 'multistep':
                scheduler.step()
            if args.wandb and rank == 0:
                wandb.log({"LR": optimizer.param_groups[0]['lr']}, commit=False)
        if rank == 0:
            print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        total_rot_loss = 0.
        total_transl_loss = 0.
        local_loss = 0.
        local_iter = 0
        total_iter = 0

        ## Training ##
        for batch_idx, sample in enumerate(TrainLoader):
            start_time = time.time()
            loss, loss_rot, loss_transl = train(model, optimizer, sample, loss_fn, exp_cfg,
                                                device, mode='pairs')

            if exp_cfg['scheduler'] == 'onecycle':
                scheduler.step()

            dist.barrier()
            dist.reduce(loss, 0)
            dist.reduce(loss_rot, 0)
            dist.reduce(loss_transl, 0)
            if rank == 0:
                loss = (loss / world_size).item()
                loss_rot = (loss_rot / world_size).item()
                loss_transl = (loss_transl / world_size).item()
                local_loss += loss
                local_iter += 1

                if batch_idx % 20 == 0 and batch_idx != 0:
                    print('Iter %d / %d training loss = %.3f , time = %.2f' % (batch_idx,
                                                                               len(TrainLoader),
                                                                               local_loss / local_iter,
                                                                               time.time() - start_time))
                    local_loss = 0.
                    local_iter = 0.

                total_train_loss += loss * sample['anchor_pose'].shape[0]
                total_rot_loss += loss_rot * sample['anchor_pose'].shape[0]
                total_transl_loss += loss_transl * sample['anchor_pose'].shape[0]

                total_iter += sample['anchor_pose'].shape[0]

        if rank == 0:
            print("------------------------------------")
            print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_sampler)))
            print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
            print("------------------------------------")

        local_iter = 0.
        transl_error_sum = 0
        yaw_error_sum = 0
        emb_list = []

        # Testing
        if exp_cfg['weight_rot'] > 0:
            for batch_idx, sample in enumerate(TestLoader):
                start_time = time.time()
                _, transl_error, yaw_error = test(model, sample, exp_cfg, device)
                dist.barrier()
                dist.reduce(transl_error, 0)
                dist.reduce(yaw_error, 0)
                if rank == 0:
                    transl_error = (transl_error / world_size).item()
                    yaw_error = (yaw_error / world_size).item()
                    transl_error_sum += transl_error
                    yaw_error_sum += yaw_error
                    local_iter += 1

                    if batch_idx % 20 == 0 and batch_idx != 0:
                        print('Iter %d time = %.2f' % (batch_idx,
                                                       time.time() - start_time))
                        local_iter = 0.

        if exp_cfg['weight_metric_learning'] > 0.:
            for batch_idx, sample in enumerate(RecallLoader):
                emb = get_database_embs(model, sample, exp_cfg, device)
                dist.barrier()
                out_emb = [torch.zeros_like(emb) for _ in range(world_size)]
                dist.all_gather(out_emb, emb)

                if rank == 0:
                    interleaved_out = torch.empty((emb.shape[0]*world_size, emb.shape[1]),
                                                  device=emb.device, dtype=emb.dtype)
                    for current_rank in range(world_size):
                        interleaved_out[current_rank::world_size] = out_emb[current_rank]
                    emb_list.append(interleaved_out.detach().clone())

        if rank == 0:
            if exp_cfg['weight_metric_learning'] > 0.:
                emb_list = torch.cat(emb_list)
                emb_list = emb_list[:len(dataset_for_recall)]
                recall, maxF1, auc, auc2 = evaluate_model_with_emb(emb_list, dataset_list_valid, positive_distance)
            final_transl_error = transl_error_sum / len(TestLoader)
            final_yaw_error = yaw_error_sum / len(TestLoader)

            if args.wandb:
                if exp_cfg['weight_rot'] > 0.:
                    wandb.log({"Rotation Loss": (total_rot_loss / len(train_sampler)),
                               "Rotation Mean Error": final_yaw_error}, commit=False)
                wandb.log({"Translation Loss": (total_transl_loss / len(train_sampler)),
                           "Translation Error": final_transl_error}, commit=False)
                if exp_cfg['weight_metric_learning'] > 0.:
                    wandb.log({"Validation Recall @ 1": recall[0],
                               "Validation Recall @ 5": recall[4],
                               "Validation Recall @ 10": recall[9],
                               "Max F1": maxF1,
                               "AUC": auc2}, commit=False)
                wandb.log({"Training Loss": (total_train_loss / len(train_sampler))})

            print("------------------------------------")
            if exp_cfg['weight_metric_learning'] > 0.:
                print("Recall@k:")
                print(recall)
                print("Max F1: ", maxF1)
                print("AUC: ", auc2)
            print("Translation Error: ", final_transl_error)
            print("Rotation Error: ", final_yaw_error)
            print("------------------------------------")

            if final_yaw_error < best_rot_error:
                best_rot_error = final_yaw_error
                
                savefilename = f'{final_dest}/checkpoint_{epoch}_rot_{final_yaw_error:.3f}.tar'
                best_model = {
                    'config': exp_cfg,
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "Rotation Mean Error": final_yaw_error
                }
                torch.save(best_model, savefilename)
                if old_saved_file is not None:
                    os.remove(old_saved_file)
                if args.wandb:
                    wandb.run.summary["best_rot_error"] = final_yaw_error
                    temp = f'{final_dest}/best_model_so_far_rot.tar'
                    torch.save(best_model, temp)
                    wandb.save(temp)
                old_saved_file = savefilename

            if exp_cfg['weight_metric_learning'] > 0.:
                if auc2 > max_auc:
                    max_auc = auc2
                    savefilename_auc = f'{final_dest}/checkpoint_{epoch}_auc_{max_auc:.3f}.tar'
                    best_model_auc = {
                        'epoch': epoch,
                        'config': exp_cfg,
                        'state_dict': model.module.state_dict(),
                        'recall@1': recall[0],
                        'max_F1': maxF1,
                        'AUC': auc2,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(best_model_auc, savefilename_auc)
                    if old_saved_file_auc is not None:
                        os.remove(old_saved_file_auc)
                    old_saved_file_auc = savefilename_auc
                    if args.wandb:
                        wandb.run.summary["best_auc"] = max_auc

                        temp = f'{final_dest}/best_model_so_far_auc.tar'
                        torch.save(best_model_auc, temp)
                        wandb.save(temp)

            savefilename = f'{final_dest}/checkpoint_last_iter.tar'
            best_model = {
                'config': exp_cfg,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(best_model, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./KITTI',
                        help='dataset directory')
    parser.add_argument('--dataset', default='kitti',
                        help='dataset')
    parser.add_argument('--without_ground', action='store_true', default=False,
                        help='Use preprocessed point clouds with ground plane removed')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size (per gpu). Minimum 2.')
    parser.add_argument('--checkpoints_dest', default='./checkpoints',
                        help='folder where to save checkpoints')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Activate wandb service')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Only use default value: -1')
    parser.add_argument('--gpu_count', type=int, default=-1,
                        help='Only use default value: -1')
    parser.add_argument('--port', type=str, default='8888',
                        help='port to be used for DDP multi-gpu training')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights to be loaded, use together with --resume'
                             'to resume a previously stopped training')
    parser.add_argument('--resume', action='store_true',
                        help='Add this flag to resume a previously stopped training,'
                             'the --weights argument must be provided.')

    args = parser.parse_args()

    if args.batch_size < 2:
        raise argparse.ArgumentTypeError("The batch size should be at least 2")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    with open("wandb_config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    if args.gpu_count == -1:
        args.gpu_count = torch.cuda.device_count()
    if args.gpu == -1:
        mp.spawn(main_process, nprocs=args.gpu_count, args=(cfg['experiment'], 42, args.gpu_count, args,))
    else:
        main_process(args.gpu, cfg['experiment'], 42, args.gpu_count, args)
