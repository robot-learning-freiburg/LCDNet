import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import distances


class TripletLoss(nn.Module):
    def __init__(self, margin: float, triplet_selector, distance: distances.BaseDistance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.distance = distance

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        if other_embeddings is None:
            other_embeddings = embeddings
        dist_mat = self.distance(embeddings, other_embeddings)
        triplets = self.triplet_selector(dist_mat, pos_mask, neg_mask, self.distance.is_inverted)
        distance_positive = dist_mat[triplets[0], triplets[1]]
        if triplets[-1] is None:
            if self.distance.is_inverted:
                return F.relu(1 - distance_positive).mean()
            else:
                return F.relu(distance_positive).mean()
        distance_negative = dist_mat[triplets[0], triplets[2]]
        curr_margin = self.distance.margin(distance_positive, distance_negative)
        loss = F.relu(curr_margin + self.margin)
        return loss.mean()


def sinkhorn_matches_loss(batch_dict, delta_pose, mode='pairs'):
    sinkhorn_matches = batch_dict['sinkhorn_matches']
    src_coords = batch_dict['point_coords']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    B, N_POINT, _ = src_coords.shape
    if mode == 'pairs':
        B = B // 2
    else:
        B = B // 3
    src_coords = src_coords[:B, :, [1, 2, 3, 0]]
    src_coords[:, :, -1] = 1.
    gt_dst_coords = torch.bmm(delta_pose.inverse(), src_coords.permute(0, 2, 1))
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    return loss


def pose_loss(batch_dict, delta_pose, mode='pairs'):
    src_coords = batch_dict['point_coords']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    B, N_POINT, _ = src_coords.shape
    if mode == 'pairs':
        B = B // 2
    else:
        B = B // 3
    src_coords = src_coords[:B, :, [1, 2, 3, 0]]
    src_coords[:, :, -1] = 1.
    delta_pose_inv = delta_pose.double().inverse()
    gt_dst_coords = torch.bmm(delta_pose_inv, src_coords.permute(0,2,1).double()).float()
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    # loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    transformation = batch_dict['transformation']
    pred_dst_coords = torch.bmm(transformation, src_coords.permute(0,2,1))
    pred_dst_coords = pred_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = torch.mean(torch.abs(pred_dst_coords - gt_dst_coords))
    # loss = (pred_dst_coords - gt_dst_coords).norm(dim=2).mean()
    return loss
