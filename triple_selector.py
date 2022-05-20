import numpy as np
import torch


def get_all_triplets(dist_mat, pos_mask, neg_mask, is_inverted=False, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    triplets = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
    return torch.where(triplets)


def hardest_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    if neg_mask.sum() == 0:
        return a, p, None
    if is_inverted:
        dist_neg = dist_mat * neg_mask
        n = torch.max(dist_neg, dim=1)
    else:
        dist_neg = dist_mat.clone()
        dist_neg[~neg_mask] = dist_neg.max()+1.
        _, n = torch.min(dist_neg, dim=1)
    n = n[a]
    return a, p, n


def random_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    selected_negs = []
    for i in range(a.shape[0]):
        possible_negs = torch.where(neg_mask[a[i]])[0]
        if len(possible_negs) == 0:
            return a, p, None

        dist_neg = dist_mat[a[i], possible_negs]
        if is_inverted:
            curr_loss = -dist_mat[a[i], p[i]] + dist_neg + margin
        else:
            curr_loss = dist_mat[a[i], p[i]] - dist_neg + margin

        if len(possible_negs[curr_loss > 0]) > 0:
            possible_negs = possible_negs[curr_loss > 0]
        random_neg = np.random.choice(possible_negs.cpu().numpy())
        selected_negs.append(random_neg)
    n = torch.tensor(selected_negs, dtype=a.dtype, device=a.device)
    return a, p, n


def semihard_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    selected_negs = []
    for i in range(a.shape[0]):
        possible_negs = torch.where(neg_mask[a[i]])[0]
        if len(possible_negs) == 0:
            return a, p, None

        dist_neg = dist_mat[a[i], possible_negs]
        if is_inverted:
            curr_loss = -dist_mat[a[i], p[i]] + dist_neg + margin
        else:
            curr_loss = dist_mat[a[i], p[i]] - dist_neg + margin

        semihard_idxs = (curr_loss > 0) & (curr_loss < margin)
        if len(possible_negs[semihard_idxs]) > 0:
            possible_negs = possible_negs[semihard_idxs]
        random_neg = np.random.choice(possible_negs.cpu().numpy())
        selected_negs.append(random_neg)
    n = torch.tensor(selected_negs, dtype=a.dtype, device=a.device)
    return a, p, n
