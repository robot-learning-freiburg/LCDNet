import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetHead(nn.Module):

    def __init__(self, input_dim, points_num, rotation_parameters=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.FC1 = nn.Linear(input_dim * 2, 1024)
        self.FC2 = nn.Linear(1024, 512)
        self.FC_transl = nn.Linear(512, 3)
        self.FC_rot = nn.Linear(512, rotation_parameters)

    def forward(self, x, compute_transl=True, compute_rotation=True):
        x = x.view(x.shape[0], -1)

        x = self.relu(self.FC1(x))
        x = self.relu(self.FC2(x))
        if compute_transl:
            transl = self.FC_transl(x)
        else:
            transl = None

        if compute_rotation:
            yaw = self.FC_rot(x)
        else:
            yaw = None
        # print(transl, yaw)
        return transl, yaw


def compute_rigid_transform(points1, points2, weights):
    """Compute rigid transforms between two point clouds via weighted SVD.
       Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        points1 (torch.Tensor): (B, M, 3) coordinates of the first point cloud
        points2 (torch.Tensor): (B, N, 3) coordinates of the second point cloud
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from points1 to points2, i.e. T*points1 = points2
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_a = torch.sum(points1 * weights_normalized, dim=1)
    centroid_b = torch.sum(points2 * weights_normalized, dim=1)
    a_centered = points1 - centroid_a[:, None, :]
    b_centered = points2 - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


def sinkhorn_slack_variables(feature1, feature2, beta, alpha, n_iters = 5, slack = True):
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix using slack variables (dustbins)
        Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        feature1 (torch.Tensor): point-wise features of the first point cloud.
        feature2 (torch.Tensor): point-wise features of the second point cloud.
        beta (torch.Tensor): annealing parameter.
        alpha (torch.Tensor): matching rejection parameter.
        n_iters (int): Number of normalization iterations.
        slack (bool): Whether to include slack row and column.
    Returns:
        log(perm_matrix): (B, J, K) Doubly stochastic matrix.
    """

    B, N, _ = feature1.shape
    _, M, _ = feature2.shape

    # Feature normalization
    feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
    feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)

    dist = -2 * torch.matmul(feature1, feature2.permute(0, 2, 1))
    dist += torch.sum(feature1 ** 2, dim=-1)[:, :, None]
    dist += torch.sum(feature2 ** 2, dim=-1)[:, None, :]

    log_alpha = -beta * (dist - alpha)

    # Sinkhorn iterations
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

    return log_alpha


def sinkhorn_unbalanced(feature1, feature2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm for Unbalanced Optimal Transport.
    Modified from https://github.com/valeoai/FLOT/
    Args:
        feature1 (torch.Tensor):
            (B, N, C) Point-wise features for points cloud 1.
        feature2 (torch.Tensor):
            (B, M, C) Point-wise features for points cloud 2.
        epsilon (torch.Tensor):
            Entropic regularization.
        gamma (torch.Tensor):
            Mass regularization.
        max_iter (int):
            Number of iteration of the Sinkhorn algorithm.
    Returns:
        T (torch.Tensor):
            (B, N, M) Transport plan between point cloud 1 and 2.
    """

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) #* support

    # Early return if no iteration
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob1 = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob2 = (
            torch.ones(
                (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
            )
            / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T


class UOTHead(nn.Module):

    def __init__(self, nb_iter=5, use_svd=False, sinkhorn_type='unbalanced'):
        super().__init__()
        self.epsilon = torch.nn.Parameter(torch.zeros(1))  # Entropic regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))  # Mass regularisation
        self.nb_iter = nb_iter
        self.use_svd = use_svd
        self.sinkhorn_type = sinkhorn_type

        if not use_svd:
            raise NotImplementedError()

    def forward(self, batch_dict, compute_transl=True, compute_rotation=True, src_coords=None, mode='pairs'):

        feats = batch_dict['point_features'].squeeze(-1)
        B, C, NUM = feats.shape

        assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
        B = B // 2
        feat1 = feats[:B]
        feat2 = feats[B:]

        coords = batch_dict['point_coords'].view(2*B, -1, 4)
        coords1 = coords[:B, :, 1:]
        coords2 = coords[B:, :, 1:]

        if self.sinkhorn_type == 'unbalanced':
            transport = sinkhorn_unbalanced(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                epsilon=torch.exp(self.epsilon) + 0.03,
                gamma=torch.exp(self.gamma),
                max_iter=self.nb_iter,
            )
        else:
            transport = sinkhorn_slack_variables(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                F.softplus(self.epsilon),
                F.softplus(self.gamma),
                self.nb_iter,
            )
            transport = torch.exp(transport)

        row_sum = transport.sum(-1, keepdim=True)

        # Compute the "projected" coordinates of point cloud 1 in
        # point cloud 2 based on the optimal transport plan
        sinkhorn_matches = (transport @ coords2) / (row_sum + 1e-8)

        batch_dict['sinkhorn_matches'] = sinkhorn_matches
        batch_dict['transport'] = transport

        if not self.use_svd:
            raise NotImplementedError()
        else:
            if src_coords is None:
                src_coords = coords1
            transformation = compute_rigid_transform(src_coords, sinkhorn_matches, row_sum.squeeze(-1))
            batch_dict['transformation'] = transformation
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None

        return batch_dict
