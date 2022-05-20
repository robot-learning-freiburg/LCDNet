import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone3D.heads import PointNetHead, UOTHead


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, h_n1=1024, h_n2=512):
        super().__init__()
        self.FC1 = nn.Linear(input_dim, h_n1)
        self.FC2 = nn.Linear(h_n1, h_n2)
        self.FC3 = nn.Linear(h_n2, output_dim)
        self.dropout = nn.Dropout(0.2, True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.FC1(x))
        x = self.dropout(x)
        x = F.relu(self.FC2(x))
        x = self.dropout(x)
        x = self.FC3(x)
        return x


class NetVlad(nn.Module):
    def __init__(self, backbone, NV, feature_norm=False):
        super().__init__()
        self.backbone = backbone
        self.NV = NV
        self.feature_norm = feature_norm

    def forward(self, x):
        x = self.backbone(x)
        if self.feature_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.NV(x)
        return x


class NetVladCustom(nn.Module):
    def __init__(self, backbone, NV, feature_norm=False, fc_input_dim=256,
                 points_num=4096, head='UOTHead', rotation_parameters=2,
                 sinkhorn_iter=5, use_svd=True, sinkhorn_type='unbalanced'):
        super().__init__()
        self.backbone = backbone
        self.NV = NV
        self.feature_norm = feature_norm
        self.head = head

        # PointNetHead
        if head == 'PointNet':
            self.pose_head = PointNetHead(fc_input_dim, points_num, rotation_parameters)
            self.mp1 = torch.nn.MaxPool2d((points_num, 1), 1)
        # UOTHead
        elif head == 'UOTHead':
            self.pose_head = UOTHead(sinkhorn_iter, use_svd, sinkhorn_type)

    def forward(self, batch_dict, metric_head=True, compute_embeddings=True, compute_transl=True,
                compute_rotation=True, compute_backbone=True, mode='pairs'):
        if compute_backbone:
            batch_dict = self.backbone(batch_dict, compute_embeddings, compute_rotation)

        if self.feature_norm:
            if compute_rotation:
                batch_dict['point_features'] = F.normalize(batch_dict['point_features'], p=2, dim=1)
            if compute_embeddings:
                batch_dict['point_features_NV'] = F.normalize(batch_dict['point_features_NV'], p=2, dim=1)

        if self.head == 'PointNet++':
            batch_dict['point_features'] = batch_dict['point_features'].permute(0, 2, 1, 3)
            batch_dict['point_features_NV'] = batch_dict['point_features_NV'].permute(0, 2, 1, 3)

        if compute_embeddings:
            embedding = self.NV(batch_dict['point_features_NV'])

        else:
            embedding = None
        batch_dict['out_embedding'] = embedding

        if metric_head:
            # PointNetHead
            if self.head == 'PointNet':
                B, C, NUM, _ = batch_dict['point_features'].shape
                if mode == 'pairs':
                    assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
                    B = B // 2
                    anchors_feature_maps = batch_dict['point_features'][:B, :, :]
                    positives_feature_maps = batch_dict['point_features'][B:, :, :]
                else:
                    assert B % 3 == 0, "Batch size must be multiple of 3: B anchor + B positive + B negative samples"
                    B = B // 3
                    anchors_feature_maps = batch_dict['point_features'][:B, :, :]
                    positives_feature_maps = batch_dict['point_features'][B:2*B, :, :]
                anchors_feature_maps = self.mp1(anchors_feature_maps)
                positives_feature_maps = self.mp1(positives_feature_maps)
                pose_head_in = torch.cat((anchors_feature_maps, positives_feature_maps), 1)
                transl, yaw = self.pose_head(pose_head_in, compute_transl, compute_rotation)
                batch_dict['out_rotation'] = yaw
                batch_dict['out_translation'] = transl
            # UOTHead
            elif self.head == 'UOTHead':
                batch_dict = self.pose_head(batch_dict, compute_transl, compute_rotation, mode=mode)

        else:
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None

        return batch_dict
