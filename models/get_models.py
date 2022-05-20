from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg as pvrcnn_cfg

from models.backbone3D.PVRCNN import PVRCNN
from models.backbone3D.NetVlad import NetVLADLoupe
from models.backbone3D.models_3d import NetVladCustom


def get_model(exp_cfg, is_training=True):
    rotation_parameters = 1
    exp_cfg['use_svd'] = True

    if exp_cfg['3D_net'] == 'PVRCNN':
        cfg_from_yaml_file('./models/backbone3D/pv_rcnn.yaml', pvrcnn_cfg)
        pvrcnn_cfg.MODEL.PFE.NUM_KEYPOINTS = exp_cfg['num_points']
        if 'PC_RANGE' in exp_cfg:
            pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE = exp_cfg['PC_RANGE']
        pvrcnn = PVRCNN(pvrcnn_cfg, is_training, exp_cfg['model_norm'], exp_cfg['shared_embeddings'])
        net_vlad = NetVLADLoupe(feature_size=pvrcnn_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES,
                                cluster_size=exp_cfg['cluster_size'],
                                output_dim=exp_cfg['feature_output_dim_3D'],
                                gating=True, add_norm=True, is_training=is_training)
        model = NetVladCustom(pvrcnn, net_vlad, feature_norm=False, fc_input_dim=640,
                              points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                              rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                              use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
    else:
        raise TypeError("Unknown 3D network")
    return model
