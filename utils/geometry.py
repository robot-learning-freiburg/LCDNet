import numpy as np
import torch

from functools import reduce
import torch.utils.data
import math
import open3d as o3d


def euler2mat(z, y, x):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2mat_torch(z, y, x):
    Ms = []
    if z:
        cosz = torch.cos(z)
        sinz = torch.sin(z)
        Ms.append(torch.tensor(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]).cuda())
    if y:
        cosy = torch.cos(y)
        siny = torch.sin(y)
        Ms.append(torch.tensor(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]).cuda())
    if x:
        cosx = torch.cos(x)
        sinx = torch.sin(x)
        Ms.append(torch.tensor(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]).cuda())
    if Ms:
        return reduce(torch.matmul, Ms[::-1])
    return torch.eye(3).float().cuda()


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix
    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin(rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)


def rototransl_pc(pc, transl, rot, name):

    pcd = o3d.geometry.PointCloud()

    R_sample = torch.from_numpy(euler2mat(rot[0], rot[1], rot[2])).cuda()

    RT1 = torch.cat((torch.eye(3).double().cuda(), transl.unsqueeze(0).T), 1)
    RT1 = torch.cat((RT1, torch.tensor([[0., 0., 0., 1.]]).cuda()), 0)
    RT2 = torch.cat((R_sample, torch.zeros((3, 1)).double().cuda()), 1)
    RT2 = torch.cat((RT2, torch.tensor([[0., 0., 0., 1.]]).cuda()), 0)

    RT_sample = RT1 @ RT2

    ones = torch.ones((1, pc.shape[0])).double().cuda()
    new_pc = RT_sample @ torch.cat((pc.T, ones), 0)
    new_pc = new_pc[0:3, :].T

    pcd.points = o3d.utility.Vector3dVector(new_pc.cpu().numpy())
    o3d.io.write_point_cloud(name, pcd)

    return RT_sample


def get_rt_matrix(transl, rot, rot_parmas='xyz'):
    if rot_parmas == 'xyz':
        # R_sample = torch.from_numpy(euler2mat(rot[2], rot[1], rot[0])).cuda()
        R_sample = euler2mat_torch(rot[2], rot[1], rot[0])
    elif rot_parmas == 'zyx':
        # R_sample = torch.from_numpy(euler2mat(rot[0], rot[1], rot[2])).cuda()
        R_sample = euler2mat_torch(rot[0], rot[1], rot[2])
    else:
        raise TypeError("Unknown rotation params order")

    RT1 = torch.cat((torch.eye(3).float().cuda(), transl.unsqueeze(0).T), 1)
    RT1 = torch.cat((RT1, torch.tensor([[0., 0., 0., 1.]]).cuda()), 0)
    RT2 = torch.cat((R_sample, torch.zeros((3, 1)).float().cuda()), 1)
    RT2 = torch.cat((RT2, torch.tensor([[0., 0., 0., 1.]]).cuda()), 0)

    RT_sample = RT1.float() @ RT2.float()

    return RT_sample
