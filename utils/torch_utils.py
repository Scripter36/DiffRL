# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import timeit
import math
import numpy as np
import gc
import torch
import cProfile

log_output = ""

def log(s):
    print(s)
    global log_output
    log_output = log_output + s + "\n"

# short hands


# torch quat/vector utils

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

@torch.jit.script
def quat_to_matrix(q):
    """
    q: (num_envs, 4)
    return: (num_envs, 3, 3)
    """
    # 쿼터니언 성분 추출
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # 행렬 요소 계산
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, yz, xz = x * y, y * z, x * z
    
    # 3x3 회전 행렬 생성
    rot = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)
    ], dim=-1).view(-1, 3, 3)
    
    return rot

# assuming q is unit quaternion
@torch.jit.script
def quat_log(q):
    return q[:, :3] * torch.arccos(torch.clamp(q[:, 3:4], min=-1.0, max=1.0))

@torch.jit.script
def quat_diff(q1, q2):
    return quat_log(quat_mul(quat_conjugate(q2), q1))

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)

@torch.jit.script
def tf_inverse_xz(transform):
    """
    Invert the transform, but only consider the xz transform and yaw rotation.
    transform: (num_envs, 7)
    return: (num_envs, 7)
    """
    num_envs = transform.shape[0]
    pos = transform[:, 0:3].clone()
    pos[:, 1] = 0
    rot = transform[:, 3:7].clone()
    forward = quat_apply(rot, torch.tensor([1.0, 0.0, 0.0], device=transform.device).unsqueeze(0).repeat(num_envs, 1))
    rot = quat_from_angle_axis(torch.atan2(-forward[:, 2], forward[:, 0]), torch.tensor([0.0, 1.0, 0.0], device=transform.device).unsqueeze(0).repeat(num_envs, 1))
    inv_pos = -pos
    inv_rot = quat_conjugate(rot)
    return torch.cat([inv_pos, inv_rot], dim=-1)

@torch.jit.script
def to_local_frame_spatial(spatial_transforms, local_transform):
    """
    spatial_transforms: (num_envs, transform_count, 7)
    local_transform: (num_envs, 7)
    """
    transform_count = spatial_transforms.shape[1]
    inv_local_transform = tf_inverse_xz(local_transform)
    inv_pos = inv_local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
    inv_rot = inv_local_transform[:, 3:7].unsqueeze(1).repeat(1, transform_count, 1)
    spatial_transforms[:, :, 0:3] = quat_apply(inv_rot, spatial_transforms[:, :, 0:3] + inv_pos)
    spatial_transforms[:, :, 3:7] = quat_mul(inv_rot, spatial_transforms[:, :, 3:7])
    return spatial_transforms

@torch.jit.script
def to_local_frame_pos(positions, local_transform):
    """
    positions: (num_envs, transform_count, 3)
    local_transform: (num_envs, 7)
    """
    transform_count = positions.shape[1]
    inv_local_transform = tf_inverse_xz(local_transform)
    inv_pos = inv_local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
    inv_rot = inv_local_transform[:, 3:7].unsqueeze(1).repeat(1, transform_count, 1)
    return quat_apply(inv_rot, positions + inv_pos)

@torch.jit.script
def to_local_frame_w(w, local_transform):
    """
    w: (num_envs, transform_count, 3)
    local_transform: (num_envs, 7)
    """
    transform_count = w.shape[1]
    inv_local_transform = tf_inverse_xz(local_transform)
    inv_pos = inv_local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
    inv_rot = inv_local_transform[:, 3:7].unsqueeze(1).repeat(1, transform_count, 1)
    return quat_apply(inv_rot, w)

@torch.jit.script
def get_center_of_mass(body_I_m, body_X_sm):
    """
    body_I_m: (num_envs, num_links, 6, 6)
    body_X_sm: (num_envs, num_links, 7)
    return: the center of mass position in the local frame (num_envs, 3)
    """
    num_envs = body_I_m.shape[0]
    mass = body_I_m[:, :, 3, 3] # (num_envs, num_links)
    com_pos = torch.sum(mass.unsqueeze(-1) * body_X_sm[:, :, 0:3], dim=1) / torch.sum(mass, dim=1).unsqueeze(-1)
    return com_pos


def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (
            #     element_type,
            #     size,
            #     mem) )
        print('Type: %s Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (mem_type, total_numel, total_mem) )

    gc.collect()

    LEN = 65
    objects = gc.get_objects()
    #print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)

def grad_norm(params):
    grad_norm = 0.
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad ** 2)
    return torch.sqrt(grad_norm)

def print_leaf_nodes(grad_fn, id_set):
    if grad_fn is None:
        return
    if hasattr(grad_fn, 'variable'):
        mem_id = id(grad_fn.variable)
        if not(mem_id in id_set):
            print('is leaf:', grad_fn.variable.is_leaf)
            print(grad_fn.variable)
            id_set.add(mem_id)

    # print(grad_fn)
    for i in range(len(grad_fn.next_functions)):
        print_leaf_nodes(grad_fn.next_functions[i][0], id_set)

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    return kl.mean()