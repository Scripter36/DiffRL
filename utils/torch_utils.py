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
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, yz, xz = x * y, y * z, x * z
    
    rot = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)
    ], dim=-1).view(-1, 3, 3)
    
    return rot

@torch.jit.script
def quat_to_ortho6d(q):
    """
    q: (..., 4)
    return: (..., 6)
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, yz, xz = x * y, y * z, x * z

    return torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz),
        2 * (xy + wz), 1 - 2 * (x2 + z2),
        2 * (xz - wy), 2 * (yz + wx),
    ], dim=-1).view(shape[:-1] + (6,))

@torch.jit.script
def normalize_angle(theta):
    """
    map [0, 2pi] to [-pi, pi]
    theta: (num_envs)
    return: (num_envs)
    """
    norm_theta = torch.fmod(theta, 2 * math.pi)
    norm_theta = torch.where(norm_theta > math.pi, -2 * math.pi + norm_theta, norm_theta)
    norm_theta = torch.where(norm_theta < -math.pi, 2 * math.pi + norm_theta, norm_theta)
    return norm_theta

@torch.jit.script
def quat_theta(q):
    """
    q: (..., 4)
    return: (..., 1)
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return normalize_angle(torch.arccos(torch.clamp(q[:, 3:4], min=-1.0 + 1e-5, max=1.0 - 1e-5))).view(shape[:-1]).unsqueeze(-1)

@torch.jit.script
def quat_diff(q1, q2):
    return quat_theta(quat_mul(q2, quat_conjugate(q1)))

# @torch.jit.script
# def quat_theta_modified(q):
#     """
#     q: (..., 4)
#     return: (..., 1)
#     """
#     shape = q.shape
#     q = q.reshape(-1, 4)
#     return (1 - torch.sum(q[:, :] ** 2, dim=-1)).view(shape[:-1]).unsqueeze(-1)

@torch.jit.script
def quat_diff_approx(q1, q2):
    """
    q1: (..., 4)
    q2: (..., 4)
    return: (..., 1)
    """
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    return (1 - torch.sum((q1 * q2) ** 2, dim=-1)).view(shape[:-1]).unsqueeze(-1)

@torch.jit.script
def quat_diff_chordal(q1, q2):
    """
    q1: (..., 4)
    q2: (..., 4)
    return: (..., 1)
    """
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # chordal distance: ||R1 - R2||_F
    R1 = quat_to_matrix(q1)
    R2 = quat_to_matrix(q2)
    return torch.sum(torch.sum((R1 - R2) ** 2, dim=-1), dim=-1).view(shape[:-1]).unsqueeze(-1) / 4.0

@torch.jit.script
def ortho6d_diff(o1, o2):
    """
    o1: (..., 6)
    o2: (..., 6)
    return: (..., 1)
    """
    shape = o1.shape
    o1 = o1.reshape(-1, 6)
    o2 = o2.reshape(-1, 6)
    return torch.sum((o1 - o2) ** 2, dim=-1).view(shape[:-1]).unsqueeze(-1) / 3.0

@torch.jit.script
def angular_velocity(q1, q2, dt: float):
    """
    q1: (num_envs, 4)
    q2: (num_envs, 4)
    return: (num_envs, 3)
    """
    # TODO: better quaternion flipping
    dq = (q2 - q1)
    dq_prime = (-q2 - q1)
    # select the one with the smallest norm
    dq_dt = torch.where(torch.norm(dq, dim=-1).unsqueeze(-1).repeat(1, 4) < torch.norm(dq_prime, dim=-1).unsqueeze(-1).repeat(1, 4), dq, dq_prime) / dt
    w = (2 * quat_mul(dq_dt, quat_conjugate(q1)))
    diff = dq_dt - 0.5 * quat_mul(w, q1)
    diff_prime = dq_dt - 0.5 * quat_mul(-w, q1)
    # select the one with the smallest norm
    final_w = torch.where(torch.norm(diff, dim=-1).unsqueeze(-1).repeat(1, 4) < torch.norm(diff_prime, dim=-1).unsqueeze(-1).repeat(1, 4), w, -w)
    # final_diff = dq_dt - 0.5 * quat_mul(final_w, q1)
    return final_w[:, :3]

# assuming q is unit quaternion
@torch.jit.script
def quat_log(q):
    return q[:, :3] * quat_theta(q)

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
def to_local_frame_spatial(spatial_transforms, local_transform):
    """
    spatial_transforms: (num_envs, transform_count, 7)
    local_transform: (num_envs, 7)
    """
    transform_count = spatial_transforms.shape[1]
    inv_pos = local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
    spatial_transforms[:, :, 0:3] = spatial_transforms[:, :, 0:3] - inv_pos
    return spatial_transforms

@torch.jit.script
def to_local_frame_pos(positions, local_transform):
    """
    positions: (num_envs, transform_count, 3)
    local_transform: (num_envs, 7)
    """
    transform_count = positions.shape[1]
    inv_pos = local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
    return positions - inv_pos

# @torch.jit.script
# def to_local_frame_w(w, local_transform):
#     """
#     w: (num_envs, transform_count, 3)
#     local_transform: (num_envs, 7)
#     """
#     transform_count = w.shape[1]
#     inv_local_transform = tf_inverse_xz(local_transform)
#     inv_pos = inv_local_transform[:, 0:3].unsqueeze(1).repeat(1, transform_count, 1)
#     inv_rot = inv_local_transform[:, 3:7].unsqueeze(1).repeat(1, transform_count, 1)
#     return quat_apply(inv_rot, w)

@torch.jit.script
def get_center_of_mass(body_I_m, body_X_sm):
    """
    body_I_m: (num_envs, num_links, 6, 6)
    body_X_sm: (num_envs, num_links, 7)
    return: the center of mass position in the local frame (num_envs, 3)
    """
    num_envs = body_I_m.shape[0]
    mass = body_I_m[:, :, 3, 3] # (num_envs, num_links, 3, 3)
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