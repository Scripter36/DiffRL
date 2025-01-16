# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle = False, drop_last = False, valid_env_mask = None):
        steps_num, envs_num, obs_dim = obs.shape
        self.obs = obs.view(-1, obs_dim)
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            raise NotImplementedError("shuffle is not implemented")
            self.shuffle()

        self.valid_obs_length = steps_num * envs_num

        if valid_env_mask is not None:
            self.valid_env_mask = valid_env_mask
            valid_env_num = self.valid_env_mask.sum().item()
            self.valid_obs_length = valid_env_num * steps_num
            self.env_index_map = ((self.valid_env_mask.nonzero().view(-1) * steps_num).unsqueeze(1).repeat(1, steps_num) + torch.arange(steps_num, device=self.valid_env_mask.device).unsqueeze(0).repeat(valid_env_num, 1)).view(-1)
        
        if drop_last:
            self.length = self.valid_obs_length // self.batch_size
        else:
            self.length = ((self.valid_obs_length - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.valid_env_mask is None:
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, self.valid_obs_length)
            return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}
        else:
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, self.valid_obs_length)
            return {'obs': self.obs[self.env_index_map[start_idx:end_idx], :], 'target_values': self.target_values[self.env_index_map[start_idx:end_idx]]}

