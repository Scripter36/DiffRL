# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter
import yaml
import mlflow
from mlflow.entities import Metric
from utils.mlflow_utils import enqueue_model_save, get_current_run, get_mlflow_client, stop_save_worker
import dflex as df

import envs
import models.actor
import models.critic
from utils.common import *
import utils.torch_utils as tu
from utils.running_mean_std import RunningMeanStd
from utils.dataset import CriticDataset
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
from utils.mlflow_utils import get_current_run

import torch.nn as nn

class SHACCheckpoint(nn.Module):
    def __init__(self, actor, critic, target_critic, obs_rms, ret_rms):
        super(SHACCheckpoint, self).__init__()
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.obs_rms = obs_rms
        self.ret_rms = ret_rms

    def forward(self, x):
        raise NotImplementedError("This checkpoint model is for storage only and not for inference")
    
    def cpu(self):
        # Clone models and move to CPU
        actor_cpu = copy.deepcopy(self.actor).to('cpu')
        critic_cpu = copy.deepcopy(self.critic).to('cpu')
        target_critic_cpu = copy.deepcopy(self.target_critic).to('cpu')
        # already copied, do not need to copy again
        obs_rms_cpu = self.obs_rms.to('cpu') if self.obs_rms is not None else None
        ret_rms_cpu = self.ret_rms.to('cpu') if self.ret_rms is not None else None
        return SHACCheckpoint(actor_cpu, critic_cpu, target_critic_cpu, obs_rms_cpu, ret_rms_cpu)

class SHAC:
    def __init__(self, cfg, render_name=None):
        self.cfg = cfg
        seed = cfg["params"]["general"]["seed"]
        stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True)

        if seed is not None:
            seeding(seed)
            save_rng_state()
        if render_name is None:
            # use experiment name
            render_name = mlflow.get_experiment(get_current_run().info.experiment_id).name
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            render = cfg["params"]["general"]["render"], \
                            seed = seed, \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = stochastic_init, \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            no_grad = False, \
                            render_name = render_name)

        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_obs = ', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg['params']['config'].get('gamma', 0.99)
        
        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step') # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        
        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)

        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]
        self.use_grad_per_env = cfg["params"]["config"].get('use_grad_per_env', False)
        
        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            user_wants_deterministic = cfg['params']['config']['player'].get('deterministic', False) or cfg['params']['config']['player'].get('determenistic', False)
            self.stochastic_evaluation = not user_wants_deterministic
            self.steps_num = self.env.episode_length
        
        set_torch_deterministic(self.stochastic_evaluation)

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP') # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = cfg["params"]["network"].get("critic", 'CriticMLP')
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor = actor_fn(self.num_obs, self.num_actions, cfg['params']['network'], device = self.device)
        critic_fn = getattr(models.critic, self.critic_name)
        self.critic = critic_fn(self.num_obs, cfg['params']['network'], device = self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
    
        if cfg['params']['general']['train']:
            self.save('init_policy')
    
        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = cfg['params']['config']['betas'], lr = self.critic_lr)

        # replay buffer
        self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_obs), dtype = torch.float32, device = self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((self.num_envs), dtype = torch.float32, device = self.device)

        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        
        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()
        
    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        
        if self.use_grad_per_env:
            actor_loss_per_env = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        else:
            actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
                
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs, deterministic = deterministic)

            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))
            
            with torch.no_grad():
                raw_rew = rew.clone()
            
            # scale the reward
            rew = rew * self.rew_scale
            
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                    
                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1
        
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            next_values[i + 1] = self.target_critic(obs).squeeze(-1)

            for id in done_env_ids:
                if torch.isnan(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or torch.isinf(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or (torch.abs(extra_info['obs_before_reset'][id]) > 1e6).sum() > 0: # ugly fix for nan values
                    next_values[i + 1, id] = 0.
                elif self.episode_length[id] < self.max_episode_length: # early termination
                    next_values[i + 1, id] = 0.
                else: # otherwise, use terminal value critic to estimate the long-term performance
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(extra_info['obs_before_reset'][id])
                    else:
                        real_obs = extra_info['obs_before_reset'][id]
                    next_values[i + 1, id] = self.target_critic(real_obs).squeeze(-1)
            
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError
            
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                if self.use_grad_per_env:
                    actor_loss_per_env[done_env_ids] += - rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                # terminate all envs at the end of optimization iteration
                if self.use_grad_per_env:
                    actor_loss_per_env[:] += - rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
        
            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids.to('cpu')])
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                            print('ep loss error')
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.
                        self.episode_discounted_loss[done_env_id] = 0.
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.

        if self.use_grad_per_env:
            actor_loss_per_env /= self.steps_num
        else:
            actor_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
            actor_loss_per_env = actor_loss_per_env * torch.sqrt(ret_var + 1e-6)
            
        if self.use_grad_per_env:
            self.actor_loss = torch.mean(actor_loss_per_env).detach().cpu().item()
        else:
            self.actor_loss = actor_loss.detach().cpu().item()
            
        self.step_count += self.steps_num * self.num_envs

        if self.use_grad_per_env:
            return actor_loss_per_env
        else:
            return actor_loss
    
    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic = False):
        if deterministic:
            restore_rng_state()

        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = int)
        episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)

        obs = self.env.reset()

        games_cnt = 0
        game_bar = tqdm(total=num_games, desc='Running games', position=0)
        while games_cnt < num_games:
            env_bars = [tqdm(total=self.env.episode_length, desc='Evaluating policy', position=i+1) for i in range(self.num_envs)]
            while True:
                if self.obs_rms is not None:
                    obs = self.obs_rms.normalize(obs)

                actions = self.actor(obs, deterministic=deterministic)

                obs, rew, done, _ = self.env.step(torch.tanh(actions))

                episode_length += 1
                for env_bar in env_bars:
                    env_bar.update(1)

                done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

                episode_loss -= rew
                episode_discounted_loss -= episode_gamma * rew
                episode_gamma *= self.gamma
                
                if len(done_env_ids) > 0:
                    break
            
            for done_env_id in done_env_ids:
                print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                episode_loss_his.append(episode_loss[done_env_id].item())
                episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                episode_length_his.append(episode_length[done_env_id].item())
                episode_loss[done_env_id] = 0.
                episode_discounted_loss[done_env_id] = 0.
                episode_length[done_env_id] = 0
                episode_gamma[done_env_id] = 1.
                games_cnt += 1
                game_bar.update(1)
                env_bars[done_env_id].close()
                env_bars[done_env_id] = tqdm(total=self.env.episode_length, desc='Evaluating policy', position=done_env_id+1)
        game_bar.close()
        
        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))
 
        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError
            
    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games = num_games, deterministic = not self.stochastic_evaluation)
        print_info('mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss, mean_policy_discounted_loss, mean_episode_length))
        
    def train(self):
        torch.autograd.set_detect_anomaly(True)
        # load checkpoint
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            if self.use_grad_per_env:
                actor_loss_per_env = self.compute_actor_loss()
            else:
                actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")

            if self.use_grad_per_env:
                parameters = list(self.actor.parameters())
                final_grads = []
                self.valid_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

                for i in range(self.num_envs):
                    grads = torch.autograd.grad(actor_loss_per_env[i], parameters, retain_graph=True)
                    grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
                    if torch.isnan(grad_norm) or grad_norm > 1000000.:
                        continue
                    self.valid_env_mask[i] = True

                    if len(final_grads) == 0:
                        final_grads = grads
                    else:
                        for g1, g2 in zip(final_grads, grads):
                            g1 += g2

                valid_envs = self.valid_env_mask.sum()
                for p, g in zip(parameters, final_grads):
                    p.grad = g / valid_envs

                if valid_envs == 0:
                    print('all envs are invalid.')
                    raise ValueError

                if valid_envs != self.num_envs:
                    print(f'truncated {self.num_envs - valid_envs} envs. invalid envs: {(~self.valid_env_mask).nonzero()}')
            else:
                self.valid_env_mask = None
                actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 
                
                # sanity check
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.:
                    print('NaN gradient. grad norm before clip = {:.2f}'.format(self.grad_norm_before_clip))
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            if self.use_grad_per_env:
                return torch.mean(actor_loss_per_env)
            else:
                return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last = False, valid_env_mask = self.valid_env_mask)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    
                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.time_report.end_timer("critic training")

            self.iter_count += 1
            
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
                
                self.writer.add_scalar('policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                self.writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, grad norm after clip {:.2f}'.format(
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length, 
                    self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, 
                    self.grad_norm_before_clip, self.grad_norm_after_clip))
            self.writer.flush()

            # ---- MLFlow logging (added) ----
            all_metrics = []
            timestamp_now = round(time.time() * 1000)
            all_metrics.append(Metric(key="lr_iter", value=lr, step=self.iter_count, timestamp=timestamp_now))
            all_metrics.append(Metric(key="actor_loss_iter", value=self.actor_loss, step=self.iter_count, timestamp=timestamp_now))
            all_metrics.append(Metric(key="value_loss_iter", value=self.value_loss, step=self.iter_count, timestamp=timestamp_now))
            if len(self.episode_loss_his) > 0:
                all_metrics.append(Metric(key="policy_loss_iter", value=mean_policy_loss, step=self.iter_count, timestamp=timestamp_now))
                all_metrics.append(Metric(key="rewards_iter", value=-mean_policy_loss, step=self.iter_count, timestamp=timestamp_now))
                all_metrics.append(Metric(key="policy_discounted_loss_iter", value=mean_policy_discounted_loss, step=self.iter_count, timestamp=timestamp_now))
                all_metrics.append(Metric(key="best_policy_loss_iter", value=self.best_policy_loss, step=self.iter_count, timestamp=timestamp_now))
                all_metrics.append(Metric(key="episode_lengths_iter", value=mean_episode_length, step=self.iter_count, timestamp=timestamp_now))
                all_metrics.append(Metric(key="grad_norm_before_clip_iter", value=self.grad_norm_before_clip, step=self.iter_count, timestamp=timestamp_now))

            get_mlflow_client().log_batch(get_current_run().info.run_id, all_metrics)
            # ---- End MLFlow logging ----

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(f"iter{self.iter_count}_reward{-mean_policy_loss:.3f}")

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)

        self.time_report.end_timer("algorithm")

        self.time_report.report()
        
        self.save('final_policy')

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.log_dir, 'episode_loss_his.npy'), 'wb'), self.episode_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_discounted_loss_his.npy'), 'wb'), self.episode_discounted_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # evaluate the final policy's performance
        self.run(self.num_envs)

        self.close()
    
    def play(self, cfg):
        self.load(cfg['params']['general']['checkpoint'])
        self.run(cfg['params']['config']['player']['games_num'])
        
    def save(self, filename = None):
        if filename is None:
            filename = 'best_policy'
        enqueue_model_save(SHACCheckpoint(self.actor, self.critic, self.target_critic, self.obs_rms, self.ret_rms).cpu(), filename)

    def load(self, path):
        print("Loading checkpoint from", path)
        loaded_model = mlflow.pytorch.load_model(path)
        self.actor = loaded_model.actor.to(self.device)
        self.critic = loaded_model.critic.to(self.device)
        self.target_critic = loaded_model.target_critic.to(self.device)
        self.obs_rms = loaded_model.obs_rms.to(self.device) if loaded_model.obs_rms is not None else None
        self.ret_rms = loaded_model.ret_rms.to(self.device) if loaded_model.ret_rms is not None else None

    def close(self):
        self.writer.close()
        stop_save_worker()
