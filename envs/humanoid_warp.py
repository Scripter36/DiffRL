# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.dflex_env import DFlexEnv
import math
import torch

import os
import sys

from envs.warp_env import WarpEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import warp as wp
import warp.sim.render
import warp.sim

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu


class HumanoidWarpEnv(WarpEnv):

    def __init__(self, render=False, device='cuda', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1):
        num_obs = 76
        num_act = 21

        super(HumanoidWarpEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init

        self.init_sim()

        # other parameters
        self.termination_height = 0.74
        self.motor_strengths = [
            200, 
            200, 
            200, 
            200, 
            200, 
            600, 
            400, 
            100, 
            100, 
            200, 
            200, 
            600, 
            400, 
            100,  
            100,
            100, 
            100, 
            200, 
            100, 
            100, 
            200]

        self.motor_scale = 0.35

        self.motor_strengths = tu.to_torch(self.motor_strengths, dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.action_penalty = -0.002
        self.joint_vel_obs_scaling = 0.1
        self.termination_tolerance = 0.1
        self.height_rew_scale = 10.0

        #-----------------------
        # set up Usd renderer
        if self.visualize:
            self.stage = Usd.Stage.CreateNew("outputs/" + "HumanoidWarp_" + str(self.num_envs) + ".usd")

            self.renderer = wp.sim.render.SimRendererUsd(self.model, self.stage, scaling=100.0)
            # self.renderer.draw_points = True
            # self.renderer.draw_springs = True
            # self.renderer.draw_shapes = True
            self.render_time = 0.0

            self.render()

    def init_sim(self):
        self.builder = wp.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 48
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 28
        self.num_joint_qd = 27

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rot = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = tu.to_torch([200.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        start_pos = []

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        start_height = 1.35

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):
            wp.sim.parse_mjcf(os.path.join(asset_folder, "humanoid.xml"), self.builder,
                stiffness=5.0,
                damping=0.1,
                contact_ke=2.e+4,
                contact_kd=5.e+3,
                contact_kf=1.e+3,
                contact_mu=0.75,
                limit_ke=1.e+3,
                limit_kd=1.e+1,
                armature=0.007,
                up_axis='Y')

            # base transform
            start_pos_z = i*self.env_dist
            start_pos.append([0.0, start_height, start_pos_z])

            self.builder.joint_q[i*self.num_joint_q:i*self.num_joint_q + 3] = start_pos[-1]
            self.builder.joint_q[i*self.num_joint_q + 3:i*self.num_joint_q + 7] = self.start_rot

        num_q = int(len(self.builder.joint_q)/self.num_environments)
        num_qd = int(len(self.builder.joint_qd)/self.num_environments)
        print(num_q, num_qd)

        print("Start joint_q: ", self.builder.joint_q[0:num_q])

        self.start_joint_q = self.builder.joint_q[7:num_q].copy()
        self.start_joint_target = self.start_joint_q.copy()

        self.start_pos = tu.to_torch(start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device.type)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state = self.model.state()
        self.control = self.model.control()
        self.tape = wp.Tape()

        num_act = int(len(self.control.joint_act) / self.num_environments)
        print('num_act = ', num_act)

        # if self.model.ground:
        #     self.model.collide(self.state)

    def render(self, mode = 'human'):
        if self.visualize:
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state)
            self.renderer.end_frame()
            self.render_time += self.dt
            # self.renderer.update(self.state, self.render_time)

            if self.num_frames == 1:
                try:
                    self.stage.Save()
                except Exception as e:
                    print("USD save error: ", e)

                self.num_frames -= 1

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

        # todo - make clip range a parameter
        actions = torch.clip(actions, -1., 1.)

        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        # def create_hook():
        #     def hook(grad):
        #         torch.nan_to_num(grad, 0.0, 0.0, 0.0, out = grad)
        #     return hook
        #
        # if self.state.joint_q.requires_grad:
        #     self.state.joint_q.register_hook(create_hook())
        # if self.state.joint_qd.requires_grad:
        #     self.state.joint_qd.register_hook(create_hook())
        # if actions.requires_grad:
        #     actions.register_hook(create_hook())
        #################################################

        self.actions = actions.clone()
        joint_act = wp.to_torch(self.control.joint_act).view(self.num_envs, -1)
        
        joint_act[:, :] = actions * self.motor_scale * self.motor_strengths
        actuation = self.control.joint_act

        for _ in range(self.sim_substeps):
            self.control.joint_act = actuation
            out_state = self.model.state()
            wp.sim.collide(self.model, self.state)
            self.integrator.simulate(self.model, self.state, out_state, self.sim_dt / float(self.sim_substeps), self.control)
            self.state = out_state

        # df.SemiImplicitIntegrator().forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        # self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if not self.no_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        if len(env_ids) > 0:
           self.reset(env_ids)

        self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = wp.clone(self.state.joint_q, requires_grad = False)
            self.state.joint_qd = wp.clone(self.state.joint_qd, requires_grad = False)

            joint_q = wp.to_torch(self.state.joint_q).view(self.num_envs, -1)
            joint_qd = wp.to_torch(self.state.joint_qd).view(self.num_envs, -1)

            # fixed start state
            joint_q[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            joint_q[env_ids, 3:7] = self.start_rotation.clone()
            joint_q[env_ids, 7:] = self.start_joint_q.clone()
            joint_qd[env_ids, :] = 0.

            # randomization
            if self.stochastic_init:
                joint_q[env_ids, 0:3] = joint_q[env_ids, 0:3] + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
                joint_q[env_ids, 3:7] = tu.quat_mul(joint_q[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                joint_q[env_ids, 7:] = joint_q[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
                joint_qd[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = wp.clone(self.state.joint_q, requires_grad = False)
                checkpoint['joint_qd'] = wp.clone(self.state.joint_qd, requires_grad = False)
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            self.state = self.model.state()
            self.state.joint_q = wp.clone(checkpoint['joint_q'], requires_grad = False)
            self.state.joint_qd = wp.clone(checkpoint['joint_qd'], requires_grad = False)
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()

    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()

        return checkpoint

    def calculateObservations(self):
        joint_q = wp.to_torch(self.state.joint_q).view(self.num_envs, -1)
        joint_qd = wp.to_torch(self.state.joint_qd).view(self.num_envs, -1)

        torso_pos = joint_q[:, 0:3]
        torso_rot = joint_q[:, 3:7]
        lin_vel = joint_qd[:, 3:6]
        ang_vel = joint_qd[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        self.obs_buf = torch.cat([torso_pos[:, 1:2], # 0
                                torso_rot, # 1:5
                                lin_vel, # 5:8
                                ang_vel, # 8:11
                                joint_q[:, 7:], # 11:32
                                self.joint_vel_obs_scaling * joint_qd[:, 6:], # 32:53
                                up_vec[:, 1:2], # 53:54
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 54:55
                                self.actions.clone()], # 55:76
                                dim = -1)

    def calculateReward(self):
        joint_q = wp.to_torch(self.state.joint_q).view(self.num_envs, -1)
        joint_qd = wp.to_torch(self.state.joint_qd).view(self.num_envs, -1)

        up_reward = 0.1 * self.obs_buf[:, 53]
        heading_reward = self.obs_buf[:, 54]

        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)
        
        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = progress_reward + up_reward + heading_reward + height_reward + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # reset agents - 1 if object is below the termination height or the episode length is reached
        self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # an ugly fix for simulation nan values
        nan_masks = torch.logical_or(torch.isnan(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isnan(joint_q).sum(-1) > 0, torch.isnan(joint_qd).sum(-1) > 0))
        inf_masks = torch.logical_or(torch.isinf(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isinf(joint_q).sum(-1) > 0, torch.isinf(joint_qd).sum(-1) > 0))
        invalid_value_masks = torch.logical_or((torch.abs(joint_q) > 1e6).sum(-1) > 0,
                                                (torch.abs(joint_qd) > 1e6).sum(-1) > 0)
        invalid_masks = torch.logical_or(invalid_value_masks, torch.logical_or(nan_masks, inf_masks))
        
        self.reset_buf = torch.where(invalid_masks, torch.ones_like(self.reset_buf), self.reset_buf)
    
        self.rew_buf[invalid_masks] = 0.