# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys

import torch

from envs.dflex_env import DFlexEnv
from utils.forward_kinematics import eval_rigid_fk_grad, eval_rigid_id_grad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd, UsdGeom, Gf
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu




class SNUHumanoidFullDeepMimicEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True,
                 stochastic_init=False, MM_caching_frequency=1):

        self.filter = {}
        self.num_joint_q = 71
        self.num_joint_qd = 56
        self.num_muscles = 284
        num_obs = 187

        self.skeletons = []
        self.muscle_strengths = []

        self.inv_control_freq = 1

        self.num_dof = self.num_joint_q - 7  # 22

        self.str_scale = 0.6

        num_act = self.num_muscles

        super(SNUHumanoidFullDeepMimicEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed,
                                                 no_grad, render, device)

        self.stochastic_init = stochastic_init

        self.init_sim()

        # other parameters
        self.termination_height = 0.74
        self.termination_tolerance = 0.05
        self.height_rew_scale = 4.0
        self.action_strength = 100.0
        self.action_penalty = -0.001
        self.joint_vel_obs_scaling = 0.1

        # -----------------------
        # set up Usd renderer
        if (self.visualize):
            self.stage = Usd.Stage.CreateInMemory("HumanoidSNUDeepMimic_" + str(self.num_envs) + ".usd")

            self.renderer = df.UsdRenderer(self.model, self.stage)
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.ModelBuilder()
        self.reference_builder = df.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 48

        self.sim_dt = self.dt

        self.ground = True

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 0.5)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = tu.to_torch([10000.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1))

        self.start_pos = []

        if self.visualize:
            self.env_dist = 2.0
        else:
            self.env_dist = 0.  # set to zero for training for numerical consistency

        # start_height = 1.0

        self.asset_folder = os.path.join(os.path.dirname(__file__), 'assets/snu')
        asset_path = os.path.join(self.asset_folder, "human.xml")
        muscle_path = os.path.join(self.asset_folder, "muscle284.xml")

        for i in range(self.num_environments):
            self.skeletons.append(lu.Skeleton(asset_path, muscle_path, self.builder, self.filter,
                                       stiffness=5.0,
                                       damping=2.0, # stiffness and damping = k_p, k_d in PD control
                                       contact_ke=5e3,
                                       contact_kd=2e3,
                                       contact_kf=1e3,
                                       contact_mu=0.5,
                                       limit_ke=1e3,
                                       limit_kd=1e1,
                                       armature=0.05))

            # load reference skeleton for this skeleton
            lu.Skeleton(asset_path, None, self.reference_builder, self.filter,
                                       stiffness=5.0,
                                       damping=2.0, # stiffness and damping = k_p, k_d in PD control
                                       contact_ke=5e3,
                                       contact_kd=2e3,
                                       contact_kf=1e3,
                                       contact_mu=0.5,
                                       limit_ke=1e3,
                                       limit_kd=1e1,
                                       armature=0.05)

        num_q = int(len(self.builder.joint_q) / self.num_environments)
        num_qd = int(len(self.builder.joint_qd) / self.num_environments)
        assert num_q == self.num_joint_q, f"num_q does not match self.num_joint_q: num_q: {num_q}, self.num_joint_q: {self.num_joint_q}"
        assert num_qd == self.num_joint_qd, f"num_qd does not match self.num_joint_qd: num_qd: {num_qd}, self.num_joint_qd: {self.num_joint_qd}"

        num_muscles = len(self.skeletons[0].muscles)
        print("Num muscles: ", num_muscles)

        # Temporarily load muscle strength and multiply it to the action
        # TODO: correct the muscle strength calculation in dflex, and use action range [0, 1]
        for m in self.skeletons[0].muscles:
            self.muscle_strengths.append(self.str_scale * m.muscle_strength)

        for mi in range(len(self.muscle_strengths)):
            self.muscle_strengths[mi] = self.str_scale * self.muscle_strengths[mi]

        self.muscle_strengths = tu.to_torch(self.muscle_strengths, device=self.device).repeat(self.num_envs)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        # load reference motion
        self.reference_frame_time, self.reference_frame_count, self.reference_joint_q, self.reference_joint_q_mask, self.reference_joint_qd = \
            lu.load_bvh(os.path.join(self.asset_folder, "motion/walk.bvh"), self.skeletons[0].bvh_map, self.model, self.dt)

        # end effector indices
        self.end_effector_indices = [4, 9, 14, 18, 22] # FootThumbR, FootThumbL, Head, HandR, HandL
        self.end_effector_indices = tu.to_torch(self.end_effector_indices, device=self.device, dtype=torch.long)

        # reference model does not use simulation
        self.reference_model = self.reference_builder.finalize(self.device)

        self.integrator = df.SemiImplicitIntegrator()

        self.state = self.model.state()
        self.reference_state = self.reference_model.state()

        if (self.model.ground):
            self.model.collide(self.state)

        # copy the reference motion to the state
        self.copy_ref_pos_to_state()

    def render(self, mode='human'):
        render_asset_folder = 'C:/Users/1350a/dev/imo/DiffRL/envs/assets/snu'

        if self.visualize:
            self.render_time += self.dt * self.inv_control_freq
            with torch.no_grad():

                muscle_start = 0
                skel_index = 0

                for s in self.skeletons:
                    for mesh, link in s.mesh_map.items():

                        if link != -1:
                            X_sc = df.transform_expand(self.state.body_X_sc[link].tolist())

                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")

                            self.renderer.add_mesh(mesh, mesh_path, X_sc, 1.0, self.render_time)

                    for m in range(len(s.muscles)):

                        start = self.model.muscle_start[muscle_start + m].item()
                        end = self.model.muscle_start[muscle_start + m + 1].item()

                        points = []

                        for w in range(start, end):
                            link = self.model.muscle_links[w].item()
                            point = self.model.muscle_points[w].cpu().numpy()

                            X_sc = df.transform_expand(self.state.body_X_sc[link].cpu().tolist())

                            points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))

                        self.renderer.add_line_strip(points, name=s.muscles[m].name + str(skel_index),
                                                        radius=0.0075, color=(
                                self.model.muscle_activation[muscle_start + m] / self.muscle_strengths[m], 0.2,
                                0.5),
                                                        time=self.render_time)

                    muscle_start += len(s.muscles)
                    skel_index += 1

                # add the observations and reference poses
                # relative reference pos
                ref_root_transform = self.reference_state.joint_q.view(self.num_envs, -1)[:, 0:7]
                ref_relative_body_X_sc = tu.to_local_frame_spatial(self.reference_state.body_X_sc.view(self.num_envs, -1, 7).clone(), ref_root_transform).view(-1, 7)
                for s in self.skeletons:
                    for mesh, link in s.mesh_map.items():

                        if link != -1:
                            link_transform = ref_relative_body_X_sc[link]
                            link_transform[1] += 2
                            X_sc = df.transform_expand(link_transform.tolist())

                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")

                            self.renderer.add_mesh(f'{mesh}_relative_refpos', mesh_path, X_sc, 1.0, self.render_time)
                # relative
                relative_body_X_sc = self.obs_buf[0, 25:-1].view(-1, 7).clone()
                for s in self.skeletons:
                    for mesh, link in s.mesh_map.items():

                        if link != -1:
                            link_transform = relative_body_X_sc[link]
                            link_transform[1] += 2
                            X_sc = df.transform_expand(link_transform.tolist())

                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")

                            self.renderer.add_mesh(f'{mesh}_relative', mesh_path, X_sc, 1.0, self.render_time)


                # Phase: add a sphere to move [10, phase, 10]
                self.renderer.add_sphere((10, self.obs_buf[0, -1], 10), 0.1, "phase", self.render_time)

                # com_pos_local: add sphere
                com_pos_local = self.obs_buf[0, 14:17].view(-1).clone()
                com_pos_local[1] += 2
                self.renderer.add_sphere(com_pos_local.tolist(), 0.1, "com", self.render_time)

            self.renderer.update(self.state, self.render_time)

    def finalize_play(self):
        if self.visualize:
            try:
                self.stage.GetRootLayer().Export("outputs/HumanoidSNUDeepMimic_" + str(self.num_envs) + ".usd")
                print(f"Saved to outputs/HumanoidSNUDeepMimic_{self.num_envs}.usd")
            except Exception as e:
                print(f"USD save error: {e}")

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

        # AddBackward error??
        actions = torch.clip(actions, -1., 1.) * 0.5 + 0.5

        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        def create_hook(name):
            def hook(grad):
                if grad is not None:
                    pass
                    # return torch.clip(torch.nan_to_num(grad, 0.0, 100.0, -100.0, out=grad), -100.0, 100.0)
                    nan_count = torch.isnan(grad).sum()
                    inf_count = torch.isinf(grad).sum()
                    big_count = (torch.abs(grad) > 1e6).sum()
                    print(f'{name} grad nan count: {nan_count}, inf count: {inf_count}, big count: {big_count}.')
                    if 0 < nan_count < 10:
                        print(f'{name} grad nan index: {torch.nonzero(torch.isnan(grad)).squeeze(-1)}')
                    if 0 < inf_count < 10:
                        print(f'{name} grad inf index: {torch.nonzero(torch.isinf(grad)).squeeze(-1)}')
                    if 0 < big_count < 10:
                        print(f'{name} grad big index: {torch.nonzero(torch.abs(grad) > 1e6).squeeze(-1)}')
            return hook

        if self.state.joint_q.requires_grad:
            self.state.joint_q.register_hook(create_hook(f'joint_q_{self.num_frames}'))
        if self.state.joint_qd.requires_grad:
            self.state.joint_qd.register_hook(create_hook(f'joint_qd_{self.num_frames}'))
        if actions.requires_grad:
            actions.register_hook(create_hook(f'actions_{self.num_frames}'))
        #################################################

        self.actions = actions.clone()

        # simulate the model
        for ci in range(self.inv_control_freq):
            self.model.muscle_activation = actions.view(-1) * self.muscle_strengths

            self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps,
                                                 self.MM_caching_frequency)
            self.sim_time += self.sim_dt
        
        # update the reference model
        with torch.no_grad():
            self.update_reference_model()

        # rehook!
        if self.state.joint_q.requires_grad:
            self.state.joint_q.register_hook(create_hook(f'joint_q_{self.num_frames}_after_step'))
        if self.state.joint_qd.requires_grad:
            self.state.joint_qd.register_hook(create_hook(f'joint_qd_{self.num_frames}_after_step'))

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        # obs_buf and rew_buf gradient check
        if self.obs_buf.requires_grad:
            self.obs_buf.register_hook(create_hook(f'obs_buf_{self.num_frames - 1}'))
        if self.rew_buf.requires_grad:
            self.rew_buf.register_hook(create_hook(f'rew_buf_{self.num_frames - 1}'))

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
            }

        if len(env_ids) > 0:
            self.reset(env_ids)

        with df.ScopedTimer("render", False):
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # copy the reference motion to the state
            # randomize the reset start frame to learn all reference frames uniformly

            # randomization
            if self.stochastic_init:
                self.progress_buf[env_ids] = 0
                self.copy_ref_pos_to_state(env_ids)
                # # start pos randomization
                # self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state.joint_q.view(self.num_envs, -1)[
                #                                                            env_ids, 0:3] + 0.05 * (torch.rand(
                #     size=(len(env_ids), 3), device=self.device) - 0.5) + torch.tensor([0.0, 0.025, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
                # # start rot randomization
                # angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 36.0
                # axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                # self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(
                #     self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                # # start vel randomization
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] += 0.05 * (
                            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)
            else:
                self.progress_buf[env_ids] = 0
                self.copy_ref_pos_to_state(env_ids)

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device=self.device,
                                                   dtype=torch.float)

            self.calculateObservations()

        return self.obs_buf


    def clear_grad(self, checkpoint=None):
        """
        cut off the gradient from the current state to previous states
        """
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}  # NOTE: any other things to restore?
                checkpoint['joint_q'] = self.state.joint_q.clone()
                checkpoint['joint_qd'] = self.state.joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()

    def initialize_trajectory(self):
        """
        This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
        It has to be called every time the algorithm starts an episode and it returns the observation vectors
        """
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
        # forward kinematics
        # body_X_sc, body_X_sm = eval_rigid_fk_grad(self.model, self.state.joint_q)
        # joint_S_s, body_I_s, body_v_s, body_f_s, body_a_s = eval_rigid_id_grad(self.model, self.state.joint_q, self.state.joint_qd, body_X_sc, body_X_sm)

        # add root pos, root rot, and torso pos.
        root_transform = self.state.body_X_sc.view(self.num_envs, -1, 7)[:, 0, :].squeeze(1)
        root_pos = root_transform[:, 0:3]
        root_rot = root_transform[:, 3:7]
        torso_pos = self.state.body_X_sc.view(self.num_envs, -1, 7)[:, 13, 0:3]
        torso_rot = self.state.body_X_sc.view(self.num_envs, -1, 7)[:, 12, 3:7]

        # DeepMimic states (observations): each link's position and rotation relative to the root joint, phase of the motion
        # 1. compute relative body X_sc
        relative_body_X_sc = tu.to_local_frame_spatial(self.state.body_X_sc.view(self.num_envs, -1, 7).clone(), root_transform)
    
        # 2. compute phase of the motion
        progress_time = self.progress_buf * self.dt
        max_time = self.reference_frame_time * self.reference_frame_count
        phase = (progress_time / max_time) % 1.0

        # 3. calculate the center of mass position, expressed in the local frame
        com_pos = tu.get_center_of_mass(self.model.body_I_m.view(self.num_envs, -1, 6, 6), self.state.body_X_sm.view(self.num_envs, -1, 7))
        com_pos_local = tu.to_local_frame_pos(com_pos.view(self.num_envs, 1, 3), root_transform)

        lin_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 0:3]
        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(root_pos, ang_vel, dim=-1)

        to_target = self.targets - root_pos
        to_target[:, 1] = 0.0

        target_dirs = tu.normalize(to_target)

        up_vec = tu.quat_rotate(root_rot, self.basis_vec1)
        heading_vec = tu.quat_rotate(root_rot, self.basis_vec0)

        # TODO: check if we can add phase (which is not differentiable) in observations
        self.obs_buf = torch.cat([
            root_pos.view(self.num_envs, -1), # 0:3
            root_rot.view(self.num_envs, -1), # 3:7
            torso_pos.view(self.num_envs, -1), # 7:10
            torso_rot.view(self.num_envs, -1), # 10:14
            com_pos_local.view(self.num_envs, -1), # 14:17
            lin_vel.view(self.num_envs, -1), # 17:20
            ang_vel.view(self.num_envs, -1), # 20:23
            up_vec[:, 1:2],  # 23:24
            (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1), # 24:25
            relative_body_X_sc.view(self.num_envs, -1),
            phase.view(self.num_envs, 1),
        ], dim=-1)

    def calculateReward(self):
        reward_type = 'deepmimic'
        if reward_type == 'deepmimic':
            # DeepMimic reward: pose reward + velocity reward + end-effector reward + center-of-mass reward
            w_p = 0.65
            w_v = 0.1
            w_e = 0.15
            w_c = 0.1

            # relative body X_sc for reference state
            relative_body_X_sc = self.obs_buf[:, 25:-1].view(self.num_envs, -1, 7)
            ref_root_transform = self.reference_state.body_X_sc.view(self.num_envs, -1, 7)[:, 0, :].squeeze(1)
            ref_relative_body_X_sc = tu.to_local_frame_spatial(self.reference_state.body_X_sc.view(self.num_envs, -1, 7).clone(), ref_root_transform)
            
            body_pos_diff = relative_body_X_sc[:, :, 0:3] - ref_relative_body_X_sc[:, :, 0:3]

            # pos reward: exp(-2 * sum(body quat, ref body quat diff **2))
            body_quat = relative_body_X_sc[:, :, 3:7]
            ref_body_quat = ref_relative_body_X_sc[:, :, 3:7]
            body_quat_diff = tu.quat_diff(body_quat, ref_body_quat)
            pos_reward = torch.exp(-2 * torch.sum(torch.sum(body_quat_diff ** 2, dim=-1), dim=-1))

            # velocity reward: exp(-0.1 * sum(body w, ref body w diff **2))
            body_w_diff = self.state.body_v_s.view(self.num_envs, -1, 6)[:, :, 0:3] - self.reference_state.body_v_s.view(self.num_envs, -1, 6)[:, :, 0:3]
            vel_reward = torch.exp(-0.1 * torch.sum(torch.sum(body_w_diff ** 2, dim=-1), dim=-1))

            # end-effector reward: exp(-40 * sum(end-effector pos, ref end-effector pos diff **2))
            end_effector_pos = relative_body_X_sc[:, self.end_effector_indices, 0:3]
            ref_end_effector_pos = ref_relative_body_X_sc[:, self.end_effector_indices, 0:3]
            end_effector_pos_diff = end_effector_pos - ref_end_effector_pos
            end_effector_reward = torch.exp(-40 * torch.sum(torch.sum(end_effector_pos_diff ** 2, dim=-1), dim=-1))

            # center-of-mass reward: exp(-10 * sum(com pos, ref com pos diff **2))
            com_pos_local = self.obs_buf[:, 14:17]
            ref_com_pos = tu.get_center_of_mass(self.model.body_I_m.view(self.num_envs, -1, 6, 6), self.reference_state.body_X_sm.view(self.num_envs, -1, 7))
            ref_com_pos_local = tu.to_local_frame_pos(ref_com_pos.view(self.num_envs, 1, 3), ref_root_transform).view(self.num_envs, -1)
            com_pos_diff = com_pos_local - ref_com_pos_local
            com_reward = torch.exp(-10 * torch.sum(com_pos_diff ** 2, dim=-1))

            imitation_reward = pos_reward.clone()

        elif reward_type == 'diffmimic':
            # diffMimic reward: pos + rot + vel + ang
            w_pos = 1
            w_rot = 0.5
            w_vel = 0.01
            w_ang = 0.01

            # relative body X_sc for reference state
            relative_body_X_sc = self.obs_buf[:, 25:-1].view(self.num_envs, -1, 7)
            ref_root_transform = self.reference_state.body_X_sc.view(self.num_envs, -1, 7)[:, 0, :].squeeze(1)
            ref_relative_body_X_sc = tu.to_local_frame_spatial(self.reference_state.body_X_sc.view(self.num_envs, -1, 7).clone(), ref_root_transform)

            body_pos_diff = relative_body_X_sc[:, :, 0:3] - ref_relative_body_X_sc[:, :, 0:3]
            pos_reward = -torch.mean(torch.sum(body_pos_diff ** 2, dim=-1), dim=-1)

            body_rot_diff = tu.quat_diff(relative_body_X_sc[:, :, 3:7], ref_relative_body_X_sc[:, :, 3:7])
            rot_reward = -torch.mean(torch.sum(body_rot_diff ** 2, dim=-1), dim=-1)

            body_vel_diff = self.state.body_v_s.view(self.num_envs, -1, 6)[:, :, 3:6] - self.reference_state.body_v_s.view(self.num_envs, -1, 6)[:, :, 3:6]
            vel_reward = -torch.mean(torch.sum(body_vel_diff ** 2, dim=-1), dim=-1)

            body_ang_vel_diff = self.state.body_v_s.view(self.num_envs, -1, 6)[:, :, 0:3] - self.reference_state.body_v_s.view(self.num_envs, -1, 6)[:, :, 0:3]
            ang_vel_reward = -torch.mean(torch.sum(body_ang_vel_diff ** 2, dim=-1), dim=-1)

            imitation_reward = w_pos * pos_reward + w_rot * rot_reward + w_vel * vel_reward + w_ang * ang_vel_reward

        else:
            imitation_reward = 0.0

        # goal reward
        up_reward = 0.1 * self.obs_buf[:, 23]
        heading_reward = self.obs_buf[:, 24]

        height_diff = self.obs_buf[:, 8] - self.termination_height
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward,
                                    height_reward)  # JIE: not smooth
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        act_penalty = torch.sum(torch.abs(self.actions),
                                dim=-1) * self.action_penalty  # torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        progress_reward = self.obs_buf[:, 17]

        goal_reward = up_reward + heading_reward + height_reward + progress_reward

        w_g = 0.1
        w_i = 0.9

        # print the mean reward
        # print(f"mean imitation reward: {torch.mean(imitation_reward).item()}, mean goal reward: {torch.mean(goal_reward).item()}")

        self.rew_buf = imitation_reward

        # reset agents (early termination)
        self.reset_buf = torch.where(self.obs_buf[:, 8] < self.termination_height, torch.ones_like(self.reset_buf),
                                        self.reset_buf)
        
        # # if pos reward is less than -1, reset
        # body_pos_diff = relative_body_X_sc[:, :, 0:3] - ref_relative_body_X_sc[:, :, 0:3]
        self.reset_buf = torch.where(torch.mean(torch.sum(body_pos_diff ** 2, dim=-1), dim=-1) > 0.2, torch.ones_like(self.reset_buf),
                                        self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

        # an ugly fix for simulation nan values
        nan_masks = torch.logical_or(torch.isnan(self.obs_buf).sum(-1) > 0, torch.logical_or(
            torch.isnan(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0,
            torch.isnan(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        inf_masks = torch.logical_or(torch.isinf(self.obs_buf).sum(-1) > 0, torch.logical_or(
            torch.isinf(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0,
            torch.isinf(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        invalid_value_masks = torch.logical_or(
            (torch.abs(self.state.joint_q.view(self.num_environments, -1)) > 1e6).sum(-1) > 0,
            (torch.abs(self.state.joint_qd.view(self.num_environments, -1)) > 1e6).sum(-1) > 0)
        invalid_masks = torch.logical_or(invalid_value_masks, torch.logical_or(nan_masks, inf_masks))

        if sum(invalid_masks) > 0:
            print("Invalid values detected")

        self.reset_buf = torch.where(invalid_masks, torch.ones_like(self.reset_buf), self.reset_buf)

        self.rew_buf[invalid_masks] = 0.0

    def update_reference_model(self):
        """
        Update the reference model's state from the phase of the motion.
        """
        
        frame_index = torch.round((self.progress_buf * self.dt) / self.reference_frame_time).long() % self.reference_frame_count
        next_state = self.reference_model.state()

        # update joint_q and joint_qd
        next_state.joint_q[:] = self.reference_joint_q[frame_index, :].view(-1)
        next_state.joint_qd[:] = self.reference_joint_qd[frame_index, :].view(-1)

        # rotate the torso
        # next_state.joint_q.view(self.num_envs, -1)[:, 3:7] = tu.quat_from_angle_axis(torch.tensor([math.pi * 0.5]).repeat(self.num_envs).to(self.device), self.y_unit_tensor)

        # perform forward kinematics
        body_X_sc, body_X_sm = eval_rigid_fk_grad(self.reference_model, next_state.joint_q)
        next_state.body_X_sc = body_X_sc
        next_state.body_X_sm = body_X_sm
        joint_S_s, body_I_s, body_v_s, body_f_s, body_a_s = eval_rigid_id_grad(self.reference_model, next_state.joint_q, next_state.joint_qd, body_X_sc, body_X_sm)
        next_state.joint_S_s = joint_S_s
        next_state.body_I_s = body_I_s
        next_state.body_v_s = body_v_s
        next_state.body_f_s = body_f_s
        next_state.body_a_s = body_a_s
        
        self.reference_state = next_state

    def copy_ref_pos_to_state(self, env_ids=None):
        """
        Copy the reference motion to the state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.state.joint_q = self.state.joint_q.clone()
        self.state.joint_qd = self.state.joint_qd.clone()

        frame_index = torch.round((self.progress_buf * self.dt) / self.reference_frame_time).long() % self.reference_joint_q.shape[0]
        # masked values are 0, so we don't need to use mask to select valid values
        self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = self.reference_joint_q[frame_index[env_ids], :]
        self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = self.reference_joint_qd[frame_index[env_ids], :]

        # reset position
        self.state.joint_q.view(self.num_envs, -1)[env_ids, 0] = 0.0
        # ground height correction
        self.state.joint_q.view(self.num_envs, -1)[env_ids, 1] = 1.0
        self.state.joint_q.view(self.num_envs, -1)[env_ids, 2] = 0.0
        # self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_from_angle_axis(torch.tensor([math.pi * 0.5]).repeat(len(env_ids)).to(self.device), torch.tensor([0.0, 1.0, 0.0]).view(1, -1).repeat(len(env_ids), 1).to(self.device))