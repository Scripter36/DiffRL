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




class SNUHumanoidDeepMimicEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True,
                 stochastic_init=False, MM_caching_frequency=1, render_name=None):
        self.filter = { "Pelvis", "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR", "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL"}

        self.skeletons = []
        self.muscle_strengths = []

        self.mtu_actuations = True 

        self.inv_control_freq = 1

        # "humanoid_snu_lower"
        self.num_joint_q = 29
        self.num_joint_qd = 24

        self.num_dof = self.num_joint_q - 7 # 22
        self.num_muscles = 152

        num_act = self.num_muscles
        num_obs = self.phase_range.stop

        super(SNUHumanoidDeepMimicEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed,
                                                 no_grad, render, render_name, device)

        self.stochastic_init = stochastic_init

        self.offset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.start_frame_offset = 0

        # Initialize render buffer
        self.render_buffer = []  # List to store frames for all environments
        self.render_buffer_max_size = episode_length  # Maximum number of frames to store

        self.init_sim()

        # other parameters
        self.termination_height = 0.46
        self.termination_tolerance = 0.05
        self.height_rew_scale = 4.0
        self.action_strength = 100.0
        self.action_penalty = -0.001
        self.joint_vel_obs_scaling = 0.1

        # -----------------------
        # set up Usd renderer
        if (self.visualize):
            if self.render_name is None:
                self.render_name = "HumanoidSNU_Low_DeepMimic_" + str(self.num_envs)
            self.stage = Usd.Stage.CreateNew("outputs/" + self.render_name + ".usd")

            self.renderer = df.UsdRenderer(self.model, self.stage, draw_ground=False)
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
        self.heading_vec = self.z_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.z_unit_tensor.clone()
        self.basis_vec1 = self.y_unit_tensor.clone()

        self.targets = tu.to_torch([0.0, 0.0, 10000.0], device=self.device, requires_grad=False).repeat(
            (self.num_envs, 1))

        if self.visualize:
            self.env_dist = 2.0
        else:
            self.env_dist = 0.  # set to zero for training for numerical consistency

        # start_height = 1.0

        self.asset_folder = os.path.join(os.path.dirname(__file__), 'assets/snu')
        asset_path = os.path.join(self.asset_folder, "human.xml")
        muscle_path = os.path.join(self.asset_folder, "muscle284.xml")

        for _ in range(self.num_environments):
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
            lu.Skeleton(asset_path, None, self.reference_builder, self.filter)

        num_q = int(len(self.builder.joint_q) / self.num_environments)
        num_qd = int(len(self.builder.joint_qd) / self.num_environments)
        assert num_q == self.num_joint_q, f"num_q does not match self.num_joint_q: num_q: {num_q}, self.num_joint_q: {self.num_joint_q}"
        assert num_qd == self.num_joint_qd, f"num_qd does not match self.num_joint_qd: num_qd: {num_qd}, self.num_joint_qd: {self.num_joint_qd}"

        num_muscles = len(self.skeletons[0].muscles)
        print("Num muscles: ", num_muscles)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)
        self.model.friction_smoothing = 0.25

        # load reference motion
        self.reference_frame_time, self.reference_frame_count, self.reference_joint_q, self.reference_joint_q_mask, self.reference_joint_qd = \
            lu.load_bvh(os.path.join(self.asset_folder, "motion/walk.bvh"), self.skeletons[0].bvh_map, self.model, self.dt)

        # end effector indices
        self.end_effector_indices = [4, 9] # FootThumbR, FootThumbL
        self.end_effector_indices = tu.to_torch(self.end_effector_indices, device=self.device, dtype=torch.long)

        # reference model does not use simulation
        self.reference_model = self.reference_builder.finalize(self.device)

        self.integrator = df.SemiImplicitIntegrator()

        self.state = self.model.state()
        self.reference_state = self.reference_model.state()
        self.reference_frame = torch.zeros((self.num_envs), dtype=torch.long, device=self.device)

        # move ref pos to the initial pos
        start_height = self.reference_joint_q[0, 1] - 0.10
        self.start_pos = torch.tensor((0.0, start_height, 0.0), dtype=torch.float32, device=self.device)
        self.reference_pos_offset = self.start_pos.unsqueeze(0).repeat(self.num_envs, 1) - self.reference_joint_q[0, 0:3]
        self.reference_pos_offset[:, 1] += 0.0
        self.start_reference_pos_offset = self.reference_pos_offset.clone()

        if (self.model.ground):
            self.model.collide(self.state)

    def render(self, mode='human', render_env_ids=None):
        """
        Render all frames in the render buffer for specified environments.
        
        Args:
            mode: Render mode (unused)
            render_env_ids: List of environment IDs to render. If None, renders only the active environment.
        """
        render_asset_folder = self.asset_folder
        
        if len(self.render_buffer) == 0:
            return
            
        if render_env_ids is None:
            # Default to the active environment
            render_env_ids = torch.arange(self.num_envs, device=self.device)
        # Reset render time
        render_time = 0.0
        
        with torch.no_grad():
            from tqdm import tqdm
            for frame in tqdm(self.render_buffer, desc="Rendering", leave=False):
                # Extract state and observation data from the buffer
                state = frame['state']
                obs = frame['obs'][render_env_ids]  # Only get observations for selected environments
                reference_state = frame['reference_state']
                muscle_activation = frame['muscle_activation']
                
                # Render this frame for selected environments
                muscle_start = 0
                skel_index = 0

                for i in render_env_ids:
                    s = self.skeletons[i]
                    for mesh, link in s.mesh_map.items():
                        if link != -1:
                            # We need to extract the right transforms for each link
                            # This depends on how the body_X_sc is organized in the state
                            link_X_sc = state.body_X_sc.view(self.num_envs, -1, 7)[i, link]
                            X_sc = df.transform_expand(link_X_sc.tolist())
                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")
                            self.renderer.add_mesh(mesh, mesh_path, X_sc, 1.0, render_time)

                    for m in range(len(s.muscles)):
                        start = self.model.muscle_start[muscle_start + m].item()
                        end = self.model.muscle_start[muscle_start + m + 1].item()
                        points = []

                        for w in range(start, end):
                            link = self.model.muscle_links[w].item()
                            point = self.model.muscle_points[w].cpu().numpy()
                            
                            # Get the transform for this specific environment and link
                            link_X_sc = state.body_X_sc.view(self.num_envs, -1, 7)[i, link]
                            X_sc = df.transform_expand(link_X_sc.cpu().tolist())
                            points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))

                        # Get muscle activation for this environment
                        env_muscle_activation = muscle_activation.view(self.num_envs, -1)[i, muscle_start + m]
                        
                        self.renderer.add_line_strip(points, name=s.muscles[m].name + str(skel_index),
                                                        radius=0.0075, color=(
                                env_muscle_activation.item(), 0.2,
                                0.5),
                                                        time=render_time)

                    muscle_start += len(s.muscles)
                    skel_index += 1

                # add the observations and reference poses
                # relative reference pos
                ref_relative_body_X_sc = reference_state.body_X_sc.view(self.num_envs, -1, 7).clone()

                for i in render_env_ids:
                    s = self.skeletons[i]
                    for mesh, link in s.mesh_map.items():
                        if link != -1:
                            link_transform = ref_relative_body_X_sc[i, link]
                            link_transform[0] += 2
                            X_sc = df.transform_expand(link_transform.tolist())
                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")
                            self.renderer.add_mesh(f'{mesh}_relative_refpos', mesh_path, X_sc, 1.0, render_time)
                
                # relative
                relative_body_X_sc = state.body_X_sc.view(self.num_envs, -1, 7).clone()
                for i in render_env_ids:
                    s = self.skeletons[i]
                    for mesh, link in s.mesh_map.items():
                        if link != -1:
                            link_transform = relative_body_X_sc[i, link]
                            link_transform[0] += 2
                            X_sc = df.transform_expand(link_transform.tolist())
                            mesh_path = os.path.join(render_asset_folder, "OBJ/" + mesh + ".usd")
                            self.renderer.add_mesh(f'{mesh}_relative', mesh_path, X_sc, 1.0, render_time)

                # Phase: add a sphere to move [10, phase, 10]
                for i in render_env_ids:
                    self.renderer.add_sphere((10, obs[i, -1], 10), 0.1, "phase", render_time)

                # com_pos: add sphere
                com_pos = tu.get_center_of_mass(
                    self.model.body_I_m.view(self.num_envs, -1, 6, 6),
                    state.body_X_sm.view(self.num_envs, -1, 7)
                ).view(-1, 3)
                for i in render_env_ids:
                    com_pos_i = com_pos[i, :]
                    com_pos_i[0] += 2
                    self.renderer.add_sphere(com_pos_i.tolist(), 0.1, "com", render_time)

                # reference com pos: add sphere
                ref_com_pos = tu.get_center_of_mass(
                    self.model.body_I_m.view(self.num_envs, -1, 6, 6),
                    reference_state.body_X_sm.view(self.num_envs, -1, 7)
                ).view(-1, 3)
                for i in render_env_ids:
                    ref_com_pos_i = ref_com_pos[i, :]
                    ref_com_pos_i[0] += 2
                    self.renderer.add_sphere(ref_com_pos_i.tolist(), 0.1, "ref_com", render_time)

                # Update with the state view for this environment
                self.renderer.update(state, render_time)
                render_time += self.dt * self.inv_control_freq

    def save_render(self):
        """Save the USD stage to a file."""
        try:
            self.stage.Save()
            print(f"Saved to outputs/{self.render_name}.usd")
        except Exception as e:
            print(f"USD save error: {e}")

    def store_frame_in_buffer(self):
        """Store current frame in render buffer for all environments"""
        if len(self.render_buffer) >= self.render_buffer_max_size:
            # If buffer is full, remove oldest frame
            self.render_buffer.pop(0)
        
        # Store a deep copy of the current state and observation
        frame = {
            'state': self.model.state(),
            'obs': self.obs_buf.clone(),
            'reference_state': self.reference_model.state(),
            'muscle_activation': self.model.muscle_activation.clone()
        }
        
        # Copy values from current state to saved state
        frame['state'].joint_q = self.state.joint_q.clone()
        frame['state'].joint_qd = self.state.joint_qd.clone()
        frame['state'].body_X_sc = self.state.body_X_sc.clone()
        frame['state'].body_X_sm = self.state.body_X_sm.clone()
        frame['state'].body_v_s = self.state.body_v_s.clone()
        
        # Copy values from current reference state
        frame['reference_state'].joint_q = self.reference_state.joint_q.clone()
        frame['reference_state'].joint_qd = self.reference_state.joint_qd.clone()
        frame['reference_state'].body_X_sc = self.reference_state.body_X_sc.clone()
        frame['reference_state'].body_X_sm = self.reference_state.body_X_sm.clone()
        frame['reference_state'].body_v_s = self.reference_state.body_v_s.clone()
        
        self.render_buffer.append(frame)

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

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
                    # TODO: too verbose; find a better way to handle this
                    # print(f'{name} grad nan count: {nan_count}, inf count: {inf_count}, big count: {big_count}.')
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
            self.model.muscle_activation = actions.view(-1)
            # self.model.muscle_activation = torch.zeros_like(self.model.muscle_activation)

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

        # Store current frame in render buffer
        if self.visualize:
            self.store_frame_in_buffer()

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
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        """
        Reset the specified environments and optionally render specific environments.
        
        Args:
            env_ids: Environment IDs to reset
            force_reset: Whether to force reset
            render_env_ids: Environment IDs to render. If None, renders the active environment.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # If this is a reset and we have visualization enabled, render the buffer
            if self.visualize and len(self.render_buffer) > 0:
                with df.ScopedTimer("render", False):
                    self.render(env_ids)
                    self.save_render()
            
            # Clear the render buffer after rendering
            self.render_buffer = []
            
            # copy the reference motion to the state
            # randomization
            if self.stochastic_init:
                self.progress_buf[env_ids] = 0
                # randomize the reset start frame to learn all reference frames uniformly
                # self.offset_buf[env_ids] = torch.floor(torch.rand(len(env_ids), device=self.device) * (self.reference_frame_count * self.reference_frame_time / self.dt - 1)).long()
                self.offset_buf[env_ids] = 0
                self.start_frame_offset = 0
                self.reference_frame[env_ids] = 0
                self.reference_pos_offset[env_ids] = self.start_reference_pos_offset[env_ids].clone()
                self.copy_ref_pos_to_state(env_ids, perform_forward_kinematics=True)
                with torch.no_grad():
                    self.update_reference_model()
                    # start pos randomization
                    # self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] += (torch.rand(
                    #     size=(len(env_ids), 3), device=self.device) - 0.5) * torch.tensor([0.05, 0.0, 0.05], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
                    # # start rot randomization
                    # angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 36.0
                    # axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                    # self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(
                    #     self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                    # # start vel randomization
                    # self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] += 0.05 * (
                    #             torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)
            else:
                self.progress_buf[env_ids] = 0
                self.offset_buf[env_ids] = 0
                self.start_frame_offset = 0
                self.reference_frame[env_ids] = 0
                self.reference_pos_offset[env_ids] = self.start_reference_pos_offset[env_ids].clone()
                self.copy_ref_pos_to_state(env_ids, perform_forward_kinematics=True)
                with torch.no_grad():
                    self.update_reference_model()

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
                checkpoint = self.get_checkpoint()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.state.body_X_sc = checkpoint['body_X_sc'].clone()
            self.state.body_X_sm = checkpoint['body_X_sm'].clone()
            self.state.body_v_s = checkpoint['body_v_s'].clone()
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()
            self.obs_buf = checkpoint['obs_buf'].clone()
            self.reference_frame = checkpoint['reference_frame'].clone()
            self.reference_pos_offset = checkpoint['reference_pos_offset'].clone()

    def initialize_trajectory(self):
        """
        This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
        It has to be called every time the algorithm starts an episode and it returns the observation vectors
        """
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        """
        Get a checkpoint of the current state.
        """
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['body_X_sc'] = self.state.body_X_sc.clone()
        checkpoint['body_X_sm'] = self.state.body_X_sm.clone()
        checkpoint['body_v_s'] = self.state.body_v_s.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()
        checkpoint['obs_buf'] = self.obs_buf.clone()
        checkpoint['reference_frame'] = self.reference_frame.clone()
        checkpoint['reference_pos_offset'] = self.reference_pos_offset.clone()

        return checkpoint

    def calculateObservations(self):
        # forward kinematics
        # body_X_sc, body_X_sm = eval_rigid_fk_grad(self.model, self.state.joint_q)
        # joint_S_s, body_I_s, body_v_s, body_f_s, body_a_s = eval_rigid_id_grad(self.model, self.state.joint_q, self.state.joint_qd, body_X_sc, body_X_sm)

        # add root pos, root rot, and torso pos.
        root_transform = self.state.body_X_sc.view(self.num_envs, -1, 7)[:, 0, :].squeeze(1)
        root_pos = root_transform[:, 0:3]
        root_rot = root_transform[:, 3:7]

        root_pos_xz = root_pos.clone()
        root_pos_xz[:, 1] = 0.0

        # DeepMimic states (observations): each link's position and rotation relative to the root joint, phase of the motion
        # 1. compute relative body X_sc
        body_X_sc = self.state.body_X_sc.view(self.num_envs, -1, 7)
        body_X_sc_local = body_X_sc.clone()
        body_X_sc_local[:, :, 0:3] = body_X_sc[:, :, 0:3] - root_pos_xz.unsqueeze(1).repeat(1, body_X_sc.shape[1], 1)
    
        # 2. compute phase of the motion
        progress_time = torch.clamp((self.progress_buf + self.offset_buf) * self.dt, min=0.0)
        max_time = self.reference_frame_time * self.reference_frame_count
        phase = ((progress_time + self.start_frame_offset * self.dt) / max_time) % 1.0

        # 3. calculate the center of mass position
        com_pos = tu.get_center_of_mass(self.model.body_I_m.view(self.num_envs, -1, 6, 6), self.state.body_X_sm.view(self.num_envs, -1, 7))
        com_pos_local = com_pos - root_pos_xz

        # calculate distance to reference com pos
        ref_com_pos = tu.get_center_of_mass(self.model.body_I_m.view(self.num_envs, -1, 6, 6), self.reference_state.body_X_sm.view(self.num_envs, -1, 7))
        com_pos_diff = com_pos - ref_com_pos
        
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
            root_pos[:, 1:2].view(self.num_envs, -1), # 0:1
            root_rot.view(self.num_envs, -1), # 1:5
            com_pos_local.view(self.num_envs, -1), # 5:8
            lin_vel.view(self.num_envs, -1), # 8:11
            ang_vel.view(self.num_envs, -1), # 11:14
            up_vec[:, 1:2],  # 14:15
            (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1), # 15:16
            com_pos_diff.view(self.num_envs, -1), # 16:19
            body_X_sc_local.view(self.num_envs, -1), # 19:96
            self.joint_vel_obs_scaling * self.state.body_v_s.view(self.num_envs, -1), # 96:162
            phase.view(self.num_envs, 1), # 162
        ], dim=-1)

    root_height_range = range(0, 1)
    root_rot_range = range(1, 5)
    com_pos_range = range(5, 8)
    lin_vel_range = range(8, 11)
    ang_vel_range = range(11, 14)
    up_vec_range = range(14, 15)
    heading_vec_range = range(15, 16)
    com_pos_diff_range = range(16, 19)
    body_X_sc_local_range = range(19, 96)
    joint_vel_range = range(96, 162)
    phase_range = range(162, 163)

    def calculateReward(self):
        # DeepMimic reward: pose reward + velocity reward + end-effector reward + center-of-mass reward
        w_p = 0.65
        w_v = 0.1
        w_e = 0.15
        w_c = 0.1

        # body transforms and velocities
        body_X_sc = self.state.body_X_sc.view(self.num_envs, -1, 7)
        body_v_s = self.state.body_v_s.view(self.num_envs, -1, 6)
        ref_body_X_sc = self.reference_state.body_X_sc.view(self.num_envs, -1, 7)
        ref_body_v_s = self.reference_state.body_v_s.view(self.num_envs, -1, 6)

        # pos reward: exp(-2 * sum(body quat, ref body quat diff **2))
        body_quat = body_X_sc[:, :, 3:7]
        ref_body_quat = ref_body_X_sc[:, :, 3:7]
        body_quat_diff = tu.quat_diff_chordal(body_quat, ref_body_quat)
        # pos_reward = torch.exp(-2 * torch.sum(torch.sum(body_quat_diff ** 2, dim=-1), dim=-1))
        rot_reward = -2 * torch.sum(torch.sum(body_quat_diff ** 2, dim=-1), dim=-1)
        
        body_pos = body_X_sc[:, :, 0:3]
        ref_body_pos = ref_body_X_sc[:, :, 0:3]
        body_pos_diff = body_pos - ref_body_pos
        pos_reward = -1.5 * torch.sum(torch.sum(body_pos_diff ** 2, dim=-1), dim=-1)

        # velocity reward: exp(-0.1 * sum(body w, ref body w diff **2))
        # body_w_diff = body_v_s[:, :, 0:3] - ref_body_v_s[:, :, 0:3]
        # vel_reward = torch.exp(-0.1 * torch.sum(torch.sum(body_w_diff ** 2, dim=-1), dim=-1))

        # end-effector reward: exp(-40 * sum(end-effector pos, ref end-effector pos diff **2))
        # let's compare end-effector pos in local frame
        end_effector_pos = body_X_sc[:, self.end_effector_indices, 0:3]
        ref_end_effector_pos = ref_body_X_sc[:, self.end_effector_indices, 0:3]
        end_effector_pos_diff = end_effector_pos - ref_end_effector_pos
        # strong penalty for xz-plane diff
        end_effector_pos_diff[:, :, 0] *= 2.5
        end_effector_pos_diff[:, :, 2] *= 2.5
        # end_effector_reward = torch.exp(-5 * torch.sum(torch.sum(end_effector_pos_diff ** 2, dim=-1), dim=-1))
        end_effector_reward = -5 * torch.sum(torch.sum(end_effector_pos_diff ** 2, dim=-1), dim=-1)

        # center-of-mass reward: exp(-10 * sum(com pos, ref com pos diff **2))
        # com_pos = self.obs_buf[:, self.com_pos_range]
        # ref_com_pos = tu.get_center_of_mass(self.model.body_I_m.view(self.num_envs, -1, 6, 6), self.reference_state.body_X_sm.view(self.num_envs, -1, 7))
        com_pos_diff = self.obs_buf[:, self.com_pos_diff_range]
        # strong penalty for xz-plane diff
        com_pos_diff[:, 0] *= 2.5
        com_pos_diff[:, 2] *= 2.5
        # com_reward = torch.exp(-10 * torch.sum(com_pos_diff ** 2, dim=-1))
        com_reward = -10 * torch.sum(com_pos_diff ** 2, dim=-1)

        # imitation_reward = w_p * pos_reward + w_v * vel_reward + w_e * end_effector_reward + w_c * com_reward
        # instead, use multiplied reward
        imitation_reward = rot_reward + end_effector_reward + com_reward + 1.0
        # live_reward = torch.ones_like(imitation_reward) * 0.3

        # goal reward
        # up_reward = 0.1 * self.obs_buf[:, 17]
        # heading_reward = 1 * self.obs_buf[:, 18]

        # height_diff = self.obs_buf[:, 1] - self.termination_height
        # height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        # height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward,
        #                             height_reward)  # JIE: not smooth
        # height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        # act_penalty = torch.sum(torch.abs(self.actions),
        #                         dim=-1) * self.action_penalty  # torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # # get z-axis velocity
        # progress_reward = self.obs_buf[:, 12]
        # # walking: speed is limited to 0.75m/s, so clip the reward
        # progress_reward = torch.where(progress_reward > 0.75, 0.75, progress_reward)

        # goal_reward = height_reward + progress_reward

        # w_g = 0.05
        # w_i = 0.95

        # print the mean reward
        # print(f"mean imitation reward: {torch.mean(imitation_reward).item()}, mean goal reward: {torch.mean(goal_reward).item()}")

        self.rew_buf = imitation_reward

        # reset agents (early termination)
        self.reset_buf = torch.where(self.obs_buf[:, self.root_height_range].squeeze(-1) < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # # if pos reward is less than -1, reset
        # body_pos_diff = relative_body_X_sc[:, :, 0:3] - ref_relative_body_X_sc[:, :, 0:3]
        # self.reset_buf = torch.where(torch.mean(torch.sum(body_pos_diff ** 2, dim=-1), dim=-1) > 0.2, torch.ones_like(self.reset_buf),
        #                                 self.reset_buf)
        # if imitation reward is less than 0.3, reset
        self.reset_buf = torch.where(imitation_reward < -1.3, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # normal termination
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

    def get_frame_indices(self):
        return torch.round((torch.clamp(self.progress_buf + self.offset_buf, min=0) + self.start_frame_offset) * self.dt / self.reference_frame_time).long() % self.reference_frame_count

    def update_reference_model(self):
        """
        Update the reference model's state from the phase of the motion.
        """
        frame_indices = self.get_frame_indices()
        next_state = self.reference_model.state()

        # if new frame index is less than the previous frame index, it means the reference model has looped
        loop_indices = torch.nonzero(frame_indices < self.reference_frame).squeeze(-1)
        # if looped, update the offset to connect the loop
        diff = self.reference_joint_q[self.reference_frame[loop_indices], :3] - self.reference_joint_q[frame_indices[loop_indices], :3]
        # prevent height error
        diff[:, 1] = 0.0
        self.reference_pos_offset[loop_indices] += diff
        self.reference_frame = frame_indices

        # update joint_q and joint_qd
        next_state.joint_q.view(self.num_envs, -1)[:, :] = self.reference_joint_q[frame_indices, :]
        next_state.joint_qd.view(self.num_envs, -1)[:, :] = self.reference_joint_qd[frame_indices, :]
        
        # apply the offset to the reference model
        next_state.joint_q.view(self.num_envs, -1)[:, :3] += self.reference_pos_offset

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

    def copy_ref_pos_to_state(self, env_ids=None, perform_forward_kinematics=False):
        """
        Copy the reference motion to the state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.state.joint_q = self.state.joint_q.clone()
        self.state.joint_qd = self.state.joint_qd.clone()

        frame_index = self.get_frame_indices()
        # masked values are 0, so we don't need to use mask to select valid values
        self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = self.reference_joint_q[frame_index[env_ids], :]
        self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = self.reference_joint_qd[frame_index[env_ids], :]

        # reset position
        self.state.joint_q.view(self.num_envs, -1)[env_ids, :3] += self.start_reference_pos_offset[env_ids]

        if perform_forward_kinematics:
            body_X_sc, body_X_sm = eval_rigid_fk_grad(self.model, self.state.joint_q)
            self.state.body_X_sc = body_X_sc
            self.state.body_X_sm = body_X_sm
            joint_S_s, body_I_s, body_v_s, body_f_s, body_a_s = eval_rigid_id_grad(self.model, self.state.joint_q, self.state.joint_qd, body_X_sc, body_X_sm)
            self.state.joint_S_s = joint_S_s
            self.state.body_I_s = body_I_s
            self.state.body_v_s = body_v_s
            self.state.body_f_s = body_f_s
            self.state.body_a_s = body_a_s
