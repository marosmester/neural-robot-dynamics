# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to test NeRD model with PendulumWithContactEnvironment.
- Loads pretrained NeRD model
- Creates custom double pendulum ModelBuilder
- Initializes environment with visualization using custom ModelBuilder
- Sets non-trivial initial conditions
- Runs simulation with rendering
"""
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import torch
import yaml
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from envs.neural_environment import NeuralEnvironment
from utils.torch_utils import num_params_torch_model
from utils.python_utils import set_random_seed
from envs.warp_sim_envs.utils import update_ground_plane

if __name__ == '__main__':
    # Configuration
    device = 'cuda:0'
    model_path = os.path.join(base_dir, 'pretrained_models/NeRD_models/Pendulum/model/nn/model.pt')
    num_envs = 1
    num_steps = 5000
    seed = 42
    
    set_random_seed(seed)
    
    # Load pretrained NeRD model
    print("Loading pretrained NeRD model...")
    neural_model, robot_name = torch.load(model_path, map_location=device, weights_only=False)
    print(f'Number of Model Parameters: {num_params_torch_model(neural_model)}')
    neural_model.to(device)
    
    # Load model configuration
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(model_path)), '../'))
    cfg_path = os.path.join(train_dir, 'cfg.yaml')
    print(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]
    
    # Create custom double pendulum ModelBuilder
    print("Creating custom double pendulum ModelBuilder...")
    def create_custom_pendulum_builder():
        """
        Creates a custom double pendulum ModelBuilder.
        This is a custom implementation that can be modified to change
        the pendulum structure (chain length, link dimensions, etc.)
        """
        # Custom pendulum parameters
        chain_length = 2
        chain_width = 1.5  # Link length
        link_radius = 0.1
        link_density = 500.0
        
        # Joint limits
        joint_limit_lower = -2 * np.pi
        joint_limit_upper = 2 * np.pi
        limit_ke = 0.0
        limit_kd = 0.0
        
        # Contact properties
        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4
        
        # Create ModelBuilder with up_vector and gravity
        # Note: up_vector and gravity will be set by Environment, but we set them here for consistency
        up_vector = np.array([0.0, 1.0, 0.0])  # Y-axis up
        gravity = -9.81
        articulation_builder = wp.sim.ModelBuilder(up_vector=up_vector, gravity=gravity)
        
        # Set ground plane
        articulation_builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )
        
        # Create pendulum links
        for i in range(chain_length):
            if i == 0:
                # First link: connected to world
                parent = -1
                parent_joint_xform = wp.transform([0.0, 2.0, 1.0], wp.quat_identity())
            else:
                # Subsequent links: connected to previous link
                parent = articulation_builder.joint_count - 1
                parent_joint_xform = wp.transform(
                    [chain_width, 0.0, 0.0], wp.quat_identity()
                )
            
            # Create body
            body = articulation_builder.add_body(
                origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()),
                armature=0.1
            )
            
            # Create capsule shape for the link
            articulation_builder.add_shape_capsule(
                pos=(chain_width * 0.5, 0.0, 0.0),
                half_height=chain_width * 0.5,
                radius=link_radius,
                up_axis=0,  # X-axis aligned
                density=link_density,
                body=body,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
            )
            
            # Create revolute joint
            articulation_builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=(0.0, 0.0, 1.0),  # Rotate around Z-axis
                parent_xform=parent_joint_xform,
                limit_lower=joint_limit_lower,
                limit_upper=joint_limit_upper,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
            )
        
        # Set initial joint positions
        articulation_builder.joint_q[:] = [0.0, 0.0]
        
        # Configure ground plane for contact (using contact config 0: contact-free)
        # This can be modified to enable different contact configurations
        ground_offset = -15.5  # Contact-free configuration
        ground_rot_xyz = np.array([0., 0., 0.])
        ground_rot = Rotation.from_euler('xyz', ground_rot_xyz).as_quat()
        update_ground_plane(
            articulation_builder,
            pos=[0.0, ground_offset, 0.0],
            rot=ground_rot,
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )
        
        return articulation_builder
    
    # Create the custom articulation builder
    custom_articulation_builder = create_custom_pendulum_builder()
    print(f"Custom pendulum created: {custom_articulation_builder.body_count} bodies, "
          f"{custom_articulation_builder.joint_count} joints")
    
    # Initialize environment with visualization and custom ModelBuilder
    print("Initializing PendulumWithContactEnvironment with custom ModelBuilder...")
    env_cfg = {
        "env_name": "PendulumWithContact",
        "num_envs": num_envs,
        "render": True,  # Enable visualization
        "warp_env_cfg": {
            "seed": seed
        },
        "neural_integrator_cfg": neural_integrator_cfg,
        "neural_model": neural_model,
        "default_env_mode": "neural",  # Use NeRD model
        "device": device,
        "custom_articulation_builder": custom_articulation_builder  # Pass custom builder
    }
    
    neural_env = NeuralEnvironment(**env_cfg)
    
    assert neural_env.robot_name == robot_name, \
        "neural_env.robot_name is not equal to neural_model's robot_name."
    
    print(f"Environment initialized. State dimension: {neural_env.state_dim}")
    print(f"Action dimension: {neural_env.action_dim}")
    
    # Set non-trivial initial conditions
    # For Pendulum: state = [θ1, θ2, θ̇1, θ̇2] (positions + velocities)
    # State dimension = dof_q_per_env + dof_qd_per_env = 2 + 2 = 4
    print("\nSetting non-trivial initial conditions...")
    initial_states = torch.zeros((num_envs, neural_env.state_dim), device=device)
    
    # Set initial joint angles (in radians)
    # First joint: 45 degrees, Second joint: -30 degrees
    initial_states[0, 0] = np.deg2rad(-90)   # θ1 = 45°
    initial_states[0, 1] = np.deg2rad(-90)  # θ2 = -30°
    
    # Set initial joint velocities (in rad/s)
    # Give some initial angular velocities for more interesting motion
    initial_states[0, 2] = 1.0   # θ̇1 = 1.0 rad/s
    initial_states[0, 3] = -0.5  # θ̇2 = -0.5 rad/s
    
    print(f"Initial state:")
    print(f"  θ1 = {np.rad2deg(initial_states[0, 0].item()):.2f}°")
    print(f"  θ2 = {np.rad2deg(initial_states[0, 1].item()):.2f}°")
    print(f"  θ̇1 = {initial_states[0, 2].item():.2f} rad/s")
    print(f"  θ̇2 = {initial_states[0, 3].item():.2f} rad/s")
    
    # Reset environment with initial states
    neural_env.reset(initial_states=initial_states)
    
    # Run simulation loop
    print(f"\nRunning simulation for {num_steps} steps...")
    print("Close the visualization window to stop the simulation.")
    
    # For passive motion, we use zero actions (no control input)
    zero_actions = torch.zeros((num_envs, neural_env.action_dim), device=device)
    
    try:
        for step in range(num_steps):
            # Step forward with zero actions (passive motion)
            states = neural_env.step(zero_actions, env_mode='neural')
            
            # Render the simulation
            neural_env.render()
            
            # Print state every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: θ1 = {np.rad2deg(states[0, 0].item()):.2f}°, "
                      f"θ2 = {np.rad2deg(states[0, 1].item()):.2f}°, "
                      f"θ̇1 = {states[0, 2].item():.2f} rad/s, "
                      f"θ̇2 = {states[0, 3].item():.2f} rad/s")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    print("Simulation completed.")
    neural_env.close()

