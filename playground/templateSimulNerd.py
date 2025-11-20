
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import torch
import yaml
import numpy as np

from envs.neural_environment import NeuralEnvironment
from utils.python_utils import set_random_seed

device = 'cuda:0'
model_path = os.path.join(base_dir, 'pretrained_models/NeRD_models/Pendulum/model/nn/model.pt')
num_envs = 1
num_steps = 1000
seed = 42

set_random_seed(seed)

# Load pretrained NeRD model
neural_model, robot_name = torch.load(model_path, map_location=device, weights_only=False)
neural_model.to(device)

# Load model configuration
train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(model_path)), '../'))
cfg_path = os.path.join(train_dir, 'cfg.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]

# Initialize environment with visualization
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
    "device": device
}

neural_env = NeuralEnvironment(**env_cfg)
assert neural_env.robot_name == robot_name, \
    "neural_env.robot_name is not equal to neural_model's robot_name."

# initial states
initial_states = torch.zeros((num_envs, neural_env.state_dim), device=device)
initial_states[0, 0] = np.deg2rad(45.0)   # θ1 = 45°
initial_states[0, 1] = np.deg2rad(-30.0)  # θ2 = -30°
initial_states[0, 2] = 1.0   # θ̇1 = 1.0 rad/s
initial_states[0, 3] = -0.5  # θ̇2 = -0.5 rad/s

# Reset environment with initial states
neural_env.reset(initial_states=initial_states)

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