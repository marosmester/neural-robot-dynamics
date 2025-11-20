"""
Simple script to simulate PendulumWithContactEnvironment using ground-truth Featherstone solver.
- Uses classical physics integrator (Featherstone)
- Initializes environment with visualization
- Sets non-trivial initial conditions
- Runs simulation with rendering
"""
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import torch
import numpy as np

from envs.neural_environment import NeuralEnvironment
from utils.python_utils import set_random_seed

device = 'cuda:0'
num_envs = 1
num_steps = 1000
seed = 42

set_random_seed(seed)

# Initialize environment with visualization
# Note: We don't load a neural model, so it will use the ground-truth Featherstone integrator
print("Initializing PendulumWithContactEnvironment with ground-truth Featherstone solver...")
env_cfg = {
    "env_name": "PendulumWithContact",
    "num_envs": num_envs,
    "render": True,  # Enable visualization
    "warp_env_cfg": {
        "seed": seed
    },
    "neural_integrator_cfg": None,  # No neural integrator config needed
    "neural_model": None,  # No neural model - use ground-truth physics
    "default_env_mode": "ground-truth",  # Use Featherstone integrator
    "device": device
}

neural_env = NeuralEnvironment(**env_cfg)

print(f"Environment initialized. State dimension: {neural_env.state_dim}")
print(f"Action dimension: {neural_env.action_dim}")
print(f"Using integrator: {neural_env.env.integrator_type}")

# Set initial states (same as NeRD version)
# For Pendulum: state = [θ1, θ2, θ̇1, θ̇2] (positions + velocities)
print("\nSetting initial conditions...")
initial_states = torch.zeros((num_envs, neural_env.state_dim), device=device)
initial_states[0, 0] = np.deg2rad(45.0)   # θ1 = 45°
initial_states[0, 1] = np.deg2rad(-30.0)  # θ2 = -30°
initial_states[0, 2] = 1.0   # θ̇1 = 1.0 rad/s
initial_states[0, 3] = -0.5  # θ̇2 = -0.5 rad/s

print(f"Initial state:")
print(f"  θ1 = {np.rad2deg(initial_states[0, 0].item()):.2f}°")
print(f"  θ2 = {np.rad2deg(initial_states[0, 1].item()):.2f}°")
print(f"  θ̇1 = {initial_states[0, 2].item():.2f} rad/s")
print(f"  θ̇2 = {initial_states[0, 3].item():.2f} rad/s")

# Reset environment with initial states
neural_env.reset(initial_states=initial_states)

# For passive motion, we use zero actions (no control input)
zero_actions = torch.zeros((num_envs, neural_env.action_dim), device=device)

print(f"\nRunning simulation for {num_steps} steps using Featherstone integrator...")
print("Close the visualization window to stop the simulation.")

try:
    for step in range(num_steps):
        # Step forward with zero actions (passive motion) using ground-truth physics
        states = neural_env.step(zero_actions, env_mode='ground-truth')
        
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

