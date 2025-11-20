"""
Compare NeRD model vs Featherstone solver for Pendulum simulation.
- Runs the same simulation twice: once with Featherstone, once with NeRD
- Records state trajectories from both
- Plots the Euclidean norm of state difference over time
"""
import sys
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from envs.neural_environment import NeuralEnvironment
from utils.python_utils import set_random_seed

device = 'cuda:0'
model_path = os.path.join(base_dir, 'pretrained_models/NeRD_models/Pendulum/model/nn/model.pt')
num_envs = 1
num_steps = 1000
seed = 42

set_random_seed(seed)

# Load pretrained NeRD model
print("Loading pretrained NeRD model...")
neural_model, robot_name = torch.load(model_path, map_location=device, weights_only=False)
neural_model.to(device)

# Load model configuration
train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(model_path)), '../'))
cfg_path = os.path.join(train_dir, 'cfg.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]

# Initialize environment (without rendering for faster computation)
print("Initializing PendulumWithContactEnvironment...")
env_cfg = {
    "env_name": "PendulumWithContact",
    "num_envs": num_envs,
    "render": False,  # Disable rendering for faster computation
    "warp_env_cfg": {
        "seed": seed
    },
    "neural_integrator_cfg": neural_integrator_cfg,
    "neural_model": neural_model,
    "default_env_mode": "ground-truth",  # Start with ground-truth, we'll switch modes
    "device": device
}

neural_env = NeuralEnvironment(**env_cfg)
assert neural_env.robot_name == robot_name, \
    "neural_env.robot_name is not equal to neural_model's robot_name."

print(f"Environment initialized. State dimension: {neural_env.state_dim}")
print(f"Action dimension: {neural_env.action_dim}")

# Set initial states (same for both simulations)
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

# Zero actions for passive motion
zero_actions = torch.zeros((num_envs, neural_env.action_dim), device=device)

# Storage for state trajectories
states_gt = []  # Ground-truth (Featherstone) states
states_neural = []  # NeRD states

# ===== Run simulation with Featherstone (ground-truth) =====
print(f"\n{'='*60}")
print("Running simulation with Featherstone solver (ground-truth)...")
print(f"{'='*60}")

neural_env.reset(initial_states=initial_states.clone())
neural_env.integrator_neural.reset()  # Reset neural integrator state history

for step in range(num_steps):
    states = neural_env.step(zero_actions, env_mode='ground-truth')
    states_gt.append(states[0].clone().cpu().numpy())  # Store state for env 0
    
    if step % 100 == 0:
        print(f"Step {step}: θ1 = {np.rad2deg(states[0, 0].item()):.2f}°, "
              f"θ2 = {np.rad2deg(states[0, 1].item()):.2f}°")

states_gt = np.array(states_gt)  # Shape: (num_steps, state_dim)

# ===== Run simulation with NeRD model =====
print(f"\n{'='*60}")
print("Running simulation with NeRD model...")
print(f"{'='*60}")

neural_env.reset(initial_states=initial_states.clone())
neural_env.integrator_neural.reset()  # Reset neural integrator state history

for step in range(num_steps):
    states = neural_env.step(zero_actions, env_mode='neural')
    states_neural.append(states[0].clone().cpu().numpy())  # Store state for env 0
    
    if step % 100 == 0:
        print(f"Step {step}: θ1 = {np.rad2deg(states[0, 0].item()):.2f}°, "
              f"θ2 = {np.rad2deg(states[0, 1].item()):.2f}°")

states_neural = np.array(states_neural)  # Shape: (num_steps, state_dim)

# ===== Compute state differences =====
print(f"\n{'='*60}")
print("Computing state differences...")
print(f"{'='*60}")

# Compute Euclidean norm of state difference at each step
state_diffs = states_neural - states_gt  # Shape: (num_steps, state_dim)
state_diff_norms = np.linalg.norm(state_diffs, axis=1)  # Shape: (num_steps,)

# Also compute per-component differences for more detailed analysis
diff_theta1 = np.abs(state_diffs[:, 0])  # Difference in θ1
diff_theta2 = np.abs(state_diffs[:, 1])  # Difference in θ2
diff_theta1_dot = np.abs(state_diffs[:, 2])  # Difference in θ̇1
diff_theta2_dot = np.abs(state_diffs[:, 3])  # Difference in θ̇2

# ===== Create visualization =====
print("Creating visualization...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Euclidean norm of state difference
steps = np.arange(num_steps)
axes[0].plot(steps, state_diff_norms, 'b-', linewidth=2, label='Euclidean norm of state difference')
axes[0].set_xlabel('Step', fontsize=12)
axes[0].set_ylabel('State Difference (Euclidean Norm)', fontsize=12)
axes[0].set_title('NeRD vs Featherstone: State Difference Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_yscale('log')  # Use log scale for better visualization if differences are small

# Plot 2: Per-component absolute differences
axes[1].plot(steps, diff_theta1, 'r-', linewidth=1.5, label='|Δθ₁| (rad)', alpha=0.7)
axes[1].plot(steps, diff_theta2, 'g-', linewidth=1.5, label='|Δθ₂| (rad)', alpha=0.7)
axes[1].plot(steps, diff_theta1_dot, 'b-', linewidth=1.5, label='|Δθ̇₁| (rad/s)', alpha=0.7)
axes[1].plot(steps, diff_theta2_dot, 'm-', linewidth=1.5, label='|Δθ̇₂| (rad/s)', alpha=0.7)
axes[1].set_xlabel('Step', fontsize=12)
axes[1].set_ylabel('Absolute State Component Difference', fontsize=12)
axes[1].set_title('Per-Component State Differences', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_yscale('log')

plt.tight_layout()

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), 'pendulum_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Print summary statistics
print(f"\n{'='*60}")
print("Summary Statistics:")
print(f"{'='*60}")
print(f"Mean state difference norm: {np.mean(state_diff_norms):.6f}")
print(f"Max state difference norm: {np.max(state_diff_norms):.6f}")
print(f"Final state difference norm: {state_diff_norms[-1]:.6f}")
print(f"\nMean per-component differences:")
print(f"  |Δθ₁|: {np.mean(diff_theta1):.6f} rad ({np.rad2deg(np.mean(diff_theta1)):.4f}°)")
print(f"  |Δθ₂|: {np.mean(diff_theta2):.6f} rad ({np.rad2deg(np.mean(diff_theta2)):.4f}°)")
print(f"  |Δθ̇₁|: {np.mean(diff_theta1_dot):.6f} rad/s")
print(f"  |Δθ̇₂|: {np.mean(diff_theta2_dot):.6f} rad/s")

# Show the plot
plt.show()

neural_env.close()
print("\nSimulation comparison completed.")

