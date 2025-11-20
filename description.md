# Neural Robot Dynamics (NeRD) - Codebase Directory Structure

This document explains what is contained in each directory of the NeRD codebase, with respect to the paper "Neural Robot Dynamics" (CoRL 2025).

## **`integrators/`** - Core NeRD Integration
This directory contains the core implementation that makes NeRD work as a drop-in replacement for analytical physics integrators in Warp.

- **`integrator_neural.py`**: Base `NeuralIntegrator` class that implements:
  - Robot-centric, spatially-invariant state representation (body frame)
  - State conversion between world and body frames
  - Neural model interface for next-state prediction
  - Handles generalized coordinates (joint positions/velocities)

- **`integrator_neural_stateful.py`**: Extends base integrator with state history for sequence models

- **`integrator_neural_transformer.py`**: Transformer-based integrator for sequence modeling

- **`integrator_neural_rnn.py`**: RNN-based integrator variant

These implement the paper's core idea: replacing low-level dynamics and contact solvers with neural networks while maintaining the same `simulate()` interface as analytical integrators.

## **`envs/`** - Environment Wrappers
Provides the abstraction layer for using NeRD in different environments.

- **`neural_environment.py`**: Main wrapper that:
  - Creates both ground-truth and neural integrators
  - Allows switching between neural and analytical dynamics
  - Provides unified interface for RL training and evaluation

- **`abstract_contact_environment.py`**: Pre-extracts contact pair information needed by NeRD (contact points, normals, depths)

- **`abstract_contact.py`**: Abstract contact representation

- **`warp_sim_envs/`**: Concrete robot environments:
  - `env_ant.py`, `env_anymal.py`, `env_cartpole.py`, `env_franka_panda.py`, `env_pendulum_with_contact.py`
  - These are the test environments from the paper (Ant, ANYmal, Cartpole, Franka, Pendulum)

- **`rlgames_env_wrapper.py`**: Wrapper for RL training with rl-games

## **`models/`** - Neural Network Architectures
Implements the neural models used for dynamics prediction.

- **`models.py`**: Main model classes:
  - `ModelMixedInput`: Handles mixed inputs (states, contacts, actions)
  - Supports MLP, RNN (LSTM/GRU), and Transformer backends

- **`model_transformer.py`**: GPT-style Transformer architecture for sequence modeling

- **`base_models.py`**: Base MLP, LSTM, GRU architectures

- **`distributions.py`**: Probabilistic output distributions (if needed)

- **`model_utils.py`**: Model utilities

These implement the neural architectures that learn robot-specific dynamics.

## **`generate/`** - Dataset Generation
Generates training data from ground-truth physics simulations.

- **`generate_dataset_*.py`**: Environment-specific dataset generators:
  - `generate_dataset_ant.py`, `generate_dataset_anymal.py`, `generate_dataset_contact_free.py`, etc.
  - Samples random trajectories using ground-truth dynamics

- **`trajectory_sampler*.py`**: Trajectory sampling strategies for different robots

- **`simulation_sampler.py`**: Base simulation sampling utilities

- **`visualize_dataset.py`**: Visualization tools for generated datasets

This corresponds to the data generation step in the paper's methodology.

## **`train/`** - Training Infrastructure
Training scripts and configurations.

- **`train.py`**: Main training script that:
  - Loads datasets
  - Creates neural models
  - Trains using `VanillaTrainer` or `SequenceModelTrainer`

- **`cfg/`**: Training configuration files (YAML) for each robot:
  - Specifies model architecture, hyperparameters, input/output configurations
  - Separate configs for MLP vs Transformer models

- **`arguments.py`**: Command-line argument parsing

## **`algorithms/`** - Training Algorithms
Training algorithms for NeRD models.

- **`vanilla_trainer.py`**: Standard training for feedforward models (MLP)
  - Handles single-step prediction training
  - Loss computation, optimization, validation

- **`sequence_model_trainer.py`**: Training for sequence models (RNN/Transformer)
  - Handles multi-step sequence prediction
  - Different loss formulations for temporal models

## **`eval/`** - Evaluation Scripts
Evaluation scripts matching the paper's experiments.

- **`eval_passive/`**:
  - **`eval_passive_motion.py`**: Long-horizon passive motion evaluation (Section 5.1)
  - Tests NeRD's ability to simulate passive dynamics over long horizons

- **`eval_rl/`**:
  - **`run_rl.py`**: Individual RL policy evaluation
  - **`batch_eval_policy.py`**: Batch evaluation of RL policies (Table 1 in paper)
  - **`cfg/`**: RL task configurations (Ant run/spin, ANYmal forward/side walk, etc.)
  - **`eval_cfg/`**: Batch evaluation configurations

These reproduce the paper's experimental results.

## **`utils/`** - Utility Functions
Supporting utilities.

- **`datasets.py`**: Dataset loading and batching
- **`evaluator.py`**: Evaluation metrics and utilities
- **`torch_utils.py`**: PyTorch utilities
- **`warp_utils.py`**: Warp simulation utilities
- **`env_utils.py`**: Environment creation utilities
- **`logger.py`**: Logging infrastructure
- **`running_mean_std.py`**: Normalization utilities
- **`visualize_env.py`**: Environment visualization

## **`pretrained_models/`** - Pre-trained Models
Pre-trained NeRD models and RL policies.

- **`NeRD_models/`**: Pre-trained NeRD dynamics models for each robot
- **`RL_policies/`**: RL policies trained in NeRD-integrated simulators

These enable testing without training from scratch.

## **`figures/`** - Paper Figures
- **`overview.png`**: Overview figure from the paper

---

## Summary

The codebase implements NeRD as described in the paper:

1. **Core innovation** (`integrators/`): Neural integrators that replace analytical physics
2. **State representation**: Robot-centric, spatially-invariant (body frame) representation
3. **Training pipeline** (`generate/` → `train/`): Data generation from ground-truth sim, then neural model training
4. **Model architectures** (`models/`): MLP, RNN, and Transformer variants
5. **Evaluation** (`eval/`): Scripts reproducing paper experiments
6. **Integration** (`envs/`): Seamless integration with Warp simulator for RL and evaluation

The design allows NeRD to be a drop-in replacement for analytical integrators, enabling neural dynamics simulation while maintaining compatibility with existing simulation infrastructure.

