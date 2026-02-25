# Dataset Generation Pipeline

This document explains how the dataset generation pipeline works, end-to-end.
It is written for someone who did **not** author this code and wants to understand
what happens when a command like the following is executed:

```bash
python generate_dataset_pendulum.py \
    --env-name PendulumWithContact \
    --num-transitions 10000 \
    --dataset-name pendulum10000.hdf5 \
    --trajectory-length 100 \
    --num-envs 512 \
    --seed 0
```

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Pipeline Execution Flow](#2-pipeline-execution-flow)
3. [CLI Arguments Explained](#3-cli-arguments-explained)
4. [Class Hierarchy and Wrapping Architecture](#4-class-hierarchy-and-wrapping-architecture)
5. [The Three Nested Loops](#5-the-three-nested-loops)
6. [Contact Sampling in Abstract Mode](#6-contact-sampling-in-abstract-mode)
7. [Substepping: Why `sim_dt` != `frame_dt`](#7-substepping-why-sim_dt--frame_dt)
8. [Output: HDF5 Dataset Structure](#8-output-hdf5-dataset-structure)
9. [File Map](#9-file-map)

---

## 1. High-Level Overview

The goal is to generate a dataset of **(state, action, next_state)** transitions
from a ground-truth physics simulator (NVIDIA Warp) so that a neural network can
later be trained to predict `next_state` given `(state, action, contact_info)`.

The pipeline:
1. Creates many parallel physics environments on the GPU.
2. Repeatedly: randomizes initial conditions, randomizes a contact plane,
   and rolls out trajectories using the Warp Featherstone integrator.
3. Records every transition (along with contact information) and saves them to
   an HDF5 file.

```
                              generate_dataset_pendulum.py (entry point)
                                           |
                                           v
                                    collect_dataset()
                                     /            \
                           NeuralEnvironment    TrajectorySamplerPendulum
                           (simulation env)     (sampling logic)
                                 |                        |
                                 v                        v
                    AbstractContactEnvironment    sample_trajectories_abstract_mode()
                    (contact abstraction layer)    /          |           \
                                 |              sample     simulate      record
                                 v             randoms    trajectories   to HDF5
                           Environment
                      (Warp physics engine)
                                 |
                                 v
                     wp.sim.FeatherstoneIntegrator
                       (actual physics solver)
```

---

## 2. Pipeline Execution Flow

Below is the **exact sequence** of what happens when you run the command above,
in the order it executes:

### Phase 1: Initialization

```
main()
  |
  +-- parse CLI arguments
  +-- set_random_seed(seed)
  +-- collect_dataset(...)
        |
        +-- Open HDF5 file for writing
        |
        +-- Create NeuralEnvironment
        |     |
        |     +-- create_abstract_contact_env()          [utils/env_utils.py]
        |     |     |
        |     |     +-- PendulumWithContactEnvironment() [envs/warp_sim_envs/env_pendulum_with_contact.py]
        |     |     |     |
        |     |     |     +-- Build articulation:
        |     |     |     |     2 revolute joints, 2 capsule links
        |     |     |     +-- Create Featherstone integrator
        |     |     |     |     (sim_substeps = 5, frame_dt = 1/60)
        |     |     |     +-- Create Warp model on GPU
        |     |     |     +-- Allocate state_0, state_1 buffers
        |     |     |
        |     |     +-- AbstractContactEnvironment(env)  [envs/abstract_contact_environment.py]
        |     |           |
        |     |           +-- initialize_contacts()
        |     |           |     Determine fixed contact point candidates
        |     |           |     (4 points: 2 capsule ends per link)
        |     |           +-- Override update() with custom collision handling
        |     |
        |     +-- Create NeuralIntegrator (dummy, with neural_model=None)
        |     +-- Store reference to GT integrator
        |     +-- Set env mode to "ground-truth"
        |
        +-- Create TrajectorySamplerPendulum
        |     |
        |     +-- Store sampling bounds from utils/commons.py:
        |           joint_q  in [-pi, pi]
        |           joint_qd in [-2*pi, 4*pi]  (per joint)
        |           joint_act_scale = 1500.0
        |
        +-- Call sample_trajectories_abstract_mode()
              (see Phase 2 below)
```

### Phase 2: Data Collection (The Main Loop)

```
sample_trajectories_abstract_mode(num_transitions=10000, trajectory_length=100)
  |
  +-- Allocate GPU buffers for states, next_states, joint_acts, contacts, etc.
  |     All shaped as (trajectory_length, num_envs, dim)
  |
  +-- Disable collision detection in the environment
  |     (contacts are sampled manually in "abstract mode")
  |
  +-- for each ROUND (repeats until num_transitions reached):
  |     |
  |     +-- Sample random initial states for all 512 envs
  |     +-- Sample random joint torques for all 512 envs x 100 steps
  |     +-- Reset all environments to the random initial states
  |     |
  |     +-- SAMPLE A RANDOM GROUND PLANE per env:
  |     |     1. Compute contact point positions in world frame
  |     |     2. Sample a random plane normal (unit vector)
  |     |     3. Compute d so the plane is near but not penetrating the pendulum
  |     |     (This plane stays FIXED for the entire trajectory)
  |     |
  |     +-- for step in range(trajectory_length):   [100 iterations]
  |     |     |
  |     |     +-- Compute contact depths from current pendulum pose vs. the plane
  |     |     +-- Compute contact_point1 (projection onto plane)
  |     |     +-- Write contact info into the environment's abstract contact buffers
  |     |     +-- Record: states[step] = current state
  |     |     +-- Call env.step_with_joint_act(joint_acts[step], env_mode='ground-truth')
  |     |     |     |
  |     |     |     +-- set_env_mode('ground-truth')
  |     |     |     |     -> integrator = FeatherstoneIntegrator
  |     |     |     |     -> sim_substeps = 5
  |     |     |     |     -> sim_dt = frame_dt / 5
  |     |     |     |
  |     |     |     +-- Assign joint torques to Warp control
  |     |     |     +-- env.update()    [AbstractContactEnvironment.update()]
  |     |     |     |     |
  |     |     |     |     +-- (collision_detection skipped, eval_collisions=False)
  |     |     |     |     +-- for _ in range(sim_substeps):    [5 iterations]
  |     |     |     |     |     |
  |     |     |     |     |     +-- state.clear_forces()
  |     |     |     |     |     +-- integrator.simulate(model, state_0, state_1, sim_dt, control)
  |     |     |     |     |     |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |     |     |     |     |     |     THIS IS WHERE PHYSICS ACTUALLY HAPPENS
  |     |     |     |     |     |     Featherstone algorithm solves:
  |     |     |     |     |     |       M(q) * qdd = tau - C(q, qd) - G(q) + J^T * f_contact
  |     |     |     |     |     |
  |     |     |     |     |     +-- swap state_0, state_1
  |     |     |     |
  |     |     |     +-- Read new state from Warp into torch tensor (joint_q, joint_qd)
  |     |     |     +-- Wrap continuous angles to [-pi, pi]
  |     |     |     +-- Return next_states
  |     |     |
  |     |     +-- Record: next_states[step] = returned state
  |     |
  |     +-- Append this round's data to rollout_batches lists
  |
  +-- Concatenate all rounds along the envs dimension
  +-- Return rollouts dict
```

### Phase 3: Saving

```
collect_dataset() continued:
  |
  +-- Write all rollout tensors to HDF5 datasets:
  |     states, next_states, joint_acts, gravity_dir, root_body_q,
  |     contact_normals, contact_depths, contact_points_0, contact_points_1,
  |     contact_thicknesses
  |
  +-- Write metadata attributes:
  |     total_trajectories, total_transitions, state_dim, joint_act_dim, etc.
  |
  +-- Flush and close the HDF5 file
```

---

## 3. CLI Arguments Explained

| Argument | Default | Meaning |
|---|---|---|
| `--num-envs` | 1024 | Number of parallel pendulum simulations on the GPU. Each env is independent with its own state, torque, and contact plane. |
| `--trajectory-length` | 100 | Number of simulation steps per trajectory. Each step advances the simulation by `frame_dt` (1/60 s). |
| `--num-transitions` | 1000000 | Target total number of transitions. The loop runs `ceil(num_transitions / (num_envs * trajectory_length))` rounds. |
| `--contact-prob` | 0.5 | Probability that a sampled contact plane actually intersects the pendulum's reachable space (controls the ratio of contact vs. contact-free data). |
| `--passive` | False | If set, all joint torques are zeroed (no actuation, gravity-only simulation). |
| `--seed` | 0 | Random seed for reproducibility. |

**Transitions per round** = `num_envs * trajectory_length`.

With the example command (`num_transitions=10000, num_envs=512, trajectory_length=100`):
- Transitions per round = 512 * 100 = 51,200
- Rounds needed = ceil(10,000 / 51,200) = 1
- Actual transitions generated = 51,200 (overshoots the target)

---

## 4. Class Hierarchy and Wrapping Architecture

The simulation environment is built from three layers, each adding
a specific concern. Understanding **why** each layer exists is key to
understanding the code.

```
+-----------------------------------------------------------------------+
|                        NeuralEnvironment                              |
|   Purpose: Provides a unified interface that can switch between       |
|            ground-truth physics and a neural-network integrator.      |
|            Also manages torch-level state tensors.                    |
|   File:    envs/neural_environment.py                                 |
|                                                                       |
|   +---------------------------------------------------------------+   |
|   |                AbstractContactEnvironment                     |   |
|   |   Purpose: Wraps around the base environment to provide a     |   |
|   |            FIXED set of contact point candidates. In the      |   |
|   |            base Warp engine, the number of contacts varies    |   |
|   |            per step (collision detection creates/removes       |   |
|   |            contacts dynamically). A neural network needs      |   |
|   |            fixed-size inputs, so this layer guarantees a      |   |
|   |            constant number of contact pairs per environment.  |   |
|   |   File:    envs/abstract_contact_environment.py               |   |
|   |                                                               |   |
|   |   Key: Overrides update() to use custom collision detection   |   |
|   |        instead of Warp's default wp.sim.collide()             |   |
|   |                                                               |   |
|   |   +-------------------------------------------------------+   |   |
|   |   |              Environment (base)                       |   |   |
|   |   |   Purpose: The core Warp simulation engine. Builds    |   |   |
|   |   |            the physics model (articulation, shapes,   |   |   |
|   |   |            ground plane), creates the integrator,     |   |   |
|   |   |            manages states and rendering.              |   |   |
|   |   |   File:    envs/warp_sim_envs/environment.py          |   |   |
|   |   |                                                       |   |   |
|   |   |   Contains: integrator.simulate() -- the actual       |   |   |
|   |   |             physics step                              |   |   |
|   |   +-------------------------------------------------------+   |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------------------------------------------+
```

### Why three layers?

- **`Environment`** handles pure physics: building the articulation (pendulum links,
  joints, shapes), choosing an integrator (Featherstone), managing Warp states,
  and calling `integrator.simulate()`.

- **`AbstractContactEnvironment`** solves a data-representation problem. Warp's
  default collision pipeline (`wp.sim.collide()`) produces a **variable** number
  of contacts each step. A neural network requires a **fixed-size** input tensor.
  This wrapper pre-computes all *possible* contact points at initialization time
  (e.g., the 4 capsule endpoints of the 2-link pendulum) and maintains them as a
  fixed-size array. During simulation, only the contact *depths* and *normals*
  change -- the contact *candidates* stay the same.

- **`NeuralEnvironment`** adds the ability to **swap integrators** at runtime.
  It holds both the ground-truth Featherstone integrator and a neural integrator.
  Calling `set_env_mode('ground-truth')` or `set_env_mode('neural')` switches which
  one is used by `update()`. During dataset generation, only the ground-truth
  integrator is used. During inference/evaluation, the neural integrator is used.

### How wrapping works (delegation pattern)

`AbstractContactEnvironment` uses Python's `__getattr__` and `__setattr__` to
delegate all attribute access to the wrapped `Environment` instance. This means
you can call any `Environment` method on the `AbstractContactEnvironment` object
transparently -- only `update()` is overridden with the custom contact-aware
version.

`NeuralEnvironment` stores the `AbstractContactEnvironment` as `self.env` and
exposes its properties via explicit `@property` decorators and forwarding methods.

---

## 5. The Three Nested Loops

The data collection has three levels of looping:

```
OUTER LOOP: Rounds
  One round = one batch of parallel trajectories.
  Iterates until cumulative transitions >= num_transitions.
  Each round produces (num_envs * trajectory_length) transitions.

    MIDDLE LOOP: Steps within a trajectory  (trajectory_length iterations)
      Each step = one frame_dt time advance.
      Records one (state, action, next_state) transition per environment.
      Calls env.step_with_joint_act() which triggers env.update().

        INNER LOOP: Physics substeps  (sim_substeps iterations, e.g. 5)
          Each substep = one sim_dt time advance (sim_dt = frame_dt / sim_substeps).
          Calls integrator.simulate(model, state_in, state_out, sim_dt, control).
          This is where the Featherstone algorithm actually solves the equations of motion.
```

Visually, for one round with `num_envs=512, trajectory_length=100, sim_substeps=5`:

```
Round (1 iteration)
 |
 +-- 512 envs reset to random states
 +-- 512 random ground planes sampled
 |
 +-- Step 0 -----> 5 substeps -----> record (state_0, act_0, state_1) x 512 envs
 +-- Step 1 -----> 5 substeps -----> record (state_1, act_1, state_2) x 512 envs
 +-- ...
 +-- Step 99 ----> 5 substeps -----> record (state_99, act_99, state_100) x 512 envs
 |
 = 512 * 100 = 51,200 transitions generated
 = 512 * 100 * 5 = 256,000 integrator.simulate() calls executed on GPU
```

---

## 6. Contact Sampling in Abstract Mode

For the Pendulum, contacts are **not** detected by the Warp collision engine.
Instead, `TrajectorySamplerPendulum` uses **abstract mode**: it randomly samples
a ground plane and analytically computes which contact points penetrate it.

This is done because:
- The neural model needs to generalize across **many** contact configurations
  (different ground orientations/positions).
- Warp's collision detection only works with the ground plane as built in the
  model definition. Abstract mode allows sampling arbitrary planes per trajectory.

### How a ground plane is sampled (per round, per env):

1. **Sample a random normal vector** n (uniform on the unit sphere).
2. **Compute d** (the plane offset) such that the plane `dot(n, x) = d` is
   near the pendulum's contact points but not penetrating them:
   - `d_upper` = the projection of the closest contact point onto n, minus the
     contact thickness (tangential case).
   - `d_lower` = the projection of the root body onto n, minus 3.5
     (ensures the plane isn't absurdly far away).
   - `d` is sampled uniformly in `[d_lower, d_upper]`.
3. This plane is **fixed for the entire trajectory** (100 steps).

At each step, the contact depth for each contact point is computed as:
```
depth = dot(contact_point_world, normal) - d
```
A positive depth means no penetration; a negative depth means the point is below
the plane surface.

---

## 7. Substepping: Why `sim_dt` != `frame_dt`

Physics integrators numerically solve ordinary differential equations (ODEs).
The accuracy and stability of this approximation depends on the time step size.

- **`frame_dt`** = 1/60 s -- the interval between consecutive data points.
  This is what the neural network will learn to predict.
- **`sim_dt`** = `frame_dt / sim_substeps` -- the actual integration step.
  For Featherstone with `sim_substeps=5`: sim_dt = 1/300 s.

Within a single `frame_dt`, the integrator runs `sim_substeps` times at the
finer `sim_dt`. This is necessary because:
- A single 1/60 s step would be too large for stable simulation of stiff contacts
  and joint dynamics.
- Multiple smaller steps accurately approximate the continuous-time physics
  while producing data at a practical frequency.

The neural integrator (used at inference time, **not** during dataset generation)
uses `sim_substeps=1` because it directly predicts the `frame_dt` transition
as a black box -- it doesn't solve an ODE.

```
Ground-truth mode:                    Neural mode (inference):
  frame_dt = 1/60 s                     frame_dt = 1/60 s
  sim_substeps = 5                      sim_substeps = 1
  sim_dt = 1/300 s                      sim_dt = 1/60 s
                                        
  |-----|-----|-----|-----|-----|        |---------------------------|
  sub1  sub2  sub3  sub4  sub5          single neural network call
  <--------- frame_dt -------->         <------- frame_dt --------->
```

---

## 8. Output: HDF5 Dataset Structure

The generated `.hdf5` file has the following structure:

```
/data                              (HDF5 group)
  |-- attrs:
  |     env                        = "PendulumWithContact"
  |     mode                       = "trajectory"
  |     total_trajectories         = number of trajectory sequences
  |     total_transitions          = trajectory_length * total_trajectories
  |     state_dim                  = 4  (2 joint angles + 2 joint velocities)
  |     joint_act_dim              = 2  (torque per joint)
  |     next_state_dim             = 4
  |     contact_prob               = 0.5
  |     num_contacts_per_env       = 4  (2 capsule endpoints per link)
  |
  |-- states                       shape: (trajectory_length, total_trajectories, 4)
  |-- next_states                  shape: (trajectory_length, total_trajectories, 4)
  |-- joint_acts                   shape: (trajectory_length, total_trajectories, 2)
  |-- gravity_dir                  shape: (trajectory_length, total_trajectories, 3)
  |-- root_body_q                  shape: (trajectory_length, total_trajectories, 7)
  |-- contact_normals              shape: (trajectory_length, total_trajectories, 4, 3)
  |-- contact_depths               shape: (trajectory_length, total_trajectories, 4)
  |-- contact_points_0             shape: (trajectory_length, total_trajectories, 4, 3)
  |-- contact_points_1             shape: (trajectory_length, total_trajectories, 4, 3)
  |-- contact_thicknesses          shape: (trajectory_length, total_trajectories, 4)
```

Each transition `(states[t, i], joint_acts[t, i]) -> next_states[t, i]` represents
one `frame_dt` advancement of environment `i` at step `t`, along with the contact
geometry at that instant.

---

## 9. File Map

Key files involved in the pipeline, listed in the order they are typically invoked:

| File | Role |
|---|---|
| `generate/generate_dataset_pendulum.py` | **Entry point.** Parses CLI args, creates the environment and sampler, calls the sampling loop, and writes the HDF5 output. |
| `utils/python_utils.py` | `set_random_seed()` for reproducibility. |
| `utils/commons.py` | Stores per-robot constants: joint position limits (`JOINT_Q_MIN/MAX`), velocity limits (`JOINT_QD_MIN/MAX`), and torque scales (`JOINT_ACT_SCALE`). |
| `utils/env_utils.py` | `create_abstract_contact_env()` -- factory function that instantiates the correct Warp environment class and wraps it in `AbstractContactEnvironment`. |
| `envs/warp_sim_envs/env_pendulum_with_contact.py` | `PendulumWithContactEnvironment` -- defines the pendulum articulation (2 revolute joints, 2 capsule links), Featherstone integrator config, ground plane setup, and randomized reset logic. |
| `envs/warp_sim_envs/environment.py` | `Environment` -- base class that builds the Warp `Model`, creates the physics integrator, allocates states, and implements `step()` which calls `integrator.simulate()`. |
| `envs/abstract_contact_environment.py` | `AbstractContactEnvironment` -- wraps `Environment` to provide a fixed-size contact representation. Overrides `update()` with custom collision detection that projects contact points onto the ground plane. |
| `envs/abstract_contact.py` | `AbstractContact` -- data class holding the fixed-size contact tensors (torch) and creating warp views into the Warp model's contact arrays. |
| `envs/neural_environment.py` | `NeuralEnvironment` -- top-level environment wrapper that holds both ground-truth and neural integrators and allows switching between them via `set_env_mode()`. Provides torch-level state management (`_update_states`, `step_with_joint_act`). |
| `generate/simulation_sampler.py` | `WarpSimDataGenerator` (base class for all samplers), `UniformSampler` / `SobolSampler`, and Warp kernels for computing contact points in world frame. |
| `generate/trajectory_sampler.py` | `TrajectorySampler` -- base trajectory sampler with `sample_trajectories_joint_act_mode()` (uses env's built-in collision detection). |
| `generate/trajectory_sampler_pendulum.py` | `TrajectorySamplerPendulum` -- pendulum-specific sampler with `sample_trajectories_abstract_mode()` that manually samples random ground planes and computes contacts analytically. |
| `integrators/integrator_neural.py` | `NeuralIntegrator` -- a Warp `Integrator` subclass whose `simulate()` runs a neural network instead of solving physics equations. Used at inference time, **not** during dataset generation (but instantiated as a dummy). |
| `utils/warp_utils.py` | GPU kernels for transferring state between Warp arrays and torch tensors, and wrappers for forward/inverse kinematics (`eval_fk`, `eval_ik`). |
