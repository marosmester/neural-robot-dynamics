# Abstract Contact Data in the Dataset Generation Pipeline

This document explains how contact data is created, used by the physics integrator,
and recorded into the dataset. There are two distinct pipelines depending on the
robot type.

---

## Table of Contents

1. [Background: What Are Abstract Contacts?](#1-background-what-are-abstract-contacts)
2. [Non-Pendulum Pipeline (Ant, Anymal, etc.)](#2-non-pendulum-pipeline-ant-anymal-etc)
3. [Pendulum Pipeline](#3-pendulum-pipeline)
4. [Side-by-Side Comparison](#4-side-by-side-comparison)
5. [Why Two Different Pipelines?](#5-why-two-different-pipelines)
6. [Why Not Use Warp's Default `wp.sim.collide()`?](#6-why-not-use-warps-default-wpsimcollide)

---

## 1. Background: What Are Abstract Contacts?

Warp's default collision pipeline (`wp.sim.collide()`) produces a **variable** number
of contact points each step. A neural network requires **fixed-size** inputs. The
`AbstractContact` class (`envs/abstract_contact.py`) solves this by maintaining a
**fixed set of contact point candidates** throughout the simulation.

At initialization, `AbstractContactEnvironment.initialize_contacts()` scans the
robot's geometry (capsules, boxes, spheres) and creates one contact candidate per
notable surface point (e.g., capsule endpoints, box corners). For the 2-link
pendulum this gives 4 candidates (2 capsule endpoints per link).

These candidates are stored as torch tensors that share GPU memory with the Warp
model's contact arrays (via `wp.from_torch()`). Writing to the torch side
automatically updates the Warp side, and vice versa.

### Contact fields

| Field | Fixed/Variable | Description |
|---|---|---|
| `contact_shape0` | Fixed | Shape index on the robot |
| `contact_point0` | Fixed | Contact point in the shape's body frame |
| `contact_thickness` | Fixed | Geometry thickness (e.g., capsule radius) |
| `contact_shape1` | Variable | Shape index of the external object (e.g., ground) |
| `contact_point1` | Variable | Contact point on the external object (world frame) |
| `contact_normal` | Variable | Contact normal (world frame) |
| `contact_depth` | Variable | Penetration depth |

All tensors have shape `(num_contacts_per_env * num_envs, ...)`.

---

## 2. Non-Pendulum Pipeline (Ant, Anymal, etc.)

Uses `TrajectorySampler.sample_trajectories_joint_act_mode` with `eval_collisions=True`.

The sampler is **read-only** with respect to contact data. All contact computation
happens inside the environment stack.

```
INITIALIZATION (once)
  AbstractContactEnvironment.initialize_contacts()
    -> Scans robot geometry (capsules, boxes, spheres)
    -> Creates fixed contact CANDIDATES in AbstractContact:
         contact_shape0, contact_point0, contact_thickness  (FIXED forever)
    -> These define WHERE on the robot contacts COULD happen

EACH STEP (trajectory_length times)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  env.step_with_joint_act(joint_acts)                               │
  │    │                                                                │
  │    │  AbstractContactEnvironment.update()                           │
  │    │    │                                                           │
  │    │    │  1. collision_detection_ground kernel                     │
  │    │    │     INPUT:  fixed contact_shape0, contact_point0          │
  │    │    │             + current body poses (state.body_q)           │
  │    │    │             + fixed ground plane from model definition    │
  │    │    │     OUTPUT: contact_depth, contact_normal, contact_point1 │
  │    │    │             (written into AbstractContact tensors)        │
  │    │    │                                                           │
  │    │    │  2. for _ in range(sim_substeps):                         │
  │    │    │       Featherstone.simulate()                             │
  │    │    │         READS: contact_depth, contact_normal, etc.        │
  │    │    │         USES them to compute contact forces               │
  │    │    │         WRITES: next state (joint_q, joint_qd)           │
  │    │    │                                                           │
  └────┼────┼───────────────────────────────────────────────────────────┘
       │    │
       v    v
  TrajectorySampler READS from AbstractContact:
    contact_point0     -> copies to recording buffer
    contact_point1     -> copies to recording buffer
    contact_normal     -> copies to recording buffer
    contact_depth      -> copies to recording buffer
    contact_thickness  -> copies to recording buffer

SAVING
  All recording buffers -> HDF5 file
```

**Key point:** The same contacts that `collision_detection_ground` computes are the
same ones that Featherstone uses for physics, and the same ones that get saved to
the dataset. Everything is consistent and the sampler never touches the contact data.

---

## 3. Pendulum Pipeline

Uses `TrajectorySamplerPendulum.sample_trajectories_abstract_mode` with
`eval_collisions=False`.

The sampler **generates and writes** contact data. Collision detection inside the
environment is disabled.

```
INITIALIZATION (once)
  AbstractContactEnvironment.initialize_contacts()
    -> Same as above: creates fixed contact CANDIDATES
         contact_shape0, contact_point0, contact_thickness  (FIXED forever)

  eval_collisions set to FALSE
    -> collision_detection_ground will NOT run during update()

EACH ROUND (until num_transitions reached)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Sampler generates a RANDOM GROUND PLANE per env:                  │
  │    1. Sample random normal vector n                                │
  │    2. Compute plane offset d so plane is near the pendulum         │
  │    (This plane stays FIXED for the entire trajectory)              │
  └─────────────────────────────────────────────────────────────────────┘

  EACH STEP (trajectory_length times)
    ┌───────────────────────────────────────────────────────────────────┐
    │  Sampler COMPUTES contacts analytically:                         │
    │    1. Transform contact_point0 to world frame                    │
    │    2. Project onto the random plane -> compute depth             │
    │    3. Compute contact_point1 from depth + normal                 │
    │                                                                   │
    │  Sampler WRITES to AbstractContact:                              │
    │    abstract_contacts.contact_normal = plane normals               │
    │    abstract_contacts.contact_depth  = computed depths             │
    │    abstract_contacts.contact_point1 = computed via warp kernel    │
    │                                                                   │
    │  (These torch writes flow to Warp via shared GPU memory)         │
    │                                                                   │
    │  Sampler also COPIES to its own recording buffers:               │
    │    contact_normals[step]     = normals                            │
    │    contact_depths[step]      = depths                             │
    │    contact_points_0[step]    = contact_point0 (fixed)             │
    │    contact_points_1[step]    = computed point1                    │
    │    contact_thicknesses[step] = thickness (fixed)                  │
    │                                                                   │
    │  env.step_with_joint_act(joint_acts)                             │
    │    │                                                              │
    │    │  AbstractContactEnvironment.update()                         │
    │    │    │                                                         │
    │    │    │  1. collision_detection -> SKIPPED (eval_collisions=F)  │
    │    │    │                                                         │
    │    │    │  2. for _ in range(sim_substeps):                       │
    │    │    │       Featherstone.simulate()                           │
    │    │    │         READS: the contacts that the SAMPLER wrote      │
    │    │    │         USES them to compute contact forces             │
    │    │    │         WRITES: next state                              │
    │    │    │                                                         │
    └────┼────┼─────────────────────────────────────────────────────────┘
         v    v
  SAVING
    Recording buffers -> HDF5 file
```

**Key point:** The sampler is the sole author of contact data. It writes contacts
into `AbstractContact`, Featherstone reads those same contacts to apply forces, and
the sampler also records those same values to the dataset. Everything is still
consistent -- the contacts that drove the physics are the contacts in the dataset.
They just didn't come from a collision detection kernel.

---

## 4. Side-by-Side Comparison

| | Non-Pendulum | Pendulum |
|---|---|---|
| **Sampler method** | `sample_trajectories_joint_act_mode` | `sample_trajectories_abstract_mode` |
| **Contact origin** | `collision_detection_ground` kernel | Sampler (random plane + analytic math) |
| **Ground plane** | Fixed (from model definition) | Random (per trajectory per env) |
| **`eval_collisions`** | `True` | `False` |
| **What Featherstone reads** | Contacts from collision kernel | Contacts written by sampler |
| **What goes into HDF5** | Same contacts from collision kernel | Same contacts written by sampler |
| **Sampler writes to AbstractContact?** | No (read-only) | Yes |
| **Are physics contacts = dataset contacts?** | Yes | Yes |
| **Contact diversity** | Low (one fixed plane) | High (random planes every trajectory) |

---

## 5. Why Two Different Pipelines?

The Pendulum uses abstract mode to achieve **contact diversity**. The neural model
needs to generalize across many different contact geometries (ground plane at
different orientations and positions). If the Pendulum used the same pipeline as
Ant/Anymal, every trajectory would interact with the same fixed ground plane defined
in the model file, producing limited contact variation in the training data.

By having the sampler generate random ground planes and analytically compute
contacts, the dataset covers a wide range of contact scenarios -- different plane
normals, different penetration depths, different contact/no-contact combinations
across the 4 contact candidates.

Ant and Anymal don't need this because their locomotion naturally produces diverse
contact patterns through varied poses and gaits on a single fixed ground plane.

---

## 6. Why Not Use Warp's Default `wp.sim.collide()`?

Both pipelines use a custom `collision_detection_ground` kernel (or manual
computation) instead of Warp's built-in `wp.sim.collide()`. The reason is the
same core problem that motivates the entire `AbstractContactEnvironment` layer:

| | `wp.sim.collide()` | Custom approach |
|---|---|---|
| Contact count | Variable per step | Fixed (`num_contacts_per_env * num_envs`) |
| Contact ordering | Non-deterministic | Deterministic (same index = same body point) |
| Generality | Handles any geometry pair | Only handles ground plane |
| Neural-network friendly | No (variable-size, unordered) | Yes (fixed-size, ordered) |

A neural network requires fixed-size, consistently-ordered input tensors. With
`wp.sim.collide()`, contact index 3 might refer to different body points across
different time steps. With the custom approach, contact index 3 always refers to
the same body surface point -- only its depth, normal, and projection change.
