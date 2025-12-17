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
Standalone function to extract generalized coordinates from warp states
and transform them to robot-centric body frame.

This function handles the complete pipeline from warp's maximal coordinates
(e.g., 26 values for double pendulum with 2 bodies) to minimal generalized
coordinates in robot-centric frame (e.g., 4 values for double pendulum).

Simplified for single articulation per environment (one robot at a time).
"""

import numpy as np
import torch
import warp as wp
from . import torch_utils


def extract_generalized_states(
    warp_states: wp.sim.State,
    model: wp.sim.Model,
    states_frame: str = 'body',
    translation_only: bool = False,
    device: str = 'cuda:0',
    wrap_continuous_dofs: bool = False,
    is_continuous_dof: np.ndarray = None,
) -> torch.Tensor:
    """
    Extract generalized coordinates from warp states and transform to robot-centric frame.
    
    Simplified for single articulation per environment (one robot at a time).
    
    This function performs two key transformations:
    1. Dimensionality reduction: Maximal coordinates (body_q, body_qd) → Generalized coordinates (joint_q, joint_qd)
    2. Frame transformation: World frame → Robot-centric body frame (if states_frame != 'world')
    
    Args:
        warp_states: Warp State object containing joint_q, joint_qd, and body_q
        model: Warp Model object containing articulation structure (single articulation)
        states_frame: 'world', 'body', or 'body_translation_only' (default: 'body')
        translation_only: If True, only transform translation, not rotation (default: False)
        device: Torch device for output tensors (default: 'cuda:0')
        wrap_continuous_dofs: If True, wrap continuous angular DOFs to [-π, π] (default: False)
        is_continuous_dof: Boolean array indicating which DOFs are continuous (required if wrap_continuous_dofs=True)
    
    Returns:
        torch.Tensor: States in generalized coordinates, shape (state_dim,)
            where state_dim = dof_q + dof_qd
    
    Example:
        For a double pendulum with 2 bodies (26 values in maximal coordinates):
        >>> states = extract_generalized_states(warp_states, model)
        >>> # Returns (4,) tensor: [q1, q2, qd1, qd2] in body frame
    """
    # Validate inputs
    assert states_frame in ['world', 'body', 'body_translation_only'], \
        f"states_frame must be 'world', 'body', or 'body_translation_only', got {states_frame}"
    
    if wrap_continuous_dofs:
        assert is_continuous_dof is not None, \
            "is_continuous_dof must be provided if wrap_continuous_dofs=True"
    
    # Extract model parameters (for single articulation)
    art_starts = model.articulation_start.numpy()
    q_starts = model.joint_q_start.numpy()
    qd_starts = model.joint_qd_start.numpy()
    
    # Get DOFs for the single articulation
    i0 = art_starts[0]
    i1 = art_starts[1] if len(art_starts) > 1 else len(q_starts)
    dof_q = int(q_starts[i1] - q_starts[i0])
    dof_qd = int(qd_starts[i1] - qd_starts[i0])
    state_dim = dof_q + dof_qd
    
    # Get root body index (first body of the articulation)
    root_body_idx = 0  # First body is the root
    
    # Get joint types (to check if root is JOINT_FREE)
    joint_types = model.joint_type.numpy()
    root_joint_type = joint_types[0] if len(joint_types) > 0 else None
    
    # Convert device string to torch device
    torch_device = torch.device(device)
    
    # Step 1: Extract generalized coordinates from warp states (direct extraction, no kernel needed)
    # Get velocity DOF indices
    qd_i0 = qd_starts[i0]
    qd_i1 = qd_starts[i1] if len(qd_starts) > i1 else len(wp.to_torch(warp_states.joint_qd))
    
    joint_q_torch = wp.to_torch(warp_states.joint_q)[i0:i1]
    joint_qd_torch = wp.to_torch(warp_states.joint_qd)[qd_i0:qd_i1]
    
    # Concatenate positions and velocities
    states_torch = torch.cat([joint_q_torch, joint_qd_torch], dim=0).to(torch_device)
    
    # Step 2: Extract root body pose
    # body_q is stored as transform with shape (num_bodies, 7) [x, y, z, qx, qy, qz, qw]
    root_body_q_torch = wp.to_torch(warp_states.body_q)[root_body_idx, :].unsqueeze(0)  # (1, 7)
    
    # Step 3: Transform to body frame if requested
    if states_frame == 'world':
        # No transformation needed
        states_body = states_torch
    else:
        # Extract body frame pose
        body_frame_pos = root_body_q_torch[:, :3]  # (1, 3)
        if translation_only or states_frame == 'body_translation_only':
            body_frame_quat = torch.zeros_like(root_body_q_torch[:, 3:7])
            body_frame_quat[:, 3] = 1.0  # Identity quaternion
        else:
            body_frame_quat = root_body_q_torch[:, 3:7]  # (1, 4)
        
        # Transform states to body frame
        states_body = states_torch.clone()
        
        # Only transform if root joint is JOINT_FREE (has position/orientation in world frame)
        if root_joint_type == wp.sim.JOINT_FREE:
            (
                states_body[0:3],
                states_body[3:7],
                states_body[dof_q:dof_q + 3],
                states_body[dof_q + 3:dof_q + 6]
            ) = torch_utils.convert_states_w2b(
                body_frame_pos.squeeze(0),  # (3,)
                body_frame_quat.squeeze(0),  # (4,)
                p=states_torch[0:3],
                quat=states_torch[3:7],
                omega=states_torch[dof_q:dof_q + 3],
                nu=states_torch[dof_q + 3:dof_q + 6]
            )
    
    # Step 4: Wrap continuous DOFs if requested
    if wrap_continuous_dofs and is_continuous_dof is not None:
        is_continuous_torch = torch.from_numpy(is_continuous_dof).to(torch_device)
        if is_continuous_torch.any():
            assert states_body.shape[-1] == is_continuous_torch.shape[0]
            wrap_delta = torch.floor(
                (states_body[is_continuous_torch] + np.pi) / (2 * np.pi)
            ) * (2 * np.pi)
            states_body[is_continuous_torch] -= wrap_delta
    
    return states_body

