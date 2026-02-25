# Contact attributes written by PendulumSimulNerd

This document briefly explains the contact-related fields written to CSV via `write_contact_inputs_to_csv` in `PendulumSimulNerd.py` (from `model_inputs` produced by the neural integrator). The data comes from Warp’s rigid contact state: each **contact slot** (0 to N−1; N=4 for the default pendulum) corresponds to one rigid contact pair in the simulation (e.g. pendulum capsule vs ground).

**Coordinate frame:** All vectors and points below are in the **world frame**. The pendulum environment uses **Y-up** (gravity along −Y). The ground is approximately the XZ plane at a given Y (depending on `CONTACT_CONFIG`).

---

## contact_mask_0 … contact_mask_3

| Property   | Description |
|-----------|-------------|
| **Meaning** | Binary flag: 1 = this contact pair is considered “in contact”, 0 = not in contact. |
| **Position** | Per contact slot (no spatial position; it’s a label for that pair). |
| **Units**   | Dimensionless. |
| **Typical values** | 0.0 when no contact; 1.0 when `contact_depth` is below the contact threshold (see below). |

The mask is derived from depth and thickness: contact is declared when  
`contact_depth < max(CONTACT_DEPTH_UPPER_RATIO * contact_thickness, MIN_CONTACT_EVENT_THRESHOLD)`  
(with `CONTACT_DEPTH_UPPER_RATIO = 4` and `MIN_CONTACT_EVENT_THRESHOLD = 0.12` in `utils/commons.py`).

---

## contact_normal_0 … contact_normal_11

| Property   | Description |
|-----------|-------------|
| **Meaning** | Contact normal vector in world frame for each slot. Stored as 3 consecutive components per slot: indices 0–2 → slot 0 (nx, ny, nz), 3–5 → slot 1, 6–8 → slot 2, 9–11 → slot 3. |
| **Position** | Same normal applies to the contact point between the two bodies of that pair (pendulum vs ground). Direction is outward from the contact surface (convention: typically from ground toward the pendulum when the pendulum hits the ground). |
| **Units**   | Dimensionless (unit vector). |
| **Typical values** | When slot is in contact: e.g. (0, 1, 0) for a horizontal ground (Y-up); when no contact the normal is zeroed. |

---

## contact_depth_0 … contact_depth_3

| Property   | Description |
|-----------|-------------|
| **Meaning** | Penetration depth between the two shapes in this contact pair: how far one shape has moved into the other. |
| **Position** | Per contact slot (along the contact normal). |
| **Units**   | Length (same as simulation, e.g. metres). |
| **Typical values** | **No contact:** large value (e.g. 1000.0) used as a sentinel. **In contact:** small positive values (e.g. ~0.06–0.13 m when the pendulum capsule touches the ground). Smaller depth = tighter contact. |

---

## contact_thickness_0 … contact_thickness_3

| Property   | Description |
|-----------|-------------|
| **Meaning** | “Thickness” used for this contact pair in the simulation (e.g. related to the combined contact margin / compliance). Used together with depth to compute the contact mask threshold. |
| **Position** | Per contact slot (no spatial position). |
| **Units**   | Length (same as simulation). |
| **Typical values** | Often constant per slot; e.g. 0.1 m. When there is no active contact, values can be 0. |

---

## contact_point_0_0 … contact_point_0_11

| Property   | Description |
|-----------|-------------|
| **Meaning** | Contact point on **body 0** (robot / pendulum side) in world frame. Stored as 3 components per slot: 0–2 → slot 0 (x, y, z), 3–5 → slot 1, etc. |
| **Position** | On the pendulum: the point on the capsule (or other shape) that is in contact (or would be), in world coordinates. |
| **Units**   | Length (same as simulation). |
| **Typical values** | When slot is in contact: coordinates of the capsule–ground contact on the pendulum (e.g. Y near ground height, X/Z depending on link pose). When no contact: often (0, 0, 0) or a default. |

---

## contact_point_1_0 … contact_point_1_11

| Property   | Description |
|-----------|-------------|
| **Meaning** | Contact point on **body 1** (ground / external object side) in world frame. Layout as for `contact_point_0_*`: 3 components per slot. |
| **Position** | On the ground (or other external shape): the point on that shape corresponding to the same contact pair, in world coordinates. |
| **Units**   | Length (same as simulation). |
| **Typical values** | When in contact: point on the ground plane (e.g. Y = ground height). When no contact: often (0, 0, 0) or default. |

---

## Summary

- **contact_mask**: 0/1 per slot — “is this pair in contact?”
- **contact_normal**: unit vector (world) per slot — direction outward from the contact.
- **contact_depth**: penetration depth (length); large when no contact, small positive when in contact.
- **contact_thickness**: length scale per slot used for the contact threshold.
- **contact_point_0**: world position of the contact on the pendulum (body 0).
- **contact_point_1**: world position of the contact on the ground (body 1).

All lengths use the same units as the simulation (e.g. metres). For the default double-pendulum, only a subset of the 4 slots are usually active at a time (typically the link that hits the ground).
