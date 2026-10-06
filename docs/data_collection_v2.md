# Tactile Data Collection Schema V2

Each run is written below `dataset/auto/run_<timestamp>/`. Every attempt produces
one compressed NPZ file and one same-stem JSON file. Failed attempts, including
`no_contact`, are retained.

## Episode lifecycle

Recording starts after the simulator and object are reset, before
`reset_settle`. It ends when the attempt reaches one of these outcomes:

- `success_on_B`: while still grasped, the object and gripper simultaneously
  contact region B and the object respectively; after that, the gripper opens,
  releases the object, and the object remains in contact with B;
- `no_contact`: closing timed out without tactile contact;
- `lift_failed`: contact occurred but the object was not lifted;
- `dropped`: a lifted object returned near the support surface with low force;
- `placement_not_confirmed`: the object/gripper/B simultaneous contact was not
  confirmed before the descent timeout;
- `placement_limit_reached`: the gripper reached the safe lower placement limit
  without the object contacting B;
- `grasp_lost`: gripper-object contact was lost for the configured confirmation
  window during lift, transfer, or placement;
- `missed_B`: release timed out before an open, contact-free gripper left the
  object on B.

`success`/`result` is the episode-level training target. `finish_reason` in JSON
is the more precise terminal cause.

## Per-frame arrays

All arrays below have the same leading length `N`:

- tactile: `force_map`, `position_map`, `raw_force`, `force_stats`;
- progress: `time`, `stage_id`, `state_label`, `contact_mask`, `object_on_b`,
  `gripper_object_contact`, `gripper_released`;
- robot: `tcp_pos`, `tcp_quat`, `gripper_ctrl`, `gripper_qpos`;
- object: `object_pos`, `object_quat`, `object_vel`.

`state_label` values are `0=no_contact`, `1=weak_contact`,
`2=stable_grasp`, and `3=slip_or_drop`. Labels are causal: only the current and
previous simulation states are used. The confirmation windows and thresholds
are saved in JSON for reproducibility.

## Episode metadata

The JSON sidecar includes `finish_reason`, stage/state mappings, frame count,
duration, attempt index, object shape/size/mass/friction/initial position,
gripper close target, random seed, collection geometry, label thresholds, and
the final `placement_confirmed`/`release_confirmed` checks.

## Collection and inspection

```bash
python main.py --episodes 100 --save-dir dataset/auto --seed 42
python scripts/showdata.py dataset/auto --summary
python scripts/showdata.py dataset/auto
```

The summary reports finish reasons and per-state frame counts. The interactive
viewer shows the current stage, state, and left/right contact mask. Before a
large collection run, inspect a small pilot run and tune label thresholds from
the observed force and state transitions.
