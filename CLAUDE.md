# CLAUDE.md — UR5 Robot VLA Simulator

## Project Overview
UR5e 6-DOF robotic arm + parallel jaw gripper pick-and-place simulation platform. Pipeline: MuJoCo simulation → grasp planning → expert trajectory generation → VLA model training (OpenVLA/ACT/π0) → closed-loop inference → React+Three.js 3D visualization.

**Action Space**: 7D Cartesian delta + gripper (dx, dy, dz, droll, dpitch, dyaw, gripper_open). Internal IK converts to joint targets.

## Conventions
- **Language**: Python 3.10+
- **Conda env**: `simvla` (PyTorch + CUDA pre-installed)
- **Branching**: `main` = stable/runnable; `dev/*` = active development
- **Commits**: Complete a runnable milestone → code review → push to dev branch → stop for user testing

## Key Directories
- `sim/` — MuJoCo simulation core (UR5e arm, gripper, scene XML, IK solver, controller)
- `sim/planner/` — Grasp planner, pick-and-place trajectory generator, data pipeline
- `backend/` — FastAPI + WebSocket server
- `frontend/` — React + Three.js / R3F
- `models/` — VLA model adapters + training scripts
- `data/` — HDF5 trajectory datasets
- `doc/step2step/` — 20-step implementation guide (update status as you go)

## How to Run
```bash
conda activate simvla
python scripts/verify_mujoco.py   # verify MuJoCo
python scripts/view_scene.py      # view UR5 scene (after S02)
```

## Development Rules
1. Follow `doc/step2step/` steps sequentially
2. Update step status (🔴→🟡→🟢) in each step doc and README index
3. After each runnable milestone: code review, run tests, commit, push to dev branch
4. Stop after push — wait for user to test before continuing
5. Keep `main` branch clean (only fully working versions)

## Action Space Detail
```
action = (dx, dy, dz, droll, dpitch, dyaw, gripper_open)
  dx, dy, dz       : position delta in meters, range ±0.02m
  droll, dpitch, dyaw : rotation delta in radians, range ±0.1rad
  gripper_open     : 0.0 = closed, 1.0 = open
```

## MuJoCo Menagerie
UR5e arm model sourced from `google-deepmind/mujoco_menagerie` → `universal_robots_ur5e/`. Gripper is custom-defined parallel jaw.
