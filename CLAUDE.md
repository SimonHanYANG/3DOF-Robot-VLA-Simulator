# CLAUDE.md — 3DOF Robot VLA Simulator

## Project Overview
3DOF (XYZ linear slide) robotic arm push-object obstacle-avoidance simulation platform. Pipeline: MuJoCo simulation → RRT expert data generation → VLA model training (OpenVLA/ACT/pi0) → closed-loop inference → React+Three.js 3D visualization.

## Conventions
- **Language**: Python 3.10+
- **Conda env**: `simvla` (PyTorch + CUDA pre-installed)
- **Branching**: `main` = stable/runnable; `dev/*` = active development
- **Commits**: Complete a runnable milestone → code review → push to dev branch → stop for user testing

## Key Directories
- `sim/` — MuJoCo simulation core (3DOF arm, scene XML, controller)
- `backend/` — FastAPI + WebSocket server
- `frontend/` — React + Three.js / R3F
- `models/` — VLA model adapters + training scripts
- `data/` — HDF5 trajectory datasets
- `doc/step2step/` — 20-step implementation guide (update status as you go)

## How to Run
```bash
conda activate simvla
python scripts/verify_mujoco.py   # verify MuJoCo
```

## Development Rules
1. Follow `doc/step2step/` steps sequentially
2. Update step status (🔴→🟡→🟢) in each step doc and README index
3. After each runnable milestone: code review, run tests, commit, push to dev branch
4. Stop after push — wait for user to test before continuing
5. Keep `main` branch clean (only fully working versions)
