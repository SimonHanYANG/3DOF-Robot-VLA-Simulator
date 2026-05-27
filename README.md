# UR5 Robot VLA Simulator

UR5e 六自由度机械臂取放仿真平台 — 仿真、数据采集、模型训练、推理与可视化一体化平台。

## Quick Start

### Prerequisites
- Python 3.10+
- MuJoCo 3.1+
- PyTorch 2.1+ (with CUDA)

### Setup

```bash
# Clone
git clone git@github.com:SimonHanYANG/3DOF-Robot-VLA-Simulator.git
cd 3DOF-Robot-VLA-Simulator

# Install dependencies
pip install -r requirements.txt

# Verify MuJoCo
python scripts/verify_mujoco.py
```

### Project Structure

```
├── sim/              # Simulation core (UR5e arm, gripper, scene, IK, controller)
│   ├── assets/       # MuJoCo XML scene files + meshes
│   └── planner/      # Grasp planner, trajectory generator, data pipeline
├── backend/          # FastAPI + WebSocket server
│   └── services/     # Business logic (sim service, inference engine)
├── frontend/         # React + Three.js 3D UI
├── models/           # VLA model adapters and training
│   ├── adapters/     # Model adapter interfaces (OpenVLA, ACT, π0)
│   ├── training/     # Training scripts + configs
│   └── checkpoints/  # Trained model weights
├── data/             # Dataset storage
│   ├── raw/          # Raw HDF5 trajectory data
│   └── processed/    # Model-specific formatted datasets
├── scripts/          # Standalone utility scripts
├── tests/            # Test suite
└── doc/              # Documentation
    ├── project_design.md
    └── step2step/    # Step-by-step implementation guide (20 steps)
```

### Action Space

7D Cartesian delta + gripper:
```
(dx, dy, dz, droll, dpitch, dyaw, gripper_open)
```

- `dx, dy, dz`: position delta (meters, ±0.02m)
- `droll, dpitch, dyaw`: rotation delta (radians, ±0.1rad)
- `gripper_open`: 0.0 = closed, 1.0 = open

Internal IK (damped least-squares Jacobian) converts Cartesian deltas to joint targets.

## Development

### Branch Strategy
- `main` — stable, runnable versions only
- `dev/sim-core` — active development branch

### Steps
See `doc/step2step/README.md` for the 20-step implementation plan.

## License
TBD
