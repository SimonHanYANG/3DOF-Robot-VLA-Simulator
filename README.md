# 3DOF Robot VLA Simulator

3DOF 推物避障仿真平台 — 仿真、数据采集、模型训练、推理与可视化一体化平台。

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
├── sim/              # Simulation core (MuJoCo 3DOF arm, scene, controller)
│   ├── assets/       # MuJoCo XML scene files
│   └── planner/      # RRT path planner
├── backend/          # FastAPI + WebSocket server
│   └── services/     # Business logic
├── frontend/         # React + Three.js 3D UI
├── models/           # VLA model adapters and training
│   ├── adapters/     # Model adapter interfaces
│   ├── training/     # Training scripts
│   └── checkpoints/  # Trained model weights
├── data/             # Dataset storage
│   ├── raw/          # Raw HDF5 trajectory data
│   └── processed/    # Model-specific formatted datasets
├── scripts/          # Standalone utility scripts
├── tests/            # Test suite
└── doc/              # Documentation
    ├── project_design.md
    └── step2step/    # Step-by-step implementation guide
```

## Development

### Branch Strategy
- `main` — stable, runnable versions only
- `dev/sim-core` — active development branch

### Steps
See `doc/step2step/README.md` for the 20-step implementation plan.

## License
TBD
