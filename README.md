<<<<<<< HEAD
# 🧗 Rock Climbing Robot — MuJoCo Simulation

> Research-quality robotics simulation for an autonomous wall-climbing robot.  
> Built with MuJoCo, modular Python architecture, and professional engineering practices.

[![CI](https://github.com/your-org/rock-climbing-robot/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rock-climbing-robot/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/simulator-MuJoCo-green.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project simulates a **3-limbed inchworm-style rock climbing robot** that navigates a
vertical wall by gripping holds, computing stable configurations, and planning motion
sequences autonomously.

The robot design is based on the professor's model: a U-shaped frame (torso) with
spherical gripper ends on each limb. The center arm is actuated and can reach new holds
while the outer arms maintain grip. The architecture is **simulator-agnostic** — MuJoCo
is the primary backend, but the abstraction layer allows migration to PyBullet, Unity,
or Godot without changing planner or kinematics code.

### How the Robot Climbs

1. Two outer limbs grip holds on the wall
2. The planner computes a reachable, stable target hold for the free limb  
3. IK solves a joint trajectory to reach that hold
4. COM stability is verified against the support polygon
5. The free limb moves and grips; the cycle repeats upward

---

## Repository Structure

```
rock-climbing-robot/
├── src/climbing_robot/         # Core Python package
│   ├── robot/                  # Robot model abstraction & config
│   ├── kinematics/             # FK (DH params), IK (Jacobian), SE3
│   ├── wall/                   # Wall geometry, hold system, reachability
│   ├── stability/              # COM computation & support polygon
│   ├── planner/                # Autonomous climbing planner (A*)
│   ├── simulators/             # MuJoCo backend + abstract interface
│   ├── visualization/          # Debug overlays, trajectory plots
│   └── utils/                  # Logging, YAML config, math helpers
├── assets/models/              # MuJoCo MJCF XML files
├── configs/                    # YAML configuration files
├── tests/                      # pytest test suite (unit/integration/sim)
├── scripts/                    # run_simulation.py, run_planner.py, etc.
├── experiments/                # Reproducible experiment logs
├── docs/                       # Architecture, setup, developer guides
└── .github/workflows/          # CI/CD (lint, type-check, tests)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or [uv](https://github.com/astral-sh/uv) (recommended)
- MuJoCo 3.x (`pip install mujoco`)

### 1. Clone and set up

```bash
git clone https://github.com/your-org/rock-climbing-robot.git
cd rock-climbing-robot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Run the test suite

```bash
pytest tests/ -v
```

### 3. Launch MuJoCo simulation

```bash
python scripts/run_simulation.py --config configs/simulation.yaml
```

### 4. Run autonomous climbing planner

```bash
python scripts/run_planner.py --wall configs/wall.yaml --robot configs/robot.yaml
```

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Repo Setup | ✅ | Structure, CI, pyproject.toml |
| 2. MuJoCo Integration | ✅ | Simulator abstraction layer |
| 3. Robot Model (MJCF) | ✅ | Joints, actuators, grippers |
| 4. Wall & Hold System | ✅ | Grid layout, reachability checks |
| 5. Forward Kinematics | ✅ | DH-parameter FK solver |
| 6. Inverse Kinematics | ✅ | Jacobian pseudoinverse IK |
| 7. Stability Checker | ✅ | COM + support polygon |
| 8. Climbing Planner | ✅ | A*-guided hold selection |
| 9. Visualization | ✅ | Debug overlays, trajectory plots |
| 10. Experiments | 🔄 | Benchmark configurations |

---

## Git Workflow

```
main      ← stable, tagged releases (v0.1.0, v0.2.0, ...)
develop   ← integration branch
feature/* ← one feature per branch (merged via PR)
fix/*     ← bug fixes
```

### Commit Convention (Conventional Commits)

```
feat: add mujoco robot loader
fix: correct COM stability calculation
refactor: separate IK solver abstraction
test: add forward kinematics unit tests
docs: update architecture diagram
chore: bump mujoco to 3.2.x
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full simulator-agnostic design.

Key interfaces:
- `BaseSimulator` — physics backend contract (step, reset, get_state)
- `RobotModel` — joint/body/actuator abstraction  
- `BasePlanner` — motion planning contract (plan, execute)
- `IKSolver` — end-effector positioning via Jacobian

---

## Academic Usage

Structured for university submission and portfolio use:

- Reproducible experiments via YAML configs + random seeds
- Deterministic simulation with fixed `mjModel.opt.timestep`
- Experiment logs with timestamps and parameter hashes
- Clean separation of concerns for paper references

---

## License

MIT License — see [LICENSE](LICENSE).
=======
# rock-climbing-robot
>>>>>>> 99a7d74c549392cd8acdd954da250d97adbc3f13
