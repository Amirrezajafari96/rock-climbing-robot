# Setup Guide

## Prerequisites

- Python 3.10 or higher
- pip or [uv](https://github.com/astral-sh/uv)

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/rock-climbing-robot.git
cd rock-climbing-robot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify installation
python -c "from climbing_robot.kinematics import FKSolver; print('OK')"
```

## Running Tests

```bash
# All tests
pytest

# Only unit tests (fast, no simulator)
pytest tests/unit/ -m unit

# With coverage
pytest --cov=src/climbing_robot --cov-report=html

# Parallel (faster on multi-core)
pytest -n auto
```

## Launching Simulation

```bash
# Pure planning (no MuJoCo required)
python scripts/run_planner.py --goal-height 1.5

# Full MuJoCo simulation (requires mujoco installed)
python scripts/run_simulation.py --config configs/simulation.yaml
```

## MuJoCo Viewer

To inspect the MJCF model directly:

```bash
pip install mujoco
python -m mujoco.viewer --mjcf=assets/models/scene.xml
```

## Environment Variables

| Variable              | Default | Description |
|-----------------------|---------|-------------|
| `CLIMBING_LOG_LEVEL`  | INFO    | Logging verbosity |
| `CLIMBING_SEED`       | 42      | Global random seed |
