# Architecture — Rock Climbing Robot

## Overview

The repository is organized around a **simulator-agnostic core**: all planning,
kinematics, and stability logic is isolated from the physics backend.

```
┌─────────────────────────────────────────────────────────┐
│                        Scripts / CLI                     │
│         run_simulation.py    run_planner.py              │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    RobotModel (robot/)                   │
│    Translates gripper/limb semantics to actuator signals │
└──────────┬───────────────────────────────┬──────────────┘
           │                               │
┌──────────▼──────────┐    ┌──────────────▼─────────────┐
│   ClimbingPlanner   │    │     BaseSimulator           │
│   (planner/)        │    │     (simulators/base.py)    │
│                     │    │                             │
│  A* hold selection  │    │   MuJoCoSimulator           │
│  IK trajectory gen  │    │   (simulators/mujoco_sim)   │
│  stability verify   │    │                             │
└──────────┬──────────┘    └─────────────────────────────┘
           │
    ┌──────┴──────────────┐
    │                     │
┌───▼──────┐    ┌────────▼──────┐    ┌─────────────────┐
│ IKSolver │    │ HoldSystem    │    │ COMStability    │
│ FKSolver │    │ (wall/)       │    │ Checker         │
│(kinemat.)│    │               │    │ (stability/)    │
└──────────┘    └───────────────┘    └─────────────────┘
```

## Key Design Decisions

### 1. Simulator Abstraction (`simulators/base.py`)

`BaseSimulator` defines the minimum contract:
- `reset() → SimulatorState`
- `step() → SimulatorState`
- `set_control(ctrl)`
- `get_body_position(name) → ndarray`
- `get_site_position(name) → ndarray`

To migrate to PyBullet: implement `BaseSimulator`, swap `MuJoCoSimulator` in configs.

### 2. Kinematics Module (`kinematics/`)

- **FK**: DH-parameter chain → 4x4 homogeneous transforms
- **IK**: Jacobian pseudoinverse with Tikhonov damping + null-space projection
- **No simulator dependency**: pure NumPy math

### 3. Wall as a Data Layer (`wall/`)

The wall holds are a static data structure (dict of `Hold` objects).
The planner queries it via spatial methods (`reachable_from`, `nearest_hold`).
This separates geometry from physics.

### 4. Stability (`stability/`)

COM computation is either:
- **Analytical**: `compute_com(positions, masses)` from body list
- **From simulator**: `check_from_simulator(sim, body_names, masses, sites)`

Support polygon is the 2D convex hull of active grip points projected onto
the XZ wall plane. `scipy.spatial.ConvexHull` handles the geometry.

### 5. Planner (`planner/`)

`ClimbingPlanner` uses a greedy inchworm strategy:
1. Pick the lowest gripper
2. Score candidate holds: `score = z * 2.0 - lateral_deviation * 0.5`
3. Verify IK convergence
4. Verify COM stability (must be positive margin)
5. Emit `ClimbMove` with joint trajectory

For research extensions: swap in `BasePlanner` subclass using RRT or RL.

## Module Dependency Rules

```
scripts → robot → simulators → (external: mujoco)
scripts → planner → kinematics, wall, stability
scripts → utils

# No upward imports:
kinematics → ONLY numpy, scipy
wall       → ONLY numpy
stability  → ONLY numpy, scipy
```

## Configuration System

All tunable parameters live in `configs/*.yaml`.
Code reads config at startup via `utils.load_config()`.
No magic numbers embedded in source.

## Testing Layers

| Layer       | Location                 | Speed   | Requires MuJoCo |
|-------------|--------------------------|---------|-----------------|
| Unit        | tests/unit/              | Fast    | No              |
| Integration | tests/integration/       | Medium  | No              |
| Simulation  | tests/simulation/        | Slow    | Yes             |
