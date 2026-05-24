# Developer Guide

## Code Style

- **Formatter**: `black --line-length 100`
- **Linter**: `ruff` with `E, F, W, I, UP, B` rules
- **Type checker**: `mypy` (strict mode for new modules)
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`)

Run all checks:
```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/
```

## Git Workflow

```
main      ← stable releases (tagged)
develop   ← integration
feature/* ← new features (PR → develop)
fix/*     ← bug fixes (PR → develop or main)
```

### Commit Examples

```
feat: add IK null-space projection for joint limit avoidance
fix: correct sign error in support polygon margin calculation
refactor: extract DH transform into transforms.py module
test: add FK round-trip tests for all joint configurations
docs: add architecture diagram to docs/architecture.md
chore: upgrade scipy to 1.12
```

## Adding a New Simulator Backend

1. Create `src/climbing_robot/simulators/my_sim.py`
2. Subclass `BaseSimulator` and implement all abstract methods
3. Add a `SimulatorConfig` entry in `configs/simulation.yaml`
4. Write integration tests in `tests/simulation/test_my_sim.py`

## Adding a New Planner

1. Create `src/climbing_robot/planner/my_planner.py`
2. Subclass `BasePlanner` and implement `plan()` and `reset()`
3. The planner receives `HoldSystem` and `COMStabilityChecker` in `__init__`
4. Return a `PlanResult` with `ClimbMove` list

## Experiment Workflow

```bash
# Create experiment directory
mkdir experiments/exp_002_rrt_planner
cp experiments/exp_001_basic_climb/config.yaml experiments/exp_002_rrt_planner/

# Edit config, run, log results
python scripts/run_planner.py \
  --planner configs/planner.yaml \
  --save-plot experiments/exp_002_rrt_planner/wall_plot.png
```
