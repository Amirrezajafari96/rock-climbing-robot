# Experiment 001 — Basic Inchworm Climb

## Objective

Demonstrate the baseline greedy climbing planner navigating a regular 3×6 hold grid
from start height z=0.30m to goal height z=1.80m.

## Setup

- **Wall**: 6 rows × 3 columns, 35cm row spacing, starts at z=0.30m
- **Robot**: 3-limbed, upper arm 25cm, lower arm 20cm
- **Planner**: Greedy inchworm — always moves the lowest gripper upward
- **Stability**: COM must project inside support polygon after each move

## Running

```bash
python scripts/run_planner.py \
  --planner configs/planner.yaml \
  --goal-height 1.8 \
  --save-plot experiments/exp_001_basic_climb/wall_plot.png
```

## Expected Behavior

1. Robot starts with all 3 grippers at row 0 (z≈0.30)
2. Center arm (lowest or first to move) reaches to row 1
3. Left/right arms follow in alternating fashion
4. Robot progressively advances to z=1.80

## Results Log

| Date | n_moves | Height | Success | Notes |
|------|---------|--------|---------|-------|
| 2026-05-23 | - | - | - | Initial setup |
