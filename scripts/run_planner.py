#!/usr/bin/env python3
"""
Run the climbing planner WITHOUT simulation (pure planning mode).
Useful for rapid algorithm development and visualization.

Usage:
    python scripts/run_planner.py --wall configs/wall.yaml --robot configs/robot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from climbing_robot.planner.base_planner import GripperID
from climbing_robot.planner.climbing_planner import ClimbingPlanner
from climbing_robot.stability.com_checker import COMStabilityChecker
from climbing_robot.utils import load_config, setup_logging
from climbing_robot.visualization.planner_viz import plot_wall_overview
from climbing_robot.wall.hold_system import HoldSystem, WallConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Rock Climbing Planner (no sim)")
    parser.add_argument("--wall",   default="configs/wall.yaml")
    parser.add_argument("--robot",  default="configs/robot.yaml")
    parser.add_argument("--planner", default="configs/planner.yaml")
    parser.add_argument("--goal-height", type=float, default=1.80)
    parser.add_argument("--save-plot", default=None, help="Save PNG plot to path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    wall_cfg = load_config(args.wall)["wall"]
    planner_cfg = load_config(args.planner)["planner"]

    holds = HoldSystem(WallConfig(
        n_rows=wall_cfg["n_rows"],
        n_cols=wall_cfg["n_cols"],
        row_spacing=wall_cfg["row_spacing"],
        base_height=wall_cfg["base_height"],
    ))

    stability = COMStabilityChecker()
    planner = ClimbingPlanner(
        holds, stability,
        max_reach=planner_cfg["max_reach"],
        n_traj_steps=planner_cfg["n_traj_steps"],
    )

    # Initial state
    active_grips = {
        GripperID.LEFT:   "r0_c0",
        GripperID.RIGHT:  "r0_c2",
        GripperID.CENTER: "r0_c1",
    }
    com = np.array([0.0, -0.08, 0.35])

    print(f"\nPlanning climb to z={args.goal_height}m...")
    result = planner.plan(active_grips, com, args.goal_height)

    if result.success:
        print(f"\n✅ Plan found: {result.n_moves} moves, height gain = {result.total_height_gain:.2f}m")
        for i, move in enumerate(result.moves):
            print(f"  Move {i+1}: [{move.gripper.value}] → {move.target_hold}"
                  f"  (margin={move.stability_margin:.3f}m, "
                  f"dur={move.estimated_duration:.1f}s)")
    else:
        print(f"\n❌ Planning failed: {result.message}")

    # Visualize
    active_hold_ids = [hid for hid in active_grips.values() if hid]
    plot_wall_overview(
        holds,
        active_grip_ids=active_hold_ids,
        com_xz=(com[0], com[2]),
        planned_moves=result.moves,
        title=f"Planned Climb (goal z={args.goal_height}m)",
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
