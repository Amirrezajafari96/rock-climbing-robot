#!/usr/bin/env python3
"""
Run a full climbing simulation.

Usage:
    python scripts/run_simulation.py --config configs/simulation.yaml
    python scripts/run_simulation.py --config configs/simulation.yaml --render
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from climbing_robot.planner.base_planner import GripperID
from climbing_robot.planner.climbing_planner import ClimbingPlanner
from climbing_robot.robot.model import RobotModel
from climbing_robot.simulators.mujoco_sim import MuJoCoSimulator
from climbing_robot.simulators.base import SimulatorConfig
from climbing_robot.stability.com_checker import COMStabilityChecker
from climbing_robot.utils import load_config, setup_logging
from climbing_robot.wall.hold_system import HoldSystem, WallConfig


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rock Climbing Robot Simulation")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load configs
    sim_cfg = load_config(args.config)
    wall_cfg = load_config("configs/wall.yaml")
    planner_cfg = load_config("configs/planner.yaml")

    # Build subsystems
    wall_config = WallConfig(
        n_rows=wall_cfg["wall"]["n_rows"],
        n_cols=wall_cfg["wall"]["n_cols"],
        row_spacing=wall_cfg["wall"]["row_spacing"],
        base_height=wall_cfg["wall"]["base_height"],
    )
    holds = HoldSystem(wall_config)
    stability = COMStabilityChecker()
    planner = ClimbingPlanner(
        holds,
        stability,
        max_reach=planner_cfg["planner"]["max_reach"],
        n_traj_steps=planner_cfg["planner"]["n_traj_steps"],
    )

    # Build simulator
    s = sim_cfg["simulation"]
    config = SimulatorConfig(
        model_path=s["model_path"],
        timestep=s["timestep"],
        gravity=tuple(s["gravity"]),
        render=args.render or s.get("render", False),
        seed=s.get("seed", 42),
    )

    try:
        sim = MuJoCoSimulator(config)
    except (ImportError, FileNotFoundError) as e:
        logger.error("Could not start simulator: %s", e)
        logger.info("Tip: Run `pip install mujoco` and ensure model_path is correct.")
        sys.exit(1)

    robot = RobotModel(sim)
    state = sim.reset()

    # Initial grips from config
    init_grips = s.get("initial_grips", {})
    active_grips: dict[GripperID, str | None] = {
        GripperID.LEFT:   init_grips.get("left"),
        GripperID.RIGHT:  init_grips.get("right"),
        GripperID.CENTER: init_grips.get("center"),
    }

    goal_height = s.get("goal_height", 1.80)
    logger.info("Starting climb. Goal height: %.2fm", goal_height)

    # Mark initial holds as occupied
    for grip_id in active_grips.values():
        if grip_id:
            holds.set_active(grip_id, True)

    # Main planning + execution loop
    total_moves = 0
    while True:
        com = robot.compute_com()
        logger.info("Step %d | COM z=%.3f | Goal z=%.3f",
                    sim.step_count, com[2], goal_height)

        result = planner.plan(active_grips, com, goal_height)
        if not result.success or not result.moves:
            logger.warning("Planner returned no valid moves. Stopping.")
            break

        for move in result.moves:
            # Execute joint trajectory
            for q_step in move.joint_traj:
                robot.set_limb_joints(move.gripper, q_step[:3])
                sim.step()

            # Update grip state
            if move.target_hold:
                old_hold = active_grips[move.gripper]
                if old_hold:
                    holds.set_active(old_hold, False)
                active_grips[move.gripper] = move.target_hold
                holds.set_active(move.target_hold, True)
                total_moves += 1
                logger.info(
                    "  [%s] → %s (stability margin=%.3fm)",
                    move.gripper.value, move.target_hold, move.stability_margin,
                )

        # Check goal
        com = robot.compute_com()
        if com[2] >= goal_height:
            logger.info("🎉 Goal reached! COM z=%.3fm after %d moves.", com[2], total_moves)
            break

        if sim.step_count > s.get("max_steps", 100000):
            logger.warning("Max simulation steps reached.")
            break

    sim.close()
    logger.info("Simulation complete. Total moves: %d", total_moves)


if __name__ == "__main__":
    main()
