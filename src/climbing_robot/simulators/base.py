"""
Abstract simulator interface.

Design principle: ALL physics backends must implement BaseSimulator.
This keeps kinematics, planning, and control code simulator-agnostic.
To migrate from MuJoCo to PyBullet/Unity, only this layer changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulatorState:
    """Snapshot of the full simulator state at a given timestep."""

    time: float
    qpos: np.ndarray          # joint positions  [nq]
    qvel: np.ndarray          # joint velocities [nv]
    ctrl: np.ndarray          # actuator controls [nu]
    sensor_data: np.ndarray   # raw sensor readings [nsensor]
    body_positions: dict[str, np.ndarray] = field(default_factory=dict)
    body_orientations: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    """Configuration passed to a simulator on construction."""

    model_path: str
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    max_steps: int = 100_000
    render: bool = False
    seed: int = 42


class BaseSimulator(abc.ABC):
    """
    Abstract physics simulator.

    Concrete subclasses: MuJoCoSimulator, PyBulletSimulator (future).

    Usage::

        sim = MuJoCoSimulator(config)
        sim.reset()
        for _ in range(1000):
            sim.set_control(ctrl)
            state = sim.step()
        sim.close()
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config
        self._step_count: int = 0

    @property
    def config(self) -> SimulatorConfig:
        return self._config

    @property
    def step_count(self) -> int:
        return self._step_count

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def reset(self) -> SimulatorState:
        """Reset to initial state and return it."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources (renderer, shared memory, etc.)."""

    # ── Simulation step ───────────────────────────────────────────────────

    @abc.abstractmethod
    def step(self) -> SimulatorState:
        """Advance physics by one timestep and return new state."""

    @abc.abstractmethod
    def set_control(self, ctrl: np.ndarray) -> None:
        """Write actuator control vector before the next step."""

    # ── State access ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_state(self) -> SimulatorState:
        """Return current state without advancing physics."""

    @abc.abstractmethod
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Teleport robot to a given configuration (for planning)."""

    # ── Geometry queries ──────────────────────────────────────────────────

    @abc.abstractmethod
    def get_body_position(self, body_name: str) -> np.ndarray:
        """World-frame position [3] of named body."""

    @abc.abstractmethod
    def get_body_orientation(self, body_name: str) -> np.ndarray:
        """World-frame rotation matrix [3,3] of named body."""

    @abc.abstractmethod
    def get_site_position(self, site_name: str) -> np.ndarray:
        """World-frame position [3] of named MuJoCo site / marker."""

    @abc.abstractmethod
    def get_sensor(self, sensor_name: str) -> np.ndarray:
        """Return sensor reading by name."""

    # ── Rendering ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def render_frame(self) -> np.ndarray | None:
        """Render and return an HxWx3 uint8 RGB image, or None."""

    # ── Utilities ────────────────────────────────────────────────────────

    def run_steps(
        self,
        ctrl_sequence: list[np.ndarray],
        *,
        callback: Any | None = None,
    ) -> list[SimulatorState]:
        """
        Execute a sequence of controls, collecting states.

        Args:
            ctrl_sequence: List of control vectors, one per step.
            callback: Optional callable(step, state) for logging/vis.

        Returns:
            List of SimulatorState, one per step.
        """
        states: list[SimulatorState] = []
        for i, ctrl in enumerate(ctrl_sequence):
            self.set_control(ctrl)
            state = self.step()
            states.append(state)
            if callback is not None:
                callback(i, state)
        return states
