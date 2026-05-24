"""
MuJoCo simulator backend.

Wraps the mujoco Python bindings behind the BaseSimulator interface.
All MuJoCo-specific calls are isolated here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from climbing_robot.simulators.base import BaseSimulator, SimulatorConfig, SimulatorState

logger = logging.getLogger(__name__)


class MuJoCoSimulator(BaseSimulator):
    """
    MuJoCo 3.x physics backend.

    Args:
        config: SimulatorConfig with model_path pointing to an MJCF XML file.

    Example::

        config = SimulatorConfig(model_path="assets/models/scene.xml", render=True)
        sim = MuJoCoSimulator(config)
        state = sim.reset()
        for ctrl in trajectory:
            sim.set_control(ctrl)
            state = sim.step()
        sim.close()
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._mj: Any = None     # mujoco module (lazy import)
        self._model: Any = None  # MjModel
        self._data: Any = None   # MjData
        self._renderer: Any = None
        self._load_model()

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Import mujoco and load the MJCF model."""
        try:
            import mujoco  # noqa: PLC0415
            self._mj = mujoco
        except ImportError as e:
            raise ImportError(
                "mujoco is not installed. Run: pip install mujoco"
            ) from e

        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MJCF model not found: {model_path}")

        self._model = self._mj.MjModel.from_xml_path(str(model_path))
        self._data = self._mj.MjData(self._model)

        # Apply config overrides
        self._model.opt.timestep = self._config.timestep
        self._model.opt.gravity[:] = self._config.gravity

        logger.info("MuJoCo model loaded: %s", model_path)
        logger.info(
            "  nq=%d  nv=%d  nu=%d  nsensor=%d",
            self._model.nq, self._model.nv,
            self._model.nu, self._model.nsensor,
        )

    def _snapshot(self) -> SimulatorState:
        """Build a SimulatorState from current MjData."""
        d, m = self._data, self._model
        body_pos = {}
        body_ori = {}
        for i in range(m.nbody):
            name = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_BODY, i)
            if name:
                body_pos[name] = d.xpos[i].copy()
                body_ori[name] = d.xmat[i].reshape(3, 3).copy()

        return SimulatorState(
            time=float(d.time),
            qpos=d.qpos.copy(),
            qvel=d.qvel.copy(),
            ctrl=d.ctrl.copy(),
            sensor_data=d.sensordata.copy(),
            body_positions=body_pos,
            body_orientations=body_ori,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> SimulatorState:
        self._mj.mj_resetData(self._model, self._data)
        self._mj.mj_forward(self._model, self._data)
        self._step_count = 0
        logger.debug("Simulator reset at t=0")
        return self._snapshot()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        logger.debug("MuJoCo simulator closed")

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self) -> SimulatorState:
        self._mj.mj_step(self._model, self._data)
        self._step_count += 1
        return self._snapshot()

    def set_control(self, ctrl: np.ndarray) -> None:
        if len(ctrl) != self._model.nu:
            raise ValueError(
                f"ctrl has {len(ctrl)} elements, model has {self._model.nu} actuators"
            )
        self._data.ctrl[:] = ctrl

    # ── State access ──────────────────────────────────────────────────────

    def get_state(self) -> SimulatorState:
        return self._snapshot()

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._mj.mj_forward(self._model, self._data)

    # ── Geometry queries ──────────────────────────────────────────────────

    def _body_id(self, name: str) -> int:
        bid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body not found: '{name}'")
        return bid

    def _site_id(self, name: str) -> int:
        sid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"Site not found: '{name}'")
        return sid

    def _sensor_id(self, name: str) -> int:
        sid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise KeyError(f"Sensor not found: '{name}'")
        return sid

    def get_body_position(self, body_name: str) -> np.ndarray:
        return self._data.xpos[self._body_id(body_name)].copy()

    def get_body_orientation(self, body_name: str) -> np.ndarray:
        return self._data.xmat[self._body_id(body_name)].reshape(3, 3).copy()

    def get_site_position(self, site_name: str) -> np.ndarray:
        return self._data.site_xpos[self._site_id(site_name)].copy()

    def get_sensor(self, sensor_name: str) -> np.ndarray:
        sid = self._sensor_id(sensor_name)
        adr = self._model.sensor_adr[sid]
        dim = self._model.sensor_dim[sid]
        return self._data.sensordata[adr : adr + dim].copy()

    # ── Rendering ────────────────────────────────────────────────────────

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray | None:
        """Return HxWx3 uint8 RGB frame using offscreen renderer."""
        if not self._config.render:
            return None
        try:
            if self._renderer is None:
                self._renderer = self._mj.Renderer(self._model, height=height, width=width)
            self._renderer.update_scene(self._data)
            return self._renderer.render()
        except Exception as exc:
            logger.warning("render_frame failed: %s", exc)
            return None

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def model(self) -> Any:
        """Direct access to MjModel (MuJoCo-specific use only)."""
        return self._model

    @property
    def data(self) -> Any:
        """Direct access to MjData (MuJoCo-specific use only)."""
        return self._data

    @property
    def n_actuators(self) -> int:
        return int(self._model.nu)

    @property
    def n_joints(self) -> int:
        return int(self._model.nq)
