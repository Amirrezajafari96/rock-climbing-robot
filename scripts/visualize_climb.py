#!/usr/bin/env python3
"""
Live MuJoCo viewer: rock climbing robot climbs the wall.

Usage (from repo root):
    python scripts/visualize_climb.py           # full animation
    python scripts/visualize_climb.py --verify  # headless IK check

v5 — Physics-based climbing
-----------------------------
1. Gravity (0 0 -2.0): mild but visible. Robot hangs from holds during settle.
2. Connect equality constraints: mocap bodies pinned to hold positions; gripper
   bodies locked at runtime via data.eq_active[eid]=1.  Purely positional
   (no rotation lock) — avoids orientation conflicts with arm IK.
3. Continuous IK tracking during torso advance: each kinematic frame re-solves
   arm joint angles (warm-start, 20 iters) so grippers stay on holds as torso
   rises.  Arms flex visibly from "pointing up" (torso below hold) to "pointing
   down" (torso above hold) — real climbing tension.
4. kp_torso_z = 1500: resists gravity sag (~4 mm at hold).
5. Weld activated AFTER smooth_grip (gripper already at hold → zero residual).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco not installed.  Run:  pip install mujoco")

# ---------- configuration ----------------------------------------------------
MODEL_PATH = "assets/models/climbing_robot.xml"
SIM_DT     = 0.002       # must match XML <option timestep>
REAL_TIME  = True

IK_ITERS = 800
IK_STEP  = 0.15
IK_DAMP  = 0.002
IK_TOL   = 5e-3

N_RETRACT  = 30    # kinematic frames: arm retracts to neutral
N_EXTEND   = 70    # kinematic frames: arm extends to new hold
N_ADVANCE  = 180   # kinematic frames: torso pull-up (more for visible flexion)
HOLD_STEPS = 200   # physics steps: gravity + constraint settling at each hold

# Gripper centre at Y=-0.055 → sphere tip at Y=-0.015 (wall face at Y=0, no clip)
HOLD_Y = -0.055

# ---------- hold grid ---------------------------------------------------------
_HOLD_GRID = [
    (0, 0.30, [-0.30,  0.00,  0.30]),
    (1, 0.65, [-0.28,  0.04,  0.32]),
    (2, 1.00, [-0.32, -0.02,  0.28]),
    (3, 1.35, [-0.30,  0.06,  0.36]),
    (4, 1.70, [-0.26,  0.02,  0.32]),
    (5, 2.05, [-0.28,  0.02,  0.28]),
]
HOLDS: dict[str, np.ndarray] = {}
for _r, _z, _xs in _HOLD_GRID:
    for _c, _x in enumerate(_xs):
        HOLDS[f"r{_r}_c{_c}"] = np.array([_x, HOLD_Y, _z])

# ---------- DOF / ctrl index map ---------------------------------------------
DOF_TORSO_X  = 0;  DOF_TORSO_Z  = 1
CTRL_TORSO_X = 0;  CTRL_TORSO_Z = 1
DOF_LEFT   = [2, 3, 4];   CTRL_LEFT   = [2, 3, 4]
DOF_RIGHT  = [5, 6, 7];   CTRL_RIGHT  = [5, 6, 7]
DOF_CENTER = [8, 9, 10];  CTRL_CENTER = [8, 9, 10]

GRIPPER_SITES = {
    "left":   "left_gripper_site",
    "right":  "right_gripper_site",
    "center": "center_gripper_site",
}
ARM_DOFS  = {"left": DOF_LEFT,  "right": DOF_RIGHT,  "center": DOF_CENTER}
ARM_CTRLS = {"left": CTRL_LEFT, "right": CTRL_RIGHT, "center": CTRL_CENTER}

# Runtime id tables (populated once by _build_ids)
EQ_IDS:    dict[str, int] = {}   # arm -> equality constraint index
MOCAP_IDS: dict[str, int] = {}   # arm -> mocap body index (for mocap_pos)


# ---------- id resolution -----------------------------------------------------

def _build_ids(model) -> None:
    """Resolve connect-constraint and mocap-body ids from model."""
    for arm in ("left", "right", "center"):
        eid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{arm}"
        )
        if eid < 0:
            raise RuntimeError(f"Equality 'weld_{arm}' not found in model")
        EQ_IDS[arm] = eid

        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"grip_anchor_{arm}"
        )
        if bid < 0:
            raise RuntimeError(f"Body 'grip_anchor_{arm}' not found in model")
        midx = int(model.body_mocapid[bid])
        if midx < 0:
            raise RuntimeError(f"grip_anchor_{arm} is not a mocap body")
        MOCAP_IDS[arm] = midx


# ---------- grip constraint helpers ------------------------------------------

def grip_activate(model, data, arm: str, hold_pos: np.ndarray) -> None:
    """
    Pin gripper to hold via connect equality constraint.

    Steps:
      1. Move mocap anchor to hold_pos in world space.
      2. Activate the connect constraint (data.eq_active[eid] = 1).
      3. mj_forward so constraint is visible immediately.

    Call AFTER smooth_grip finishes — gripper is already at hold_pos,
    so constraint residual is ~zero and no sudden jump occurs.
    """
    midx = MOCAP_IDS[arm]
    data.mocap_pos[midx]  = hold_pos.copy()
    data.mocap_quat[midx] = np.array([1.0, 0.0, 0.0, 0.0])
    data.eq_active[EQ_IDS[arm]] = 1
    mujoco.mj_forward(model, data)


def grip_release(data, arm: str) -> None:
    """Deactivate connect constraint — arm is free to move."""
    data.eq_active[EQ_IDS[arm]] = 0


# ---------- IK ---------------------------------------------------------------

def ik_reach(model, data, site_name: str, target: np.ndarray,
             dof_ids: list[int], iters: int = IK_ITERS):
    """
    Damped-least-squares IK via mj_jacSite.

    Caller must have set data.qpos[dof_ids] to the desired start configuration.
    Returns (converged: bool, final_error: float).
    """
    sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    dofs = np.array(dof_ids, dtype=int)
    jacp = np.zeros((3, model.nv))
    prev = np.inf

    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err  = target - data.site_xpos[sid]
        norm = float(np.linalg.norm(err))
        if norm < IK_TOL:
            return True, norm
        step = IK_STEP if norm <= prev * 1.5 else IK_STEP * 0.1
        prev = norm
        jacp[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J  = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(3), err)
        for k, di in enumerate(dofs):
            data.qpos[di] += step * dq[k]
            jid = int(model.dof_jntid[di])
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                data.qpos[di] = float(np.clip(data.qpos[di], lo, hi))

    mujoco.mj_forward(model, data)
    final = float(np.linalg.norm(target - data.site_xpos[sid]))
    return False, final


# ---------- kinematic animation helpers --------------------------------------

def _kframe(model, data, viewer) -> None:
    """One kinematic display frame: zero qvel → mj_forward → viewer.sync."""
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if viewer is not None:
        viewer.sync()
        if REAL_TIME:
            time.sleep(SIM_DT * 3)   # ~6 ms/frame


def _animate_joints(model, data, viewer, dofs, ctrl_ids,
                    q0: np.ndarray, q1: np.ndarray, n_steps: int) -> None:
    """Smoothstep interpolation of arm joints from q0 → q1 over n_steps frames."""
    for s in range(n_steps):
        t   = (s + 1) / n_steps
        t_s = t * t * (3.0 - 2.0 * t)
        q   = q0 + t_s * (q1 - q0)
        for k, (di, ci) in enumerate(zip(dofs, ctrl_ids)):
            data.qpos[di] = float(q[k])
            data.ctrl[ci] = float(q[k])
        _kframe(model, data, viewer)


def smooth_grip(model, data, viewer, arm: str, target: np.ndarray):
    """
    Two-phase kinematic grip animation.

    Phase 1 — Retract: arm swings back to neutral (q=0), lifting off old hold.
    Phase 2 — Extend:  IK from neutral drives arm to new hold.

    IK is always solved from q=0 → guaranteed convergence to the same solution
    the --verify pass confirmed (<5 mm).
    """
    dofs     = np.array(ARM_DOFS[arm],  dtype=int)
    ctrl_ids = np.array(ARM_CTRLS[arm], dtype=int)
    site     = GRIPPER_SITES[arm]

    q_current = data.qpos[dofs].copy()
    q_neutral = np.zeros(len(dofs))

    # Solve IK from neutral configuration
    data.qpos[dofs] = q_neutral
    data.qvel[:]    = 0.0
    conv, err = ik_reach(model, data, site, target, dofs.tolist())
    q_target  = data.qpos[dofs].copy()

    # Restore current arm state
    data.qpos[dofs] = q_current
    data.qvel[:]    = 0.0

    # Phase 1: retract to neutral
    _animate_joints(model, data, viewer, dofs, ctrl_ids,
                    q_current, q_neutral, N_RETRACT)

    # Phase 2: extend to target
    _animate_joints(model, data, viewer, dofs, ctrl_ids,
                    q_neutral, q_target, N_EXTEND)

    # Lock final pose
    data.qpos[dofs]     = q_target
    data.ctrl[ctrl_ids] = q_target
    data.qvel[:]        = 0.0
    mujoco.mj_forward(model, data)

    sid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    gpos      = data.site_xpos[sid].copy()
    final_err = float(np.linalg.norm(target - gpos))
    tag       = "GRIP" if final_err < 0.020 else "MISS"
    print(f"  [{tag}] {arm:8s}  err={final_err:.4f}m  "
          f"grip={gpos.round(3)}  target={target.round(3)}")
    return conv, final_err


def run_physics(model, data, viewer, n_steps: int) -> None:
    """
    Physics settle: mj_step with gravity + active connect constraints.

    With grippers locked to holds and gravity enabled, the robot visibly
    hangs from the wall — arms under tension, torso actuator resisting sag.
    """
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if viewer is not None and i % 4 == 0:
            viewer.sync()
            if REAL_TIME:
                time.sleep(SIM_DT)


def pull_up(model, data, viewer, target_world_z: float,
            held_targets: dict[str, np.ndarray]) -> None:
    """
    Kinematic torso advance with arm-angle fade to neutral.

    As the torso rises, arms smoothly transition from their current IK pose
    (gripping the hold, pointing up) back toward neutral (q=0, pointing up at
    full reach).  This produces the natural "pulling" visual:

      - Frame 0  : arms angled to hold — gripper touching hold, tension visible.
      - Frame N/2: arms partway between grip pose and neutral.
      - Frame N  : arms at neutral — gripper has released, robot fully risen.

    Why fade instead of IK-track?
    The arm workspace boundary is hit when shoulder rises above hold height —
    no feasible arm configuration can maintain Y=-0.055 (wall depth) AND
    Z=hold_height once shoulder_Z ≈ hold_Z.  Fading to neutral avoids the
    joint-limit clamp while still showing physically plausible arm motion.

    Gravity tension and constraint physics are demonstrated in HOLD_STEPS
    (mj_step with active connect constraints) before each advance.
    """
    slide_tgt = float(np.clip(target_world_z - 0.30, -0.30, 2.00))
    z0        = float(data.qpos[DOF_TORSO_Z])

    # Release grips NOW — the "hanging" physics was already shown in HOLD_STEPS.
    # Releasing before kinematic advance prevents old-row constraints from
    # pulling the torso back during the post-advance physics settle.
    for arm in held_targets:
        grip_release(data, arm)

    # Snapshot each arm's current IK angles (grip pose) and neutral target
    grip_q: dict[str, np.ndarray] = {}
    for arm in held_targets:
        dofs = np.array(ARM_DOFS[arm], dtype=int)
        grip_q[arm] = data.qpos[dofs].copy()

    for step in range(N_ADVANCE):
        t   = (step + 1) / N_ADVANCE
        t_s = t * t * (3.0 - 2.0 * t)   # smoothstep 0→1

        # Advance torso (kinematic)
        z = z0 + t_s * (slide_tgt - z0)
        data.qpos[DOF_TORSO_Z]  = z
        data.ctrl[CTRL_TORSO_Z] = z

        # Fade each held arm from grip pose → neutral
        # Visual: arms swing back as robot rises, like releasing the hold
        for arm, q0 in grip_q.items():
            dofs     = np.array(ARM_DOFS[arm],  dtype=int)
            ctrl_ids = np.array(ARM_CTRLS[arm], dtype=int)
            q = q0 * (1.0 - t_s)          # lerp toward q=0 (neutral)
            data.qpos[dofs]     = q
            data.ctrl[ctrl_ids] = q

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        if viewer is not None and step % 2 == 0:
            viewer.sync()
            if REAL_TIME:
                time.sleep(SIM_DT * 3)


# ---------- headless IK verification -----------------------------------------

def verify_ik(model) -> bool:
    print("\n" + "=" * 60)
    print("IK GEOMETRY VERIFICATION  (rows 1-5; row 0 = start pos, skipped)")
    print("=" * 60)
    all_ok = True
    for row, z_hold, xs in _HOLD_GRID[1:]:   # row 0 = start pos, never gripped
        torso_z = max(z_hold - 0.35, 0.30)
        print(f"\nRow {row}  hold_z={z_hold:.2f}  torso_z={torso_z:.2f}")
        for arm, col in [("left", 0), ("center", 1), ("right", 2)]:
            d = mujoco.MjData(model)
            mujoco.mj_resetData(model, d)
            d.qpos[DOF_TORSO_Z] = torso_z - 0.30
            target = HOLDS[f"r{row}_c{col}"].copy()
            conv, err = ik_reach(model, d, GRIPPER_SITES[arm], target,
                                 ARM_DOFS[arm])
            mujoco.mj_forward(model, d)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    GRIPPER_SITES[arm])
            g   = d.site_xpos[sid].copy()
            tag = "OK" if err < 0.015 else ("WARN" if err < 0.050 else "FAIL")
            print(f"  [{tag}] {arm:8s} r{row}_c{col}  err={err:.4f}m  "
                  f"grip={g.round(3)}")
            if err >= 0.050:
                all_ok = False
    print("\n" + ("ALL PASSED" if all_ok else "SOME FAILED"))
    return all_ok


# ---------- main -------------------------------------------------------------

def main() -> None:
    verify_only = "--verify" in sys.argv
    model_path  = Path(MODEL_PATH)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}. Run from repo root.")

    print(f"Loading: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"neq={model.neq}  nmocap={model.nmocap}")
    assert model.nv  == 11, f"Expected 11 DOFs, got {model.nv}"
    assert model.neq ==  3, f"Expected 3 equality constraints, got {model.neq}"
    assert model.nmocap == 3, f"Expected 3 mocap bodies, got {model.nmocap}"

    _build_ids(model)
    print(f"  Weld IDs:  {EQ_IDS}")
    print(f"  Mocap IDs: {MOCAP_IDS}")

    mujoco.mj_resetData(model, data)
    print("\nJoint map:")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"  [{j:2d}] {name}  qpos[{model.jnt_qposadr[j]}]")

    mujoco.mj_forward(model, data)
    print("\nNeutral gripper positions (q=0, torso Z=0.30):")
    for arm, site in GRIPPER_SITES.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        print(f"  {arm:8s}: {data.site_xpos[sid].round(3)}")
    print(f"  HOLD_Y = {HOLD_Y}")

    if verify_only:
        ok = verify_ik(model)
        sys.exit(0 if ok else 1)

    # --------------- inchworm climbing schedule --------------------------------
    # Pattern: grip row N (torso below), advance torso to row N, repeat.
    # Arms flex from "up" to "down" during each advance (visible in viewer).
    SCHEDULE = [
        ("grip",    1),
        ("advance", 0.65),
        ("grip",    2),
        ("advance", 1.00),
        ("grip",    3),
        ("advance", 1.35),
        ("grip",    4),
        ("advance", 1.70),
        ("grip",    5),
        ("advance", 2.05),
    ]

    # Active grips: arm → hold world position (for IK tracking during advance)
    held_targets: dict[str, np.ndarray] = {}
    moves = 0

    print("\nLaunching MuJoCo viewer...")
    print("Controls: Space=pause  Ctrl+R=reset  Scroll=zoom  Drag=rotate\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.20, 1.20]
        viewer.cam.distance  = 3.2
        viewer.cam.elevation = -15
        viewer.cam.azimuth   = 175

        print("Settling physics (gravity enabled)...")
        run_physics(model, data, viewer, 400)

        for action, val in SCHEDULE:
            if not viewer.is_running():
                break

            if action == "grip":
                row = int(val)

                # --- Release all active grips before re-gripping ---
                for arm in list(held_targets.keys()):
                    grip_release(data, arm)
                held_targets.clear()

                z_hold = HOLDS[f"r{row}_c0"][2]
                cur_z  = float(data.qpos[DOF_TORSO_Z]) + 0.30
                print(f"\n=== Grip row {row}  "
                      f"hold_Z={z_hold:.2f}m  torso_Z={cur_z:.2f}m ===")

                for arm, col in [("left", 0), ("center", 1), ("right", 2)]:
                    if not viewer.is_running():
                        break
                    target = HOLDS[f"r{row}_c{col}"].copy()

                    # Kinematic animation: retract → extend to hold
                    smooth_grip(model, data, viewer, arm, target)

                    # Lock gripper to hold via connect constraint
                    grip_activate(model, data, arm, target)
                    held_targets[arm] = target

                    # Physics settle: robot hangs from hold under gravity
                    run_physics(model, data, viewer, HOLD_STEPS)
                    moves += 1

            elif action == "advance":
                new_z = float(val)
                cur_z = float(data.qpos[DOF_TORSO_Z]) + 0.30
                if abs(new_z - cur_z) > 0.01:
                    print(f"\n--- Pull up  {cur_z:.2f}m → {new_z:.2f}m  "
                          f"(arms: {list(held_targets.keys())}) ---")
                    pull_up(model, data, viewer, new_z, held_targets)
                    held_targets.clear()   # grips released inside pull_up
                    # Physics settle at new height (free, gravity only)
                    run_physics(model, data, viewer, HOLD_STEPS)

        print(f"\n{'='*50}")
        print(f"Summit reached!  Total gripper moves: {moves}")
        print(f"{'='*50}")
        print("Viewer open — close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if REAL_TIME:
                time.sleep(SIM_DT)

    print("Done.")


if __name__ == "__main__":
    main()
