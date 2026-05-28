"""Test S03: IK solver for UR5e arm."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from sim.ik_solver import solve_ik, ik_from_cartesian_delta, ARM_JOINT_NAMES

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sim", "assets", "scene.xml"))

# UR5e home configuration
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def _get_arm_qpos_idx(model):
    return np.array([model.joint(name).id for name in ARM_JOINT_NAMES])


def _set_home(model, data):
    """Set arm to home pose."""
    qpos_idx = _get_arm_qpos_idx(model)
    for i, jid in enumerate(qpos_idx):
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = HOME_QPOS[i]
    mujoco.mj_forward(model, data)


def _get_arm_qpos(model, data):
    """Get current arm joint angles."""
    qpos_idx = _get_arm_qpos_idx(model)
    return np.array([data.qpos[model.jnt_qposadr[jid]] for jid in qpos_idx])


def test_home_ik_converges():
    """IK from home pose back to current pose converges immediately."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    _set_home(model, data)

    # Get current EE pose as target
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    target_pos = data.body(body_id).xpos.copy()
    target_quat = data.body(body_id).xquat.copy()

    current_q = _get_arm_qpos(model, data)

    # Reset to home (which is already the target)
    success, q_sol, info = solve_ik(model, data, target_pos, target_quat)

    assert success, f"IK should converge at home, got pos_err={info['pos_err']:.6f}"
    assert info["iterations"] < 5, f"Should converge in <5 iters, took {info['iterations']}"
    assert np.allclose(q_sol, HOME_QPOS, atol=1e-3), "Solution should be near home"
    print(f"[PASS] Home IK converges in {info['iterations']} iterations, "
          f"pos_err={info['pos_err']:.6f}, rot_err={info['rot_err']:.6f}")


def test_small_offset_ik():
    """IK converges for small offset from home."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    _set_home(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    current_pos = data.body(body_id).xpos.copy()
    current_quat = data.body(body_id).xquat.copy()

    # Target: 1cm offset in X
    target_pos = current_pos + np.array([0.01, 0, 0])

    success, q_sol, info = solve_ik(model, data, target_pos, current_quat)

    assert success, f"IK should converge, got pos_err={info['pos_err']:.6f}"
    assert info["pos_err"] < 1e-4, f"Position error {info['pos_err']:.6f} > 1e-4"
    assert info["rot_err"] < np.radians(1), f"Rotation error {np.degrees(info['rot_err']):.2f}° > 1°"

    # Verify by forward kinematics
    qpos_addr = np.array([model.jnt_qposadr[j] for j in _get_arm_qpos_idx(model)])
    data.qpos[qpos_addr] = q_sol
    mujoco.mj_forward(model, data)
    final_pos = data.body(body_id).xpos.copy()

    final_err = np.linalg.norm(final_pos - target_pos)
    print(f"[PASS] Small offset IK: converged in {info['iterations']} iters, "
          f"final_pos_err={final_err:.6f}")
    assert final_err < 1e-3, f"FK verification failed, pos_err={final_err:.6f}"


def test_unreachable_target():
    """Unreachable target returns success=False."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    _set_home(model, data)

    # Target far outside workspace
    target_pos = np.array([2.0, 2.0, 2.0])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity quat

    success, q_sol, info = solve_ik(model, data, target_pos, target_quat,
                                     max_iter=50)

    assert not success, "Should not converge for unreachable target"
    print(f"[PASS] Unreachable target: success=False, "
          f"pos_err={info['pos_err']:.4f} after {info['iterations']} iters")


def test_cartesian_delta():
    """ik_from_cartesian_delta produces small, consistent changes."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    _set_home(model, data)

    current_q = _get_arm_qpos(model, data)

    # Single small delta
    success, new_q, info = ik_from_cartesian_delta(
        model, data, current_q,
        dx=0.01, dy=0, dz=0, droll=0, dpitch=0, dyaw=0
    )

    assert success, f"Delta IK should converge, pos_err={info['pos_err']:.6f}"
    assert not np.allclose(new_q, current_q), "Joints should have changed"

    # Accumulate 10 small deltas
    q_accum = current_q.copy()
    for _ in range(10):
        success, q_accum, info = ik_from_cartesian_delta(
            model, data, q_accum,
            dx=0.005, dy=0, dz=0, droll=0, dpitch=0, dyaw=0
        )
        if not success:
            break

    # Verify final EE position is close to home + 0.06m in X
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    qpos_idx = np.array([model.jnt_qposadr[j] for j in _get_arm_qpos_idx(model)])
    data.qpos[qpos_idx] = q_accum
    mujoco.mj_forward(model, data)
    final_pos = data.body(body_id).xpos.copy()

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    _set_home(model, data)
    home_pos = data.body(home_id).xpos.copy()

    offset = final_pos - home_pos
    print(f"[PASS] Cartesian delta: accumulated offset = [{offset[0]:.4f}, "
          f"{offset[1]:.4f}, {offset[2]:.4f}], expected ~0.05 in X")
    assert abs(offset[0] - 0.05) < 0.015, f"X offset {offset[0]:.4f} not near 0.05"


def test_ik_performance():
    """IK solves in under 1ms per call."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    _set_home(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    target_pos = data.body(body_id).xpos.copy() + np.array([0.005, 0.003, 0.002])
    target_quat = data.body(body_id).xquat.copy()

    # Warm up
    _set_home(model, data)
    solve_ik(model, data, target_pos, target_quat)

    # Benchmark
    times = []
    for _ in range(100):
        _set_home(model, data)
        t0 = time.perf_counter()
        solve_ik(model, data, target_pos, target_quat)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    print(f"[PASS] IK performance: avg={avg_ms:.3f}ms, p99={p99_ms:.3f}ms")
    assert avg_ms < 1.0, f"Average IK time {avg_ms:.3f}ms > 1ms"


def main():
    print("=" * 60)
    print("S03 — IK Solver Validation")
    print("=" * 60)

    test_home_ik_converges()
    test_small_offset_ik()
    test_unreachable_target()
    test_cartesian_delta()
    test_ik_performance()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
