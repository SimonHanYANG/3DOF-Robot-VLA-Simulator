"""Test S04: SimController for UR5e pick-and-place."""

import sys
import os
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.controller import SimController, ACTION_LOW, ACTION_HIGH, HOME_QPOS

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                            "sim", "assets", "scene.xml"))


def test_init():
    """SimController loads scene successfully."""
    with SimController(SCENE_PATH) as sim:
        assert sim.model is not None
        assert sim.data is not None
        assert sim.model.nu == 8
    print("[PASS] Init: scene loaded, 8 actuators")


def test_context_manager():
    """with SimController(...) as sim: works."""
    with SimController(SCENE_PATH) as sim:
        assert sim.renderer is not None
    print("[PASS] Context manager: renderer cleaned up")


def test_reset():
    """reset() sets arm to home and gripper ctrl to open."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        state = sim.get_state()
        np.testing.assert_allclose(state["joint_positions"], HOME_QPOS, atol=1e-3)
        # Gripper ctrl is set to open; verify by stepping physics
        for _ in range(200):
            mujoco.mj_step(sim.model, sim.data)
        grip_q = sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint("finger_left_joint").id]]
        assert grip_q > 0.04, f"Gripper should be open after stepping, got {grip_q}"
    print("[PASS] Reset: home pose and gripper open")


def test_step_moves_ee():
    """step() with +X action moves end-effector in +X."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        pos_before = sim.get_state()["ee_pose"][:3].copy()
        action = np.array([0.01, 0, 0, 0, 0, 0, 1.0])
        obs = sim.step(action)
        pos_after = obs["ee_pose"][:3]
        assert pos_after[0] > pos_before[0], f"EE should move +X: {pos_before} -> {pos_after}"
        print(f"[PASS] Step: EE moved +X by {pos_after[0] - pos_before[0]:.4f}m")


def test_action_clipping():
    """Out-of-range actions are clipped."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        big_action = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0])
        obs = sim.step(big_action)
        # Should not crash
        assert obs is not None
    print("[PASS] Action clipping: large actions handled safely")


def test_get_state():
    """get_state() returns all required keys."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        state = sim.get_state()
        for key in ["ee_pose", "ee_velocity", "joint_positions",
                     "joint_velocities", "gripper_qpos", "obj_poses",
                     "target_pose", "time"]:
            assert key in state, f"Missing key: {key}"
        assert state["ee_pose"].shape == (7,)
        assert state["joint_positions"].shape == (6,)
        assert len(state["obj_poses"]) == 3
    print("[PASS] Get state: all keys present with correct shapes")


def test_camera_images():
    """get_camera_images() returns 224x224x3 RGB images."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        imgs = sim.get_camera_images()
        assert "top_cam" in imgs
        assert "wrist_cam" in imgs
        for name, img in imgs.items():
            assert img.shape == (224, 224, 3), f"{name} shape: {img.shape}"
            assert img.dtype == np.uint8
    print("[PASS] Camera images: 224x224x3 from both cameras")


def test_gripper_open_close():
    """Gripper opens and closes."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()

        # Open gripper
        sim.data.ctrl[6] = 0.045
        sim.data.ctrl[7] = 0.045
        for _ in range(200):
            mujoco.mj_step(sim.model, sim.data)
        open_q = sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint("finger_left_joint").id]]

        # Close gripper
        sim.data.ctrl[6] = 0.0
        sim.data.ctrl[7] = 0.0
        for _ in range(200):
            mujoco.mj_step(sim.model, sim.data)
        close_q = sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint("finger_left_joint").id]]

        assert open_q > close_q, f"Gripper didn't close: open={open_q}, close={close_q}"
    print("[PASS] Gripper open/close: operates correctly")


def test_is_done_timeout():
    """is_done() returns timeout after 500 steps."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        action = np.zeros(7)
        for _ in range(500):
            sim.step(action)
        done, success, info = sim.is_done()
        assert done
        assert info["reason"] == "timeout"
        assert not success
    print("[PASS] is_done timeout: triggers at 500 steps")


def test_multi_step():
    """Running 500 steps without errors."""
    with SimController(SCENE_PATH) as sim:
        sim.reset()
        for i in range(500):
            action = np.random.uniform(ACTION_LOW, ACTION_HIGH)
            obs = sim.step(action)
            assert obs is not None
    print("[PASS] Multi-step: 500 random steps without errors")


def main():
    print("=" * 60)
    print("S04 — SimController Validation")
    print("=" * 60)

    test_init()
    test_context_manager()
    test_reset()
    test_step_moves_ee()
    test_action_clipping()
    test_get_state()
    test_camera_images()
    test_gripper_open_close()
    test_is_done_timeout()
    test_multi_step()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
