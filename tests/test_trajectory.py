"""Test S06: Pick-and-place trajectory generator."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.controller import SimController, HOME_QPOS
from sim.planner.grasp_planner import GraspPlanner, GraspPose, OBJ_SIZES
from sim.planner.pick_and_place import (
    interpolate_cartesian, pose_to_delta_action, get_home_ee_pose,
    PickPlaceTrajectoryGenerator, quat_mujoco_to_scipy, quat_scipy_to_mujoco,
)

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                            "sim", "assets", "scene.xml"))

# Object parameters from scene.xml
OBJ_PARAMS = {
    "obj_cube": {"type": "cube", "pos": np.array([0.3, 0.1, 0.842]),
                 "size": {"half_size": 0.03}},
    "obj_sphere": {"type": "sphere", "pos": np.array([0.5, -0.1, 0.845]),
                   "size": {"radius": 0.035}},
    "obj_cylinder": {"type": "cylinder", "pos": np.array([0.4, 0.2, 0.85]),
                     "size": {"radius": 0.025, "half_height": 0.05}},
}

TARGET_POS = np.array([0.5, 0.0, 0.845])


def test_quat_conversion():
    """Quaternion conversion roundtrip."""
    from scipy.spatial.transform import Rotation
    # Identity quaternion
    q_mj = np.array([1.0, 0.0, 0.0, 0.0])
    q_sc = quat_mujoco_to_scipy(q_mj)
    q_back = quat_scipy_to_mujoco(q_sc)
    np.testing.assert_allclose(q_mj, q_back, atol=1e-10)

    # Arbitrary quaternion
    R = Rotation.from_euler("ZYX", [0.3, -0.2, 0.1])
    q_sc2 = R.as_quat()
    q_mj2 = quat_scipy_to_mujoco(q_sc2)
    q_sc_back = quat_mujoco_to_scipy(q_mj2)
    np.testing.assert_allclose(q_sc2, q_sc_back, atol=1e-10)
    print("[PASS] Quaternion conversion: roundtrip correct")


def test_interpolate_cartesian():
    """Cartesian interpolation produces correct number of steps."""
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([0.1, 0.0, 0.0])
    start_quat = np.array([1.0, 0.0, 0.0, 0.0])
    end_quat = np.array([1.0, 0.0, 0.0, 0.0])

    positions, quats = interpolate_cartesian(start_pos, start_quat,
                                              end_pos, end_quat, 5)
    assert positions.shape == (5, 3), f"Shape: {positions.shape}"
    assert quats.shape == (5, 4), f"Shape: {quats.shape}"
    np.testing.assert_allclose(positions[0], start_pos, atol=1e-10)
    np.testing.assert_allclose(positions[-1], end_pos, atol=1e-10)
    # Positions are linearly spaced
    for i in range(1, 5):
        assert positions[i][0] > positions[i-1][0]
    print("[PASS] Interpolate Cartesian: 5 steps, start/end correct")


def test_interpolate_single_step():
    """Single-step interpolation returns exact endpoint."""
    pos = np.array([0.5, 0.3, 0.8])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    positions, quats = interpolate_cartesian(pos, quat, pos, quat, 1)
    assert positions.shape == (1, 3)
    np.testing.assert_allclose(positions[0], pos, atol=1e-10)
    print("[PASS] Single-step interpolation: returns exact position")


def test_pose_to_delta_action():
    """pose_to_delta_action computes correct deltas."""
    current_pos = np.array([0.0, 0.0, 0.0])
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([0.01, 0.0, 0.0])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    action = pose_to_delta_action(current_pos, current_quat,
                                   target_pos, target_quat, 1.0)
    assert action.shape == (7,)
    np.testing.assert_allclose(action[:3], [0.01, 0, 0], atol=1e-6)
    np.testing.assert_allclose(action[3:6], [0, 0, 0], atol=1e-6)
    assert action[6] == 1.0
    print("[PASS] Pose to delta: correct position delta and gripper")


def test_pose_to_delta_clipping():
    """Large deltas are clipped to safety bounds."""
    current_pos = np.array([0.0, 0.0, 0.0])
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([1.0, 0.0, 0.0])  # 1m away
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    action = pose_to_delta_action(current_pos, current_quat,
                                   target_pos, target_quat, 0.5)
    assert np.all(action[:3] <= 0.02), f"Position not clipped: {action[:3]}"
    assert np.all(action[:3] >= -0.02)
    assert np.all(action[3:6] <= 0.1)
    assert np.all(action[3:6] >= -0.1)
    print("[PASS] Pose to delta clipping: large deltas clipped to bounds")


def test_get_home_ee_pose():
    """get_home_ee_pose returns valid EE position."""
    with SimController(SCENE_PATH) as sim:
        home_pos, home_quat = get_home_ee_pose(sim.model, sim.data)
        assert home_pos.shape == (3,)
        assert home_quat.shape == (4,)
        # Home EE should be roughly above the table
        assert home_pos[2] > 0.0, f"Home Z too low: {home_pos[2]}"
        print(f"[PASS] Home EE pose: pos={home_pos}, quat={home_quat}")


def test_generate_trajectory():
    """generate() returns a list of 7D actions."""
    with SimController(SCENE_PATH) as sim:
        planner = GraspPlanner()
        gen = PickPlaceTrajectoryGenerator(sim, planner)

        params = OBJ_PARAMS["obj_cube"]
        actions = gen.generate("obj_cube", params["pos"], params["size"],
                               TARGET_POS)

        assert actions is not None, "generate() returned None"
        assert len(actions) > 0, "Empty trajectory"
        for i, a in enumerate(actions):
            assert a.shape == (7,), f"Action {i} shape: {a.shape}"
            assert np.all(a >= -1.0) and np.all(a <= 1.0), \
                f"Action {i} out of range: {a}"
        print(f"[PASS] Generate trajectory: {len(actions)} actions, all 7D")


def test_trajectory_has_phases():
    """Trajectory contains distinct phases (approach, descend, grasp, etc.)."""
    with SimController(SCENE_PATH) as sim:
        planner = GraspPlanner()
        gen = PickPlaceTrajectoryGenerator(sim, planner)

        params = OBJ_PARAMS["obj_cube"]
        actions = gen.generate("obj_cube", params["pos"], params["size"],
                               TARGET_POS)

        # Check that we have gripper open (1.0) and closed (0.0) phases
        gripper_values = [a[6] for a in actions]
        has_open = any(g > 0.5 for g in gripper_values)
        has_closed = any(g < 0.5 for g in gripper_values)
        assert has_open, "No gripper-open phase found"
        assert has_closed, "No gripper-closed phase found"

        # Check for zero-action phases (grasp/release waits)
        zero_actions = [i for i, a in enumerate(actions)
                        if np.allclose(a[:6], 0, atol=1e-6)]
        assert len(zero_actions) >= 10, f"Too few wait steps: {len(zero_actions)}"

        print(f"[PASS] Trajectory phases: {len(actions)} actions, "
              f"gripper open+closed, {len(zero_actions)} wait steps")


def test_trajectory_replay():
    """Replay trajectory in simulation without crashing."""
    with SimController(SCENE_PATH) as sim:
        planner = GraspPlanner()
        gen = PickPlaceTrajectoryGenerator(sim, planner)

        params = OBJ_PARAMS["obj_cube"]
        actions = gen.generate("obj_cube", params["pos"], params["size"],
                               TARGET_POS)

        # Replay
        sim.reset(obj_name="obj_cube", obj_pos=params["pos"],
                  target_pos=TARGET_POS)
        for i, action in enumerate(actions):
            obs = sim.step(action)
            assert obs is not None, f"step() returned None at step {i}"

        state = sim.get_state()
        print(f"[PASS] Trajectory replay: {len(actions)} steps completed, "
              f"final EE z={state['ee_pose'][2]:.3f}")


def test_generate_all_objects():
    """Generate trajectories for all object types."""
    with SimController(SCENE_PATH) as sim:
        planner = GraspPlanner()
        gen = PickPlaceTrajectoryGenerator(sim, planner)

        for name, params in OBJ_PARAMS.items():
            actions = gen.generate(name, params["pos"], params["size"],
                                   TARGET_POS)
            assert actions is not None, f"{name}: generate() returned None"
            assert len(actions) > 20, f"{name}: too few actions ({len(actions)})"
            print(f"  {name}: {len(actions)} actions")
        print("[PASS] Generate all objects: trajectories for cube, sphere, cylinder")


def test_generation_speed():
    """Trajectory generation is fast enough (< 0.5s)."""
    import time
    with SimController(SCENE_PATH) as sim:
        planner = GraspPlanner()
        gen = PickPlaceTrajectoryGenerator(sim, planner)

        params = OBJ_PARAMS["obj_cube"]
        t0 = time.perf_counter()
        for _ in range(10):
            actions = gen.generate("obj_cube", params["pos"], params["size"],
                                   TARGET_POS)
        elapsed = (time.perf_counter() - t0) / 10
        print(f"  Generation time: {elapsed:.4f}s per trajectory")
        assert elapsed < 0.5, f"Too slow: {elapsed:.4f}s > 0.5s"
        print("[PASS] Generation speed: < 0.5s per trajectory")


def main():
    print("=" * 60)
    print("S06 — Pick-and-Place Trajectory Generator Validation")
    print("=" * 60)

    test_quat_conversion()
    test_interpolate_cartesian()
    test_interpolate_single_step()
    test_pose_to_delta_action()
    test_pose_to_delta_clipping()
    test_get_home_ee_pose()
    test_generate_trajectory()
    test_trajectory_has_phases()
    test_trajectory_replay()
    test_generate_all_objects()
    test_generation_speed()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
