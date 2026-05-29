"""Test S05: Grasp pose generator for geometric objects."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.planner.grasp_planner import (
    GraspPose, GraspPlanner, OBJ_SIZES, _top_down_quat, _finger_width_for_object,
)

# Object positions from scene.xml
OBJ_PARAMS = {
    "obj_cube": {"type": "cube", "pos": np.array([0.3, 0.1, 0.83]),
                 "size": {"half_size": 0.02}},
    "obj_sphere": {"type": "sphere", "pos": np.array([0.5, -0.1, 0.815]),
                   "size": {"radius": 0.025}},
    "obj_cylinder": {"type": "cylinder", "pos": np.array([0.4, 0.2, 0.83]),
                     "size": {"radius": 0.015, "half_height": 0.03}},
}

TABLE_Z = 0.81  # table surface


def test_top_down_quat():
    """Top-down quaternion has Z-axis pointing down."""
    quat = _top_down_quat(0.0)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    # Z-axis of gripper should point down (0, 0, -1)
    z_axis = R[:, 2]
    np.testing.assert_allclose(z_axis, [0, 0, -1], atol=1e-6)
    print("[PASS] Top-down quat: gripper Z-axis points down")


def test_top_down_quat_with_rotation():
    """Top-down quaternion with Z rotation."""
    from scipy.spatial.transform import Rotation
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        quat = _top_down_quat(angle)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        z_axis = R[:, 2]
        np.testing.assert_allclose(z_axis, [0, 0, -1], atol=1e-6)
    print("[PASS] Top-down quat with rotation: Z-axis always down")


def test_finger_width():
    """Finger width values are reasonable for each object type."""
    planner = GraspPlanner()
    for name, params in OBJ_PARAMS.items():
        obj_type = params["type"]
        obj_size = params["size"]
        fw = _finger_width_for_object(obj_type, obj_size)
        assert 0 <= fw <= 0.045, f"{name}: finger_width={fw} out of range"
        print(f"  {name}: finger_width={fw:.4f}")
    print("[PASS] Finger width: all within valid range")


def test_plan_grasp_cube():
    """Plan a single grasp for cube."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cube"]
    grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])

    assert isinstance(grasp, GraspPose)
    assert grasp.grasp_pos.shape == (3,)
    assert grasp.grasp_quat.shape == (4,)
    assert grasp.pre_grasp_pos.shape == (3,)
    assert grasp.approach_dir.shape == (3,)
    # Grasp at object XY
    np.testing.assert_allclose(grasp.grasp_pos[:2], params["pos"][:2], atol=1e-6)
    # Pre-grasp above grasp
    assert grasp.pre_grasp_pos[2] > grasp.grasp_pos[2]
    print(f"[PASS] Plan grasp cube: grasp_z={grasp.grasp_pos[2]:.4f}, "
          f"pre_grasp_z={grasp.pre_grasp_pos[2]:.4f}")


def test_plan_grasp_sphere():
    """Plan a single grasp for sphere."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_sphere"]
    grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])

    np.testing.assert_allclose(grasp.grasp_pos[:2], params["pos"][:2], atol=1e-6)
    assert grasp.pre_grasp_pos[2] > grasp.grasp_pos[2]
    print(f"[PASS] Plan grasp sphere: grasp_z={grasp.grasp_pos[2]:.4f}")


def test_plan_grasp_cylinder():
    """Plan a single grasp for cylinder."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cylinder"]
    grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])

    np.testing.assert_allclose(grasp.grasp_pos[:2], params["pos"][:2], atol=1e-6)
    assert grasp.pre_grasp_pos[2] > grasp.grasp_pos[2]
    print(f"[PASS] Plan grasp cylinder: grasp_z={grasp.grasp_pos[2]:.4f}")


def test_grasp_above_table():
    """All grasp poses are above table surface."""
    planner = GraspPlanner()
    for name, params in OBJ_PARAMS.items():
        grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])
        assert grasp.grasp_pos[2] > TABLE_Z, \
            f"{name}: grasp_z={grasp.grasp_pos[2]:.4f} < table_z={TABLE_Z}"
        assert grasp.pre_grasp_pos[2] > TABLE_Z, \
            f"{name}: pre_grasp_z={grasp.pre_grasp_pos[2]:.4f} < table_z={TABLE_Z}"
    print("[PASS] All grasps above table surface")


def test_pre_grasp_offset():
    """Pre-grasp is at least 0.08m above grasp."""
    planner = GraspPlanner()
    for name, params in OBJ_PARAMS.items():
        grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])
        offset = grasp.pre_grasp_pos[2] - grasp.grasp_pos[2]
        assert offset >= 0.08, f"{name}: pre_grasp offset={offset:.4f} < 0.08"
    print("[PASS] Pre-grasp offset >= 0.08m for all objects")


def test_sample_grasps_count():
    """sample_grasps returns correct number of grasps."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cube"]
    grasps = planner.sample_grasps(params["type"], params["pos"], params["size"],
                                   n=50)
    assert len(grasps) == 50
    print("[PASS] sample_grasps returns correct count")


def test_sample_grasps_diversity():
    """Sampled grasps have diverse positions and rotations."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cube"]
    rng = np.random.default_rng(42)
    grasps = planner.sample_grasps(params["type"], params["pos"], params["size"],
                                   n=100, rng=rng)

    # Check XY diversity
    positions = np.array([g.grasp_pos for g in grasps])
    xy_std = positions[:, :2].std(axis=0)
    assert xy_std[0] > 0.002, f"X std too low: {xy_std[0]:.4f}"
    assert xy_std[1] > 0.002, f"Y std too low: {xy_std[1]:.4f}"

    # Check finger width diversity
    widths = np.array([g.finger_width for g in grasps])
    assert widths.std() > 0.0005, f"Width std too low: {widths.std():.4f}"

    print(f"[PASS] Sample diversity: XY_std=({xy_std[0]:.4f}, {xy_std[1]:.4f}), "
          f"width_std={widths.std():.4f}")


def test_sample_grasps_all_above_table():
    """All 100 sampled grasps are above table for each object type."""
    planner = GraspPlanner()
    rng = np.random.default_rng(123)

    for name, params in OBJ_PARAMS.items():
        grasps = planner.sample_grasps(params["type"], params["pos"], params["size"],
                                       n=100, rng=rng)
        for i, g in enumerate(grasps):
            assert g.grasp_pos[2] > TABLE_Z, \
                f"{name} grasp {i}: z={g.grasp_pos[2]:.4f} < {TABLE_Z}"
        print(f"  {name}: 100 grasps all above table")
    print("[PASS] All sampled grasps above table")


def test_validate_grasp():
    """_validate_grasp accepts good grasps, rejects bad ones."""
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cube"]
    grasp = planner.plan_grasp(params["type"], params["pos"], params["size"])
    assert planner._validate_grasp(grasp), "Valid grasp should pass"

    # Reject grasp below table
    bad_grasp = GraspPose(
        pre_grasp_pos=np.array([0.3, 0.1, 0.7]),
        pre_grasp_quat=grasp.grasp_quat,
        grasp_pos=np.array([0.3, 0.1, 0.7]),
        grasp_quat=grasp.grasp_quat,
        finger_width=0.005,
        approach_dir=np.array([0, 0, -1]),
    )
    assert not planner._validate_grasp(bad_grasp), "Below-table grasp should fail"
    print("[PASS] Validate grasp: accepts good, rejects bad")


def test_generation_speed():
    """Generate > 1000 grasp poses per second."""
    import time
    planner = GraspPlanner()
    params = OBJ_PARAMS["obj_cube"]
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    n = 2000
    grasps = planner.sample_grasps(params["type"], params["pos"], params["size"],
                                   n=n, rng=rng)
    elapsed = time.perf_counter() - t0
    rate = n / elapsed
    print(f"  Generated {n} grasps in {elapsed:.3f}s ({rate:.0f} grasps/s)")
    assert rate > 1000, f"Too slow: {rate:.0f} grasps/s < 1000"
    print("[PASS] Generation speed: > 1000 grasps/s")


def main():
    print("=" * 60)
    print("S05 — Grasp Pose Generator Validation")
    print("=" * 60)

    test_top_down_quat()
    test_top_down_quat_with_rotation()
    test_finger_width()
    test_plan_grasp_cube()
    test_plan_grasp_sphere()
    test_plan_grasp_cylinder()
    test_grasp_above_table()
    test_pre_grasp_offset()
    test_sample_grasps_count()
    test_sample_grasps_diversity()
    test_sample_grasps_all_above_table()
    test_validate_grasp()
    test_generation_speed()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
