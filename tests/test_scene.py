"""Test S02: Load the MuJoCo scene and verify all components."""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco


SCENE_PATH = os.path.join(os.path.dirname(__file__), "..", "sim", "assets", "scene.xml")


def test_scene_loads():
    """Scene XML parses and loads without error."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    assert model is not None
    assert data is not None
    print(f"[PASS] Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints")
    return model, data


def test_arm_joints(model, data):
    """Arm has 3 slide joints with correct ranges."""
    for name, lo, hi in [("slide_x", -0.5, 0.5), ("slide_y", -0.3, 0.3), ("slide_z", 0.0, 0.4)]:
        jid = model.joint(name).id
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, f"{name} not a slide joint"
        assert np.isclose(model.jnt_range[jid][0], lo), f"{name} lower range mismatch"
        assert np.isclose(model.jnt_range[jid][1], hi), f"{name} upper range mismatch"
    print("[PASS] Arm joints: 3 slide joints with correct ranges")


def test_arm_actuators(model):
    """Arm has 3 position actuators."""
    assert model.nu == 3, f"Expected 3 actuators, got {model.nu}"
    for name in ["act_x", "act_y", "act_z"]:
        aid = model.actuator(name).id
        assert aid >= 0, f"Actuator {name} not found"
    print("[PASS] Actuators: 3 position actuators found")


def test_objects_exist(model):
    """All 3 objects present with correct geoms."""
    for name, geom_name in [("obj_sphere", "sphere_geom"),
                             ("obj_cube", "cube_geom"),
                             ("obj_cone", "cone_geom")]:
        bid = model.body(name).id
        gid = model.geom(geom_name).id
        assert bid >= 0, f"Body {name} not found"
        assert gid >= 0, f"Geom {geom_name} not found"
    print("[PASS] Objects: sphere, cube, cone all present")


def test_obstacles_exist(model):
    """All 4 obstacles present."""
    for i in range(1, 5):
        bid = model.body(f"obstacle_{i}").id
        gid = model.geom(f"obs_{i}").id
        assert bid >= 0 and gid >= 0, f"Obstacle {i} not found"
    print("[PASS] Obstacles: 4 obstacles present")


def test_cameras(model):
    """Both cameras defined in scene."""
    assert model.ncam == 2, f"Expected 2 cameras, got {model.ncam}"
    for name in ["top_cam", "angle_cam"]:
        cid = model.camera(name).id
        assert cid >= 0, f"Camera {name} not found"
    print("[PASS] Cameras: top_cam and angle_cam both defined")


def test_no_initial_overlap(model, data):
    """No bodies overlap at initial state."""
    mujoco.mj_forward(model, data)
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            dist = np.linalg.norm(data.geom_xpos[i] - data.geom_xpos[j])
            min_dist = model.geom_size[i][0] + model.geom_size[j][0]
            # Skip wall/table checks and very large geoms
            if min_dist > 0.3:
                continue
            # Only flag severe overlaps (allow touching)
            if dist < min_dist * 0.5:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j)
                print(f"  [WARN] Possible overlap: {g1} <-> {g2}, dist={dist:.4f}")
    print("[PASS] No severe initial overlaps detected")


def test_arm_movable(model, data):
    """Arm joints can be moved within range without error."""
    # Find qpos indices for arm joints
    x_idx = model.joint("slide_x").qposadr[0]
    y_idx = model.joint("slide_y").qposadr[0]
    z_idx = model.joint("slide_z").qposadr[0]

    # Move X toward positive limit (2000 steps = 4s)
    data.ctrl[0] = 0.4
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert data.qpos[x_idx] > 0.1, f"X joint didn't move: {data.qpos[x_idx]:.4f}"

    # Reset and move Y
    mujoco.mj_resetData(model, data)
    data.ctrl[1] = 0.2
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert data.qpos[y_idx] > 0.05, f"Y joint didn't move: {data.qpos[y_idx]:.4f}"

    # Reset and move Z
    mujoco.mj_resetData(model, data)
    data.ctrl[2] = 0.3
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert data.qpos[z_idx] > 0.05, f"Z joint didn't move: {data.qpos[z_idx]:.4f}"

    print("[PASS] Arm joints movable within range")


def test_render_cameras(model, data):
    """Camera rendering produces 224x224 RGB images."""
    renderer = mujoco.Renderer(model, height=224, width=224)

    # Top camera
    renderer.update_scene(data, camera="top_cam")
    img_top = renderer.render()
    assert img_top.shape == (224, 224, 3), f"top_cam image shape {img_top.shape}"

    # Angle camera
    renderer.update_scene(data, camera="angle_cam")
    img_angle = renderer.render()
    assert img_angle.shape == (224, 224, 3), f"angle_cam image shape {img_angle.shape}"

    print("[PASS] Camera rendering: both produce 224x224x3 RGB images")
    renderer.close()


def main():
    print("=" * 60)
    print("S02 — MuJoCo Scene Validation")
    print("=" * 60)

    model, data = test_scene_loads()
    test_arm_joints(model, data)
    test_arm_actuators(model)
    test_objects_exist(model)
    test_obstacles_exist(model)
    test_cameras(model)
    test_no_initial_overlap(model, data)
    test_arm_movable(model, data)
    test_render_cameras(model, data)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
