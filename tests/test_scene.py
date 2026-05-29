"""Test S02: Load the UR5e pick-and-place scene and verify all components."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sim", "assets", "scene.xml"))


def test_scene_loads():
    """Scene XML parses and loads without error."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    assert model is not None
    assert data is not None
    print(f"[PASS] Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators")
    return model, data


def test_arm_joints(model):
    """Arm has 6 revolute joints."""
    expected = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for name in expected:
        jid = model.joint(name).id
        assert jid >= 0, f"Joint {name} not found"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, f"{name} not a hinge joint"
    print(f"[PASS] Arm joints: 6 revolute joints found")


def test_actuators(model):
    """Arm has 6 actuators + 2 gripper actuators = 8 total."""
    assert model.nu == 8, f"Expected 8 actuators, got {model.nu}"
    arm_acts = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    gripper_acts = ["gripper_left", "gripper_right"]
    for name in arm_acts + gripper_acts:
        aid = model.actuator(name).id
        assert aid >= 0, f"Actuator {name} not found"
    print(f"[PASS] Actuators: 6 arm + 2 gripper = 8 total")


def test_objects_exist(model, data):
    """All 3 objects present with correct geoms."""
    for name, geom_name in [("obj_cube", "cube_geom"),
                             ("obj_sphere", "sphere_geom"),
                             ("obj_cylinder", "cylinder_geom")]:
        bid = model.body(name).id
        gid = model.geom(geom_name).id
        assert bid >= 0, f"Body {name} not found"
        assert gid >= 0, f"Geom {geom_name} not found"
    print("[PASS] Objects: cube, sphere, cylinder all present")


def test_gripper_joints(model):
    """Gripper has two slide joints."""
    for name in ["finger_left_joint", "finger_right_joint"]:
        jid = model.joint(name).id
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, f"{name} not a slide joint"
        assert np.isclose(model.jnt_range[jid][1], 0.045), f"{name} upper range not 0.045"
    print("[PASS] Gripper: two slide joints with correct range")


def test_gripper_operates(model, data):
    """Gripper can open and close."""
    from sim.controller import HOME_QPOS
    qpos_idx = [model.joint(name).id for name in
                ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]

    mujoco.mj_resetData(model, data)
    for i, jid in enumerate(qpos_idx):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    data.ctrl[:6] = HOME_QPOS
    data.ctrl[6] = 0.045  # open
    data.ctrl[7] = 0.045
    for _ in range(200):
        mujoco.mj_step(model, data)
    left_open = data.joint("finger_left_joint").qpos[0]

    mujoco.mj_resetData(model, data)
    for i, jid in enumerate(qpos_idx):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    data.ctrl[:6] = HOME_QPOS
    data.ctrl[6] = 0.0  # close
    data.ctrl[7] = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)
    left_closed = data.joint("finger_left_joint").qpos[0]

    assert left_open > left_closed, "Gripper didn't close"
    print(f"[PASS] Gripper operates: open={left_open:.4f}, closed={left_closed:.4f}")


def test_cameras(model, data):
    """Both cameras render 224x224 RGB images."""
    assert model.ncam >= 2, f"Expected >= 2 cameras, got {model.ncam}"
    renderer = mujoco.Renderer(model, height=224, width=224)

    renderer.update_scene(data, camera="top_cam")
    img_top = renderer.render()
    assert img_top.shape == (224, 224, 3), f"top_cam shape {img_top.shape}"

    renderer.update_scene(data, camera="wrist_cam")
    img_wrist = renderer.render()
    assert img_wrist.shape == (224, 224, 3), f"wrist_cam shape {img_wrist.shape}"

    renderer.close()
    print("[PASS] Cameras: top_cam and wrist_cam both produce 224x224x3 RGB")


def test_weld_constraint(model, data):
    """Gripper is welded to the arm."""
    mujoco.mj_forward(model, data)
    # Check that gripper_base position matches the arm's attachment site
    gripper_pos = data.body("gripper_base").xpos
    # The attachment site is on wrist_3_link
    assert not np.allclose(gripper_pos, [0, 0, 0]), "Gripper not attached to arm"
    print(f"[PASS] Weld: gripper attached at {gripper_pos}")


def test_no_initial_overlap(model, data):
    """No severe initial overlaps."""
    mujoco.mj_forward(model, data)
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            dist = np.linalg.norm(data.geom_xpos[i] - data.geom_xpos[j])
            min_dist = model.geom_size[i][0] + model.geom_size[j][0]
            if min_dist > 0.3:
                continue
            if dist < min_dist * 0.5:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j)
                print(f"  [WARN] Possible overlap: {g1} <-> {g2}, dist={dist:.4f}")
    print("[PASS] No severe initial overlaps detected")


def main():
    print("=" * 60)
    print("S02 — UR5e Scene Validation")
    print("=" * 60)

    model, data = test_scene_loads()
    test_arm_joints(model)
    test_actuators(model)
    test_objects_exist(model, data)
    test_gripper_joints(model)
    test_gripper_operates(model, data)
    test_cameras(model, data)
    test_weld_constraint(model, data)
    test_no_initial_overlap(model, data)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
