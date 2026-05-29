"""UR5e pick-and-place simulation controller."""

import numpy as np
import mujoco

from .ik_solver import (
    solve_ik, ik_from_cartesian_delta,
    ARM_JOINT_NAMES, DEFAULT_EEF_BODY, _get_arm_joint_indices,
)

# Action space bounds
ACTION_LOW = np.array([-0.02, -0.02, -0.02, -0.1, -0.1, -0.1, 0.0])
ACTION_HIGH = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 1.0])

HOME_QPOS = np.array([0.341609, -2.347910, 0.574560, -1.368243, -0.341605, 2.941593])

OBJ_BODY_NAMES = ["obj_cube", "obj_sphere", "obj_cylinder"]
GRIPPER_GEOMS = ["finger_left_geom", "finger_right_geom"]
EEF_BODY = DEFAULT_EEF_BODY
GRIPPER_MAX = 0.045


class SimController:
    """UR5e pick-and-place simulation controller."""

    def __init__(self, scene_path, render_size=(224, 224), substeps=10):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_size = render_size
        self.substeps = substeps
        self.renderer = mujoco.Renderer(self.model, height=render_size[0],
                                        width=render_size[1])

        self.qpos_idx, self.dof_idx = _get_arm_joint_indices(self.model)
        self.eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")

        self.ctrl_arm_idx = np.arange(6)
        self.ctrl_grip_left = 6
        self.ctrl_grip_right = 7
        self.grip_left_qpos = self.model.joint("finger_left_joint").id
        self.grip_right_qpos = self.model.joint("finger_right_joint").id

        self._step_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.renderer.close()

    def reset(self, obj_name="obj_cube", obj_pos=None, target_pos=None):
        """Reset scene: place object, set target, move arm to home.

        The reset sequence:
        1. Clear simulation data
        2. Set arm joints to HOME_QPOS
        3. Use FK (mj_forward) to compute EE position
        4. Pre-position gripper_base at EE (avoids weld sweep collision)
        5. Place object
        6. Set target marker and gripper control
        """
        mujoco.mj_resetData(self.model, self.data)

        # Arm to home
        for i, jid in enumerate(self.qpos_idx):
            self.data.qpos[self.model.jnt_qposadr[jid]] = HOME_QPOS[i]
        self.data.ctrl[self.ctrl_arm_idx] = HOME_QPOS

        # Set gripper_base freejoint to EE position computed via FK.
        # mj_forward updates fixed-body (wrist_3_link) position from
        # joint angles. We copy that to the gripper's freejoint.
        mujoco.mj_forward(self.model, self.data)
        eef_pos = self.data.body(self.eef_id).xpos.copy()
        eef_quat = self.data.body(self.eef_id).xquat.copy()
        gripper_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "gripper_base")
        gripper_jid = self.model.body_jntadr[gripper_bid]
        gripper_qpos_adr = self.model.jnt_qposadr[gripper_jid]
        self.data.qpos[gripper_qpos_adr:gripper_qpos_adr+3] = eef_pos
        self.data.qpos[gripper_qpos_adr+3:gripper_qpos_adr+7] = eef_quat

        # Place object
        if obj_name is not None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            jid = self.model.body_jntadr[bid]
            if jid >= 0:
                qpos_adr = self.model.jnt_qposadr[jid]
                if obj_pos is not None:
                    self.data.qpos[qpos_adr:qpos_adr+3] = obj_pos
                self.data.qpos[qpos_adr+3] = 1.0  # unit quaternion
                self.data.qpos[qpos_adr+4:qpos_adr+7] = 0.0

        # Target marker position
        if target_pos is not None:
            self.data.site_xpos[self.target_site_id] = target_pos

        # Gripper open
        self.data.ctrl[self.ctrl_grip_left] = GRIPPER_MAX
        self.data.ctrl[self.ctrl_grip_right] = GRIPPER_MAX

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    def disable_collisions(self):
        """Temporarily disable all collisions (e.g., during approach phase)."""
        self._saved_contype = self.model.geom_contype.copy()
        self._saved_conaffinity = self.model.geom_conaffinity.copy()
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0

    def enable_collisions(self):
        """Re-enable collisions after approach phase."""
        if hasattr(self, '_saved_contype'):
            self.model.geom_contype[:] = self._saved_contype
            self.model.geom_conaffinity[:] = self._saved_conaffinity

    def step(self, action):
        """Execute one step with 7D action, return observation."""
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)

        current_q = np.array([self.data.qpos[adr] for adr in self.qpos_idx])
        ik_ok, target_q, _ = ik_from_cartesian_delta(
            self.model, self.data, current_q,
            action[0], action[1], action[2],
            action[3], action[4], action[5]
        )

        if ik_ok:
            self.data.ctrl[self.ctrl_arm_idx] = target_q

        self.data.ctrl[self.ctrl_grip_left] = action[6] * GRIPPER_MAX
        self.data.ctrl[self.ctrl_grip_right] = action[6] * GRIPPER_MAX

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        return self.get_obs()

    def step_to_pose(self, target_pos, target_quat, gripper_cmd):
        """Move EE to target pose via IK from actual position.

        Unlike step() which uses Cartesian deltas, this computes IK directly
        from the current arm state, avoiding drift accumulation.

        Args:
            target_pos: target EE position (3,)
            target_quat: target EE quaternion (w,x,y,z)
            gripper_cmd: gripper command (0.0=closed, 1.0=open)

        Returns:
            obs: observation dict
        """
        success, q_sol, info = solve_ik(self.model, self.data,
                                         target_pos, target_quat)
        if success:
            self.data.ctrl[self.ctrl_arm_idx] = q_sol

        self.data.ctrl[self.ctrl_grip_left] = gripper_cmd * GRIPPER_MAX
        self.data.ctrl[self.ctrl_grip_right] = gripper_cmd * GRIPPER_MAX

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        return self.get_obs()

    def set_ee_pose(self, pos, quat):
        """Directly set end-effector pose via IK."""
        success, q_sol, info = solve_ik(self.model, self.data, pos, quat)
        if success:
            self.data.ctrl[self.ctrl_arm_idx] = q_sol
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def get_state(self):
        """Get full scene state dict."""
        eef_pos = self.data.body(self.eef_id).xpos.copy()
        eef_quat = self.data.body(self.eef_id).xquat.copy()
        eef_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.eef_id, eef_vel, True)

        joint_pos = np.array([self.data.qpos[adr] for adr in self.qpos_idx])
        joint_vel = np.array([self.data.qvel[dof] for dof in self.dof_idx])

        grip_left = self.data.qpos[self.model.jnt_qposadr[self.grip_left_qpos]]
        grip_right = self.data.qpos[self.model.jnt_qposadr[self.grip_right_qpos]]

        obj_poses = {}
        for name in OBJ_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            qpos_adr = self.model.jnt_qposadr[jid]
            obj_poses[name] = {
                "pos": self.data.qpos[qpos_adr:qpos_adr+3].copy(),
                "quat": self.data.qpos[qpos_adr+3:qpos_adr+7].copy(),
            }

        return {
            "ee_pose": np.concatenate([eef_pos, eef_quat]),
            "ee_velocity": eef_vel,
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "gripper_qpos": (grip_left + grip_right) / 2.0,
            "obj_poses": obj_poses,
            "target_pose": self.data.site_xpos[self.target_site_id].copy(),
            "time": self.data.time,
        }

    def get_camera_images(self):
        """Render top_cam and wrist_cam RGB images."""
        imgs = {}
        for cam_name in ["top_cam", "wrist_cam"]:
            self.renderer.update_scene(self.data, camera=cam_name)
            imgs[cam_name] = self.renderer.render().copy()
        return imgs

    def check_grasp(self, obj_name="obj_cube"):
        """Check if both fingers are in contact with the object."""
        obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"{obj_name.split('_')[1]}_geom")
        left_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                          "finger_left_geom")
        right_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "finger_right_geom")

        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == left_geom_id and g2 == obj_geom_id) or \
               (g1 == obj_geom_id and g2 == left_geom_id):
                left_contact = True
            if (g1 == right_geom_id and g2 == obj_geom_id) or \
               (g1 == obj_geom_id and g2 == right_geom_id):
                right_contact = True

        return left_contact and right_contact

    def is_done(self, obj_name="obj_cube"):
        """Check termination. Returns (done, success, info)."""
        info = {"reason": None}

        # Time limit
        if self._step_count >= 500:
            info["reason"] = "timeout"
            return True, False, info

        # Object fell off table
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        jid = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jid]
        obj_z = self.data.qpos[qpos_adr + 2]
        if obj_z < 0.5:
            info["reason"] = "object_fallen"
            return True, False, info

        # Success: object above table near target while grasped
        if self.check_grasp(obj_name):
            obj_xy = self.data.qpos[qpos_adr:qpos_adr+2]
            target_xy = self.data.site_xpos[self.target_site_id][:2]
            if np.linalg.norm(obj_xy - target_xy) < 0.02 and obj_z > 0.85:
                info["reason"] = "success"
                return True, True, info

        return False, False, info

    def get_obs(self):
        """Get observation dict for RL/VLA."""
        state = self.get_state()
        imgs = self.get_camera_images()
        return {**state, "images": imgs}
