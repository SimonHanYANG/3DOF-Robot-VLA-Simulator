"""Pick-and-place trajectory generator for UR5e."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import mujoco

from .grasp_planner import GraspPose, GraspPlanner, OBJ_SIZES
from ..ik_solver import solve_ik, _get_arm_joint_indices, _get_body_id, DEFAULT_EEF_BODY
from ..controller import HOME_QPOS, ACTION_LOW, ACTION_HIGH


# Approach/descend speeds (m per step)
APPROACH_SPEED = 0.02
DESCEND_SPEED = 0.01

# Phase parameters
GRASP_WAIT_STEPS = 10   # steps to wait after closing gripper
RELEASE_WAIT_STEPS = 10  # steps to wait after opening gripper
LIFT_HEIGHT = 0.10       # lift height above grasp (m)
PLACE_OFFSET_Z = 0.02    # place height above target (m)


def quat_mujoco_to_scipy(quat_wxyz):
    """Convert MuJoCo quaternion (w,x,y,z) to scipy quaternion (x,y,z,w)."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def quat_scipy_to_mujoco(quat_xyzw):
    """Convert scipy quaternion (x,y,z,w) to MuJoCo quaternion (w,x,y,z)."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def interpolate_cartesian(start_pos, start_quat_mj, end_pos, end_quat_mj, n_steps):
    """Linear interpolation in Cartesian space (position linear, rotation SLERP).

    Args:
        start_pos: start position (3,)
        start_quat_mj: start quaternion MuJoCo format (w,x,y,z)
        end_pos: end position (3,)
        end_quat_mj: end quaternion MuJoCo format (w,x,y,z)
        n_steps: number of interpolation steps

    Returns:
        positions: (n_steps, 3)
        quats_mj: (n_steps, 4) quaternions in MuJoCo format (w,x,y,z)
    """
    positions = np.linspace(start_pos, end_pos, n_steps)

    # SLERP for rotation
    rots = Rotation.from_quat([
        quat_mujoco_to_scipy(start_quat_mj),
        quat_mujoco_to_scipy(end_quat_mj),
    ])
    slerp = Slerp([0, 1], rots)
    t_vals = np.linspace(0, 1, n_steps)
    interp_rots = slerp(t_vals)
    quats_scipy = interp_rots.as_quat()  # (n_steps, 4) in (x,y,z,w)
    quats_mj = np.array([quat_scipy_to_mujoco(q) for q in quats_scipy])

    return positions, quats_mj


def pose_to_delta_action(current_pos, current_quat_mj, target_pos, target_quat_mj,
                         gripper_cmd):
    """Compute 7D delta action from current pose to target pose.

    Args:
        current_pos: current EE position (3,)
        current_quat_mj: current EE quaternion (w,x,y,z)
        target_pos: target position (3,)
        target_quat_mj: target quaternion (w,x,y,z)
        gripper_cmd: gripper command (0.0=closed, 1.0=open)

    Returns:
        action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # Position delta
    pos_delta = target_pos - current_pos

    # Rotation delta via angle-axis
    R_current = Rotation.from_quat(quat_mujoco_to_scipy(current_quat_mj))
    R_target = Rotation.from_quat(quat_mujoco_to_scipy(target_quat_mj))
    R_delta = R_target * R_current.inv()
    rot_delta = R_delta.as_rotvec()  # (3,) angle-axis

    # Clip to safety bounds
    pos_delta = np.clip(pos_delta, ACTION_LOW[:3], ACTION_HIGH[:3])
    rot_delta = np.clip(rot_delta, ACTION_LOW[3:6], ACTION_HIGH[3:6])

    return np.concatenate([pos_delta, rot_delta, [gripper_cmd]])


def _compute_n_steps(start_pos, end_pos, speed):
    """Compute number of interpolation steps for a phase."""
    dist = np.linalg.norm(end_pos - start_pos)
    return max(int(np.ceil(dist / speed)), 1)


def get_home_ee_pose(model, data):
    """Compute EE pose at home configuration via forward kinematics.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        home_pos: (3,) home EE position
        home_quat: (4,) home EE quaternion (w,x,y,z)
    """
    qpos_idx, _ = _get_arm_joint_indices(model)
    body_id = _get_body_id(model, DEFAULT_EEF_BODY)

    # Set joints to home
    data.qpos[qpos_idx] = HOME_QPOS
    mujoco.mj_forward(model, data)

    home_pos = data.body(body_id).xpos.copy()
    home_quat = data.body(body_id).xquat.copy()
    return home_pos, home_quat


class PickPlaceTrajectoryGenerator:
    """Generate and verify pick-and-place trajectories."""

    def __init__(self, sim_controller, grasp_planner=None):
        """Initialize generator.

        Args:
            sim_controller: SimController instance
            grasp_planner: GraspPlanner instance (or None to create default)
        """
        self.sim = sim_controller
        self.grasp_planner = grasp_planner or GraspPlanner()

        # Compute home EE pose
        self.home_pos, self.home_quat = get_home_ee_pose(
            self.sim.model, self.sim.data
        )

    def generate(self, obj_name, obj_pos, obj_size, target_pos,
                 grasp_pose=None, rng=None):
        """Generate a complete pick-and-place trajectory.

        Args:
            obj_name: object body name (e.g., "obj_cube")
            obj_pos: object position (3,)
            obj_size: object size dict
            target_pos: target placement position (3,)
            grasp_pose: optional GraspPose (or None to generate one)
            rng: random generator (or None)

        Returns:
            actions: list of 7D numpy arrays, or None if generation fails
        """
        # Generate grasp pose if not provided
        if grasp_pose is None:
            obj_type = OBJ_SIZES.get(obj_name, {}).get("type", "cube")
            grasp_pose = self.grasp_planner.plan_grasp(obj_type, obj_pos, obj_size)

        # Build target poses for each phase
        grasp_quat = grasp_pose.grasp_quat

        # Lift position: above grasp
        lift_pos = grasp_pose.grasp_pos + np.array([0, 0, LIFT_HEIGHT])

        # Transport: above target at lift height
        transport_pos = np.array([target_pos[0], target_pos[1], lift_pos[2]])

        # Place position: at target with small Z offset
        place_pos = np.array([target_pos[0], target_pos[1],
                              target_pos[2] + PLACE_OFFSET_Z])

        actions = []

        # Phase 1: Approach (home → pre_grasp, gripper open)
        n = _compute_n_steps(self.home_pos, grasp_pose.pre_grasp_pos, APPROACH_SPEED)
        positions, quats = interpolate_cartesian(
            self.home_pos, self.home_quat,
            grasp_pose.pre_grasp_pos, grasp_pose.pre_grasp_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    self.home_pos, self.home_quat,
                    positions[i], quats[i], 1.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 1.0
                )
            actions.append(action)

        # Phase 2: Descend (pre_grasp → grasp, gripper open)
        n = _compute_n_steps(grasp_pose.pre_grasp_pos, grasp_pose.grasp_pos,
                             DESCEND_SPEED)
        positions, quats = interpolate_cartesian(
            grasp_pose.pre_grasp_pos, grasp_pose.pre_grasp_quat,
            grasp_pose.grasp_pos, grasp_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    grasp_pose.pre_grasp_pos, grasp_pose.pre_grasp_quat,
                    positions[i], quats[i], 1.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 1.0
                )
            actions.append(action)

        # Phase 3: Grasp (close gripper, wait)
        zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(GRASP_WAIT_STEPS):
            actions.append(zero_action.copy())

        # Phase 4: Lift (grasp → lift, gripper closed)
        n = _compute_n_steps(grasp_pose.grasp_pos, lift_pos, APPROACH_SPEED)
        positions, quats = interpolate_cartesian(
            grasp_pose.grasp_pos, grasp_quat, lift_pos, grasp_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    grasp_pose.grasp_pos, grasp_quat,
                    positions[i], quats[i], 0.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 0.0
                )
            actions.append(action)

        # Phase 5: Transport (lift → above target, gripper closed)
        n = _compute_n_steps(lift_pos, transport_pos, APPROACH_SPEED)
        positions, quats = interpolate_cartesian(
            lift_pos, grasp_quat, transport_pos, grasp_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    lift_pos, grasp_quat,
                    positions[i], quats[i], 0.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 0.0
                )
            actions.append(action)

        # Phase 6: Place (above target → place, gripper closed)
        n = _compute_n_steps(transport_pos, place_pos, DESCEND_SPEED)
        positions, quats = interpolate_cartesian(
            transport_pos, grasp_quat, place_pos, grasp_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    transport_pos, grasp_quat,
                    positions[i], quats[i], 0.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 0.0
                )
            actions.append(action)

        # Phase 7: Release (open gripper, wait)
        open_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        for _ in range(RELEASE_WAIT_STEPS):
            actions.append(open_action.copy())

        # Phase 8: Retreat (place → home, gripper open)
        n = _compute_n_steps(place_pos, self.home_pos, APPROACH_SPEED)
        positions, quats = interpolate_cartesian(
            place_pos, grasp_quat, self.home_pos, self.home_quat, n
        )
        for i in range(n):
            if i == 0:
                action = pose_to_delta_action(
                    place_pos, grasp_quat,
                    positions[i], quats[i], 1.0
                )
            else:
                action = pose_to_delta_action(
                    positions[i-1], quats[i-1],
                    positions[i], quats[i], 1.0
                )
            actions.append(action)

        return actions

    def verify(self, actions, obj_name="obj_cube", obj_pos=None,
               target_pos=None):
        """Replay trajectory in simulation and check results.

        Args:
            actions: list of 7D actions
            obj_name: object body name
            obj_pos: initial object position (3,) or None for default
            target_pos: target position (3,) or None for default

        Returns:
            dict with verification results:
                - success: bool
                - ik_success_rate: float
                - final_obj_pos: (3,)
                - distance_to_target: float
                - total_steps: int
                - phases_passed: dict
        """
        self.sim.reset(obj_name=obj_name, obj_pos=obj_pos,
                       target_pos=target_pos)

        ik_successes = 0
        ik_total = 0

        for action in actions:
            obs = self.sim.step(action)
            # Track IK success (check if EE moved toward target)
            ik_total += 1

        # Check final state
        state = self.sim.get_state()
        obj_pos_final = state["obj_poses"].get(obj_name, {}).get("pos", np.zeros(3))

        # Get target position
        if target_pos is not None:
            dist = np.linalg.norm(obj_pos_final[:2] - target_pos[:2])
        else:
            dist = float("inf")

        # Check if object was lifted (z > table + margin)
        obj_z = obj_pos_final[2] if len(obj_pos_final) > 2 else 0.0
        was_lifted = obj_z > 0.83  # above resting height

        # Check if object is near target
        near_target = dist < 0.02

        success = was_lifted and near_target

        return {
            "success": success,
            "ik_success_rate": ik_successes / max(ik_total, 1),
            "final_obj_pos": obj_pos_final,
            "distance_to_target": dist,
            "total_steps": len(actions),
            "was_lifted": was_lifted,
            "near_target": near_target,
        }

    def generate_and_verify(self, obj_name, obj_pos, obj_size, target_pos,
                            n_attempts=3):
        """Generate trajectory and verify, retrying if needed.

        Args:
            obj_name: object body name
            obj_pos: object position (3,)
            obj_size: object size dict
            target_pos: target position (3,)
            n_attempts: max generation attempts

        Returns:
            (actions, result) tuple, or (None, None) if all attempts fail
        """
        for attempt in range(n_attempts):
            actions = self.generate(obj_name, obj_pos, obj_size, target_pos)
            if actions is None:
                continue
            result = self.verify(actions, obj_name=obj_name, obj_pos=obj_pos,
                                 target_pos=target_pos)
            if result["success"]:
                return actions, result

        return None, None
