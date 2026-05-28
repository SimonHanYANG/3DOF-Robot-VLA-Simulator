"""Inverse kinematics solver using Damped Least Squares (DLS) Jacobian method."""

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

# UR5e joint names in order
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

DEFAULT_EEF_BODY = "wrist_3_link"


def _get_arm_joint_indices(model):
    """Return (qpos_indices, qvel_indices) for the 6 arm joints."""
    qpos_idx = []
    qvel_idx = []
    for name in ARM_JOINT_NAMES:
        jid = model.joint(name).id
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])
    return np.array(qpos_idx), np.array(qvel_idx)


def _get_body_id(model, body_name):
    """Get body ID, raise if not found."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found in model")
    return bid


def _compute_error(target_pos, target_quat, current_pos, current_quat):
    """Compute 6D error vector (3 position + 3 rotation).

    Args:
        target_pos: target position (3,)
        target_quat: target quaternion in MuJoCo format (w, x, y, z)
        current_pos: current position (3,)
        current_quat: current quaternion in MuJoCo format (w, x, y, z)

    Returns:
        error: 6D error vector (6,)
    """
    # Position error
    e_pos = target_pos - current_pos

    # Rotation error via angle-axis
    # MuJoCo quat is (w, x, y, z), scipy wants (x, y, z, w)
    R_target = Rotation.from_quat(target_quat[[1, 2, 3, 0]]).as_matrix()
    R_current = Rotation.from_quat(current_quat[[1, 2, 3, 0]]).as_matrix()
    R_err = R_target @ R_current.T
    e_rot = Rotation.from_matrix(R_err).as_rotvec()

    return np.concatenate([e_pos, e_rot])


def solve_ik(model, data, target_pos, target_quat, body_name=DEFAULT_EEF_BODY,
             damping=0.01, max_iter=50, pos_tol=1e-4, rot_tol=1e-3):
    """Damped Least Squares IK solver.

    Args:
        model, data: MuJoCo model and data
        target_pos: target position (3,)
        target_quat: target quaternion (w, x, y, z) MuJoCo format
        body_name: end-effector body name
        damping: DLS damping coefficient lambda
        max_iter: max iterations
        pos_tol: position convergence threshold (m)
        rot_tol: rotation convergence threshold (rad)

    Returns:
        success: whether IK converged
        q_sol: solved joint angles (6,)
        info: dict with iteration count and final error
    """
    qpos_idx, dof_idx = _get_arm_joint_indices(model)
    body_id = _get_body_id(model, body_name)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for i in range(max_iter):
        mujoco.mj_forward(model, data)

        # Current end-effector pose
        current_pos = data.body(body_id).xpos.copy()
        current_quat = data.body(body_id).xquat.copy()

        # Check convergence
        e = _compute_error(target_pos, target_quat, current_pos, current_quat)
        pos_err = np.linalg.norm(e[:3])
        rot_err = np.linalg.norm(e[3:])

        if pos_err < pos_tol and rot_err < rot_tol:
            return True, data.qpos[qpos_idx].copy(), {
                "iterations": i, "pos_err": pos_err, "rot_err": rot_err
            }

        # Compute Jacobian
        jacp[:] = 0
        jacr[:] = 0
        body_pos = data.body(body_id).xpos
        mujoco.mj_jac(model, data, jacp, jacr, body_pos, body_id)

        # Extract arm columns: 6x6
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])  # (6, 6)

        # DLS: dq = J^T @ inv(J @ J^T + lambda^2 * I) @ e
        JJT = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)

        # Apply joint increment
        data.qpos[qpos_idx] += dq

    # Did not converge, return best effort
    mujoco.mj_forward(model, data)
    current_pos = data.body(body_id).xpos.copy()
    current_quat = data.body(body_id).xquat.copy()
    e = _compute_error(target_pos, target_quat, current_pos, current_quat)
    return False, data.qpos[qpos_idx].copy(), {
        "iterations": max_iter,
        "pos_err": np.linalg.norm(e[:3]),
        "rot_err": np.linalg.norm(e[3:])
    }


def ik_from_cartesian_delta(model, data, current_q, dx, dy, dz,
                            droll, dpitch, dyaw,
                            body_name=DEFAULT_EEF_BODY,
                            damping=0.01, max_iter=50):
    """Convert Cartesian delta to joint angle change.

    Args:
        model, data: MuJoCo model and data
        current_q: current joint angles (6,)
        dx, dy, dz: position deltas (m)
        droll, dpitch, dyaw: rotation deltas (rad), Euler ZYX convention
        body_name: end-effector body name
        damping: DLS damping coefficient
        max_iter: max iterations

    Returns:
        success: whether IK converged
        new_q: new joint angles (6,)
        info: dict with iteration count and final error
    """
    qpos_idx, _ = _get_arm_joint_indices(model)
    body_id = _get_body_id(model, body_name)

    # Set joints to current_q to compute current EE pose
    data.qpos[qpos_idx] = current_q
    mujoco.mj_forward(model, data)

    current_pos = data.body(body_id).xpos.copy()
    current_quat = data.body(body_id).xquat.copy()

    # Compute target pose = current + delta
    target_pos = current_pos + np.array([dx, dy, dz])

    # Apply rotation delta
    R_current = Rotation.from_quat(current_quat[[1, 2, 3, 0]])
    R_delta = Rotation.from_euler("ZYX", [droll, dpitch, dyaw])
    R_target = R_delta * R_current
    target_quat_scipy = R_target.as_quat()  # (x, y, z, w)
    target_quat = np.array([target_quat_scipy[3], target_quat_scipy[0],
                            target_quat_scipy[1], target_quat_scipy[2]])  # (w, x, y, z)

    # Solve IK
    success, q_sol, info = solve_ik(
        model, data, target_pos, target_quat,
        body_name=body_name, damping=damping, max_iter=max_iter
    )

    return success, q_sol, info
