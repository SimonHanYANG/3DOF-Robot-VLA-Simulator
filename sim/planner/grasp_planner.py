"""Grasp pose generator for top-down grasping of geometric objects."""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class GraspPose:
    """Grasp pose definition."""
    pre_grasp_pos: np.ndarray    # (3,) pre-grasp position (above object)
    pre_grasp_quat: np.ndarray   # (4,) pre-grasp quaternion (wxyz MuJoCo format)
    grasp_pos: np.ndarray        # (3,) grasp position
    grasp_quat: np.ndarray       # (4,) grasp quaternion (wxyz)
    finger_width: float          # gripper control value for finger opening
    approach_dir: np.ndarray     # (3,) approach direction (unit vector)


# Top-down approach: gripper Z-axis points down in world frame
APPROACH_DIR = np.array([0.0, 0.0, -1.0])

# Pre-grasp height offset above grasp position
PRE_GRASP_OFFSET = 0.10  # 10cm

# Object size definitions (from scene.xml)
OBJ_SIZES = {
    "obj_cube": {"type": "cube", "half_size": 0.02},
    "obj_sphere": {"type": "sphere", "radius": 0.025},
    "obj_cylinder": {"type": "cylinder", "radius": 0.015, "half_height": 0.03},
}


def _top_down_quat(z_rotation=0.0):
    """Compute quaternion for top-down gripper orientation.

    Args:
        z_rotation: rotation around world Z-axis (radians)

    Returns:
        quat: (4,) quaternion in MuJoCo format (w, x, y, z)
    """
    R_base = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=float)
    if abs(z_rotation) > 1e-6:
        R_z = Rotation.from_rotvec([0, 0, z_rotation]).as_matrix()
        R = R_z @ R_base
    else:
        R = R_base
    scipy_quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return np.array([scipy_quat[3], scipy_quat[0],
                     scipy_quat[1], scipy_quat[2]])  # (w, x, y, z)


def _finger_width_for_object(obj_type, obj_size):
    """Compute gripper control value to grip an object.

    The gripper gap = 0.032 + 2*ctrl_value.
    We set gap = object_width * factor, where factor < 1 for secure grip.

    Args:
        obj_type: "cube"/"sphere"/"cylinder" or "obj_cube"/"obj_sphere"/"obj_cylinder"
        obj_size: dict with object dimensions

    Returns:
        finger_width: gripper control value
    """
    # Strip "obj_" prefix if present
    base_type = obj_type.replace("obj_", "") if obj_type.startswith("obj_") else obj_type

    MIN_GAP = 0.032  # gripper minimum gap (fingers at rest)

    if base_type == "cube":
        # Cube: grip at full side width (flat surfaces, secure contact)
        object_width = 2 * obj_size["half_size"]
        gap = object_width
    elif base_type == "sphere":
        # Sphere: grip at 80% of diameter (contact at ~equator level)
        object_width = 2 * obj_size["radius"]
        gap = object_width * 0.8
    elif base_type == "cylinder":
        # Cylinder: grip at full diameter (cylindrical contact)
        object_width = 2 * obj_size["radius"]
        gap = object_width
    else:
        raise ValueError(f"Unknown object type: {obj_type}")

    # Ensure gap is at least the gripper minimum
    gap = max(gap, MIN_GAP + 0.002)  # small margin above minimum

    # gap = 0.032 + 2 * finger_width
    finger_width = (gap - MIN_GAP) / 2.0
    return max(finger_width, 0.0)


def _grasp_z_offset(obj_type, obj_size):
    """Compute Z offset from object center to gripper weld site.

    The weld site is placed at the object center height.
    The palm bottom (weld_z - 0.033) sits slightly below the center,
    which positions the fingers around the object for a stable grasp.

    Returns:
        z_offset: offset to add to object Z position
    """
    return 0.0  # weld site at object center


class GraspPlanner:
    """Grasp pose planner for geometric objects."""

    def __init__(self, model=None, data=None):
        """Initialize planner.

        Args:
            model: MuJoCo model (optional, for future workspace validation)
            data: MuJoCo data (optional)
        """
        self.model = model
        self.data = data

    def plan_grasp(self, obj_type, obj_pos, obj_size, z_rotation=0.0):
        """Plan a single top-down grasp pose.

        Args:
            obj_type: "cube", "sphere", or "cylinder"
            obj_pos: object center position (3,)
            obj_size: dict with object dimensions
            z_rotation: rotation around Z-axis (radians)

        Returns:
            GraspPose
        """
        return self._top_down_grasp(obj_type, obj_pos, obj_size, z_rotation)

    def sample_grasps(self, obj_type, obj_pos, obj_size, n=10, rng=None):
        """Generate n candidate grasp poses with random perturbations.

        Args:
            obj_type: "cube", "sphere", or "cylinder"
            obj_pos: object center position (3,)
            obj_size: dict with object dimensions
            n: number of grasps to generate
            rng: numpy random generator (or None for default)

        Returns:
            list of GraspPose
        """
        if rng is None:
            rng = np.random.default_rng()

        grasps = []
        for _ in range(n):
            # Random perturbations
            dx = rng.uniform(-0.01, 0.01)
            dy = rng.uniform(-0.01, 0.01)
            dz_rot = rng.uniform(-np.deg2rad(5), np.deg2rad(5))
            dw = rng.uniform(-0.002, 0.002)

            perturbed_pos = obj_pos + np.array([dx, dy, 0.0])

            grasp = self._top_down_grasp(obj_type, perturbed_pos, obj_size,
                                         z_rotation=dz_rot)
            # Apply finger width perturbation
            grasp.finger_width = np.clip(grasp.finger_width + dw, 0.0, 0.045)
            grasps.append(grasp)

        return grasps

    def _top_down_grasp(self, obj_type, obj_pos, obj_size, z_rotation=0.0):
        """Construct a top-down grasp pose.

        Args:
            obj_type: "cube", "sphere", or "cylinder"
            obj_pos: object center position (3,)
            obj_size: dict with object dimensions
            z_rotation: rotation around Z-axis (radians)

        Returns:
            GraspPose
        """
        quat = _top_down_quat(z_rotation)
        z_offset = _grasp_z_offset(obj_type, obj_size)
        finger_width = _finger_width_for_object(obj_type, obj_size)

        grasp_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + z_offset])
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += PRE_GRASP_OFFSET

        return GraspPose(
            pre_grasp_pos=pre_grasp_pos,
            pre_grasp_quat=quat.copy(),
            grasp_pos=grasp_pos,
            grasp_quat=quat.copy(),
            finger_width=finger_width,
            approach_dir=APPROACH_DIR.copy(),
        )

    def _validate_grasp(self, grasp, table_z=0.81, max_reach=0.85):
        """Validate that a grasp pose is feasible.

        Args:
            grasp: GraspPose to validate
            table_z: table surface Z coordinate
            max_reach: maximum XY reach from base (approximate)

        Returns:
            True if grasp is feasible
        """
        # Grasp must be above table
        if grasp.grasp_pos[2] < table_z:
            return False
        if grasp.pre_grasp_pos[2] < table_z:
            return False

        # Pre-grasp must be above grasp by at least 0.08m
        if grasp.pre_grasp_pos[2] - grasp.grasp_pos[2] < 0.08:
            return False

        # Approximate workspace check: XY distance from base
        xy_dist = np.linalg.norm(grasp.grasp_pos[:2])
        if xy_dist > max_reach:
            return False

        # Finger width must be valid
        if grasp.finger_width < 0 or grasp.finger_width > 0.045:
            return False

        return True
