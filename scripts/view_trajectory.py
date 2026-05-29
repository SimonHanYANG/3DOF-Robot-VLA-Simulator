"""Visualize pick-and-place trajectory in MuJoCo interactive viewer.

Usage:
    python scripts/view_trajectory.py                  # default cube
    python scripts/view_trajectory.py --obj obj_sphere # sphere
    python scripts/view_trajectory.py --obj obj_cylinder # cylinder
    python scripts/view_trajectory.py --slow           # slow motion (2x)
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer

from sim.controller import SimController, HOME_QPOS
from sim.planner.grasp_planner import GraspPlanner, OBJ_SIZES
from sim.planner.pick_and_place import PickPlaceTrajectoryGenerator

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                            "sim", "assets", "scene.xml"))

# Default object positions (from scene.xml)
OBJ_DEFAULTS = {
    "obj_cube":     {"pos": np.array([0.3, 0.1, 0.83]),
                     "size": {"half_size": 0.02}},
    "obj_sphere":   {"pos": np.array([0.5, -0.1, 0.815]),
                     "size": {"radius": 0.025}},
    "obj_cylinder": {"pos": np.array([0.4, 0.2, 0.83]),
                     "size": {"radius": 0.015, "half_height": 0.03}},
}

TARGET_POS = np.array([0.5, 0.0, 0.83])


def main():
    parser = argparse.ArgumentParser(description="Visualize pick-and-place trajectory")
    parser.add_argument("--obj", type=str, default="obj_cube",
                        choices=["obj_cube", "obj_sphere", "obj_cylinder"],
                        help="Object to pick up")
    parser.add_argument("--target", type=float, nargs=3, default=[0.5, 0.0, 0.83],
                        help="Target position (x y z)")
    parser.add_argument("--slow", action="store_true",
                        help="Slow motion (2x slower)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override timestep between steps (seconds)")
    args = parser.parse_args()

    obj_name = args.obj
    target_pos = np.array(args.target)
    params = OBJ_DEFAULTS[obj_name]

    print(f"Object: {obj_name}")
    print(f"Object pos: {params['pos']}")
    print(f"Target pos: {target_pos}")

    # Create sim and generate trajectory
    sim = SimController(SCENE_PATH)
    planner = GraspPlanner()
    gen = PickPlaceTrajectoryGenerator(sim, planner)

    print("Generating trajectory...")
    actions = gen.generate(obj_name, params["pos"], params["size"], target_pos)
    if actions is None:
        print("ERROR: Failed to generate trajectory")
        return
    print(f"Trajectory: {len(actions)} steps")

    # Reset sim
    sim.reset(obj_name=obj_name, obj_pos=params["pos"], target_pos=target_pos)

    # Determine delay
    if args.dt is not None:
        dt = args.dt
    elif args.slow:
        dt = sim.model.opt.timestep * sim.substeps * 2
    else:
        dt = sim.model.opt.timestep * sim.substeps

    # Launch viewer
    print("Opening viewer... (close window to exit)")
    print("  - Drag to rotate camera")
    print("  - Scroll to zoom")
    print("  - Right-drag to pan")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Sync initial state
        viewer.sync()

        for step_idx, action in enumerate(actions):
            # Step simulation
            sim.step(action)

            # Update viewer
            viewer.sync()

            # Print progress every 20 steps
            if step_idx % 20 == 0:
                state = sim.get_state()
                ee = state["ee_pose"][:3]
                gripper = state["gripper_qpos"]
                obj_z = state["obj_poses"][obj_name]["pos"][2]
                print(f"  Step {step_idx:3d}/{len(actions)}: "
                      f"EE=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}] "
                      f"gripper={gripper:.3f} obj_z={obj_z:.3f}")

            # Wait to match real-time
            time.sleep(dt)

            # Check if viewer is still open
            if not viewer.is_running():
                print("Viewer closed by user")
                break

        # Final state
        state = sim.get_state()
        obj_pos = state["obj_poses"][obj_name]["pos"]
        dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        print(f"\nFinal object pos: {obj_pos}")
        print(f"Distance to target: {dist:.4f}m")
        if dist < 0.02:
            print("SUCCESS: Object placed near target!")
        else:
            print("Object not at target (may need tuning)")

        # Keep viewer open until user closes it
        print("\nTrajectory complete. Close viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)

    sim.renderer.close()


if __name__ == "__main__":
    main()
