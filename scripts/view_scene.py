"""Interactive scene viewer — open MuJoCo passive viewer to inspect the UR5e scene.

Controls:
  Left drag   — rotate
  Right drag  — pan
  Scroll      — zoom
  Space       — pause/resume simulation
  Backspace   — reset
  Esc         — quit
"""

import os
import time
import mujoco
import mujoco.viewer

SCENE_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "assets", "scene.xml"))


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Set gripper open
    data.ctrl[6] = 0.045  # gripper_left
    data.ctrl[7] = 0.045  # gripper_right
    mujoco.mj_forward(model, data)

    print("Opening MuJoCo viewer for UR5e pick-and-place scene...")
    print(f"  Bodies: {model.nbody}, Geoms: {model.ngeom}, Actuators: {model.nu}")
    print("  Left drag=rotate  Right drag=pan  Scroll=zoom")
    print("  Space=pause  Backspace=reset  Esc=quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
