"""Verify MuJoCo installation by loading an empty scene and stepping 1000 steps."""

import mujoco
import numpy as np


def main():
    print(f"MuJoCo version: {mujoco.__version__}")

    # Create a minimal scene: a ground plane
    xml = """
    <mujoco>
        <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <body name="sphere" pos="0 0 0.1">
                <freejoint/>
                <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Model loaded: {model.nbody} bodies, {model.ngeom} geoms")
    mujoco.mj_forward(model, data)
    print("Stepping 1000 steps...")

    for _ in range(1000):
        mujoco.mj_step(model, data)

    print(f"Done. Sphere position: {data.body('sphere').xpos}")
    print("MuJoCo verification PASSED.")


if __name__ == "__main__":
    main()
