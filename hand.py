"""
hand.py — a MuJoCo hand driven by synergy coefficients.

Self-contained MJCF, no model downloads. Five digits, two flexion joints each,
which is the coarse version of the 15-angle set Santello measured.

The point of this file is that the control input is TWO NUMBERS. Everything
else is the fixed basis doing the coordination. If you can drive a plausible
grasp from two sliders, you have demonstrated the whole argument without
saying a word.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering

import numpy as np
import mujoco

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _finger(name, x, y, ang, scale=1.0):
    L1, L2 = 0.040 * scale, 0.030 * scale
    return f"""
      <body name="{name}_prox" pos="{x} {y} 0.02" euler="0 0 {ang}">
        <joint name="{name}_mcp" type="hinge" axis="0 1 0" range="-0.2 1.7"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.007" rgba=".82 .68 .60 1"/>
        <body name="{name}_dist" pos="{L1} 0 0">
          <joint name="{name}_pip" type="hinge" axis="0 1 0" range="-0.1 1.8"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.006" rgba=".86 .72 .64 1"/>
          <site name="{name}_tip" pos="{L2} 0 0" size="0.006" rgba="1 .3 .2 1"/>
        </body>
      </body>"""


def build_xml():
    fingers = (
        _finger("thumb",  0.005, -0.030, -55, 0.85) +
        _finger("index",  0.055, -0.018,   0) +
        _finger("middle", 0.058, -0.002,   0, 1.06) +
        _finger("ring",   0.055,  0.014,   0, 0.97) +
        _finger("pinky",  0.048,  0.029,   0, 0.82)
    )
    acts = "".join(
        f'<position joint="{f}_{j}" kp="8" kv="0.8" ctrlrange="-0.2 1.8"/>'
        for f in FINGERS for j in ("mcp", "pip"))
    return f"""
<mujoco model="synergy_hand">
  <!-- MuJoCo XML defaults to DEGREES for angles. Every joint range below is in
       radians, so this line is load-bearing: without it the fingers are
       limited to about 1.7 degrees of travel and the hand silently refuses to
       close while every other number looks fine. -->
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>
  <default>
    <!-- Damping and armature are what keep a position-controlled finger from
         ringing itself into a NaN. Without them the solver diverges in a few
         hundred steps and every readout silently freezes. -->
    <joint damping="0.12" armature="0.002" limited="true"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.4 -1" diffuse=".9 .9 .9"/>
    <camera name="view" pos="0.075 -0.26 0.135" xyaxes="1 0 0 0 0.45 0.89"/>
    <body name="palm" pos="0 0 0.1">
      <geom type="box" size="0.028 0.032 0.010" rgba=".78 .64 .56 1"/>
      {fingers}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


class SynergyHand:
    """Two numbers in, twenty coordinated joint commands out."""

    def __init__(self, basis, joint_names=None):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.basis = basis
        # Map the 15 measured angles onto the 10 actuated joints. The four
        # abductions and thumb opposition are measured but not actuated here,
        # which is itself an honest statement about hardware: prosthetic hands
        # actuate far fewer degrees of freedom than a real hand has.
        self.actuated_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def set_synergy(self, coeffs):
        pose_deg = self.basis.decode(np.atleast_2d(coeffs))[0]
        ctrl = np.deg2rad(pose_deg[self.actuated_idx])
        self.data.ctrl[:] = np.clip(ctrl, -0.2, 1.8)

    def settle(self, steps=400):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        return self.fingertips()

    def fingertips(self):
        return np.array([self.data.site(f"{f}_tip").xpos.copy() for f in FINGERS])

    def aperture(self):
        """Thumb-to-index tip distance. The number a clinician cares about."""
        t = self.fingertips()
        return float(np.linalg.norm(t[0] - t[1]))

    def render(self, w=900, h=700):
        r = mujoco.Renderer(self.model, height=h, width=w)
        mujoco.mj_forward(self.model, self.data)
        r.update_scene(self.data, camera="view")
        img = r.render()
        r.close()
        return img
