"""Drive the two-beam robot with fixed 135-degree steps and film it. No learning: this checks the model.

    python check_model.py                 # writes open_loop.mp4 and open_loop.csv
    python check_model.py --no-video      # table only (fast)

On the cluster, from the mujoco-hpc folder:
    python3 runIt.py --gpu --with 10_drive_straight/onshape --with 10_drive_straight/robot_model.py \
        10_drive_straight/check_model.py

Each leg moves like LEGO's run_for_degrees(135): forward 135 degrees at full speed, then hold. The table it prints
(x forward, y sideways, heading) is also what you can measure on the real LEGO car with the same steps, to see how
close the simulation is. The average speed in the "both legs" phase is the robot's top speed.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import mujoco
import numpy as np

import robot_model as rm

VIDEO = "--no-video" not in sys.argv
OUT = "open_loop.mp4"

# (until this second, label, rule): rule(time, side) says whether that leg should start its next step now
PLAN = [
    (1.0, "settle", lambda t, side: False),
    (6.0, "both legs", lambda t, side: True),                                   # step, step, step ... on both sides
    (9.0, "left only", lambda t, side: side == rm.LEFT),                        # only the left leg: turns right
    (12.0, "alternate", lambda t, side: (int(t / 0.25) % 2 == 0) == (side == rm.LEFT)),  # legs take turns
    (15.0, "both, slower", lambda t, side: (t % 0.5) < 0.02),                   # both legs, one step every 0.5 s
]


def build():
    spec = rm.build_spec()
    spec.body("chassis").add_freejoint()
    # a floor with 10 cm squares, and a light
    tex = spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           width=512, height=512, rgb1=[.3, .4, .5], rgb2=[.15, .2, .25])
    mat = spec.add_material(name="checker", texrepeat=[10, 10], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, .1], material="checker")
    spec.worldbody.add_light(pos=[0, -1, 3], dir=[0, .3, -1])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    spec.option.timestep = 0.002
    # position motors, like mjlab's: torque = KP * (target angle - angle) - KD * speed, capped at the torque limit
    for side in (rm.LEFT, rm.RIGHT):
        a = spec.add_actuator(name=side, target=side, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = rm.BEAM_KP, -rm.BEAM_KP, -rm.BEAM_KD
        a.forcelimited, a.forcerange = True, [-rm.BEAM_TORQUE_LIMIT, rm.BEAM_TORQUE_LIMIT]
    return spec.compile()


def lowest_point(model, data):
    """Lowest world z over all collision geoms (corners of their bounding boxes)."""
    low = np.inf
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 or model.geom_bodyid[g] == 0:
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T
        low = min(low, world[:, 2].min())
    return low


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


model = build()
data = mujoco.MjData(model)
rm.describe(model)
mujoco.mj_forward(model, data)
data.qpos[2] += 0.002 - lowest_point(model, data)   # just above the floor
act = {side: model.actuator(side).id for side in (rm.LEFT, rm.RIGHT)}
qadr = {side: model.jnt_qposadr[model.joint(side).id] for side in (rm.LEFT, rm.RIGHT)}
motor = {side: rm.StepMotor(data.qpos[qadr[side]]) for side in (rm.LEFT, rm.RIGHT)}

renderer = mujoco.Renderer(model, 480, 640) if VIDEO else None
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
camera.trackbodyid = model.body("chassis").id
camera.distance, camera.azimuth, camera.elevation = 0.5, 135, -20

FPS = 30
frames, rows = [], []
nframe, phase_start = 0, (0.0, 0.0)
print(f"\n  time  phase         x (m)   y (m)   heading (deg)   speed (m/s)   steps left, right")
for end, label, rule in PLAN:
    start_x = data.qpos[0]
    start_t = data.time
    while data.time < end - 1e-9:
        while data.time < nframe / FPS:
            for side, m in motor.items():
                m.command(rule(data.time, side))
                data.ctrl[act[side]] = m.update(model.opt.timestep)
            mujoco.mj_step(model, data)
        nframe += 1
        if renderer:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
        if nframe % FPS == 0:
            x, y, h = data.qpos[0], data.qpos[1], np.degrees(yaw(data.qpos[3:7]))
            rows.append((data.time, label, x, y, h, data.qvel[0]))
            print(f"  {data.time:4.0f}  {label:<12} {x:7.3f} {y:7.3f}  {h:11.1f}  {data.qvel[0]:12.3f}"
                  f"   {motor[rm.LEFT].steps:4d}, {motor[rm.RIGHT].steps:4d}")
    if label == "both legs":
        straight_speed = (data.qpos[0] - start_x) / (data.time - start_t)

with open("open_loop.csv", "w") as f:
    f.write("time,phase,x,y,heading_deg,vx\n")
    for r in rows:
        f.write(f"{r[0]:.2f},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.2f},{r[5]:.4f}\n")
print(f"\nAverage forward speed with both legs stepping nonstop: {straight_speed:.3f} m/s "
      f"(the top speed; TARGET_SPEED in beam_task.py should be below this)")
if renderer:
    import imageio
    renderer.close()
    imageio.mimwrite(OUT, frames, fps=FPS)
    imageio.mimsave("open_loop.gif", [f[::2, ::2] for f in frames[::3]], duration=0.1, loop=0)
    print(f"wrote {OUT}: {len(frames)} frames, open_loop.gif, open_loop.csv")
