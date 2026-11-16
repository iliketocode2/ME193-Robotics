"""Build the two-beam robot from the Onshape export. Used by check_model.py and beam_task.py.

The Onshape export (onshape/robot.xml) has the chassis, the front sphere and the two beams. If the
assembly has no revolute mates for the beams (named dof_...), the exporter writes each beam as a loose
part. In that case this file attaches each beam to the chassis itself, with a motor axle through the
beam's centre, pointing across the robot (y). If you do add the mates in Onshape, their joints are used instead.

Everything you may want to match to the real LEGO car is in the constants block.
"""
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())

# ---- constants to match to the real car -----------------------------------------------------------
LEFT, RIGHT = "left_beam", "right_beam"   # joint names (left = +y side when the robot faces +x)
BEAM_MASS = 0.005           # kg per beam (Onshape gave the beams no material)
TOTAL_MASS = None           # kg; weigh the LEGO car and put it here (None keeps Onshape's chassis mass, 0.118 kg)
BEAM_FRICTION = 1.0         # grip of the beam ends on the floor
SLIDE_FRICTION = 0.05       # chassis and front sphere: they slide (the sphere stands in for a ball caster)
# LEGO Education double motor. Placeholders until you measure yours:
MAX_BEAM_SPEED = 6.0        # rad/s (about 57 rpm): the speed of a 135 degree step (0.4 s). Halved from 12 to match the real car
BEAM_TORQUE_LIMIT = 0.15    # N*m stall torque
# The motor works like LEGO's run_for_degrees(135) followed by "hold": it turns its leg forward 135 degrees at
# MAX_BEAM_SPEED, then holds that angle until it is told to step again.
STEP_ANGLE = np.radians(135.0)
BEAM_KP = 2.0               # N*m per rad: how hard the motor holds / pulls toward its target angle
BEAM_KD = 0.05              # N*m per rad/s: damping (the motor resists fast motion)


class StepMotor:
    """The 135-degree stepping motor for plain MuJoCo (check_model.py). beam_task.py has the same logic for mjlab.

    `goal` is where the current step ends; `setpoint` moves toward it at MAX_BEAM_SPEED and is what the position
    servo follows. A new step can only start once the setpoint has reached the goal.
    """

    def __init__(self, start_angle=0.0):
        self.goal = self.setpoint = float(start_angle)
        self.steps = 0

    @property
    def stepping(self):
        return self.setpoint < self.goal - 1e-9

    def command(self, step: bool):
        if step and not self.stepping:
            self.goal += STEP_ANGLE
            self.steps += 1

    def update(self, dt):
        self.setpoint = min(self.setpoint + MAX_BEAM_SPEED * dt, self.goal)
        return self.setpoint
# ----------------------------------------------------------------------------------------------------


def _beam_bodies(spec, model, data):
    """Bodies other than the chassis that carry geometry: the beams. Returns [(name, world centre)]."""
    out = []
    for b in range(1, model.nbody):
        name = model.body(b).name
        if name == "chassis":
            continue
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        if not geoms:
            continue
        g = geoms[0]
        centre = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3]
        out.append((name, centre))
    return out


def build_spec(xml=ROBOT_XML) -> mujoco.MjSpec:
    """The robot with two beam hinges, no actuators and no freejoint (callers add those)."""
    spec = mujoco.MjSpec.from_file(str(xml))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    chassis = spec.body("chassis")
    hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]

    if len(hinges) < 2:
        # No mates in Onshape: rebuild each loose beam as a child of the chassis with an axle through its centre.
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        chassis_y = data.xipos[model.body("chassis").id][1]
        beams = _beam_bodies(spec, model, data)
        if len(beams) != 2:
            raise SystemExit(f"expected 2 beams in {xml}, found {[b[0] for b in beams]}")
        for old_name, centre in beams:
            old = spec.body(old_name)
            side = LEFT if centre[1] > chassis_y else RIGHT
            new = chassis.add_body(name=side, pos=centre)
            new.add_joint(name=side, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], armature=1e-4)
            for g in old.geoms:
                is_collision = g.contype != 0 or g.conaffinity != 0
                new.add_geom(
                    name=f"{side}_geom" if is_collision else "",
                    type=g.type, meshname=g.meshname, rgba=[1.0, 0.55, 0.1, 1],   # orange, to see them turn
                    pos=np.array(g.pos) - centre, quat=g.quat,
                    contype=g.contype, conaffinity=g.conaffinity, group=g.group,
                    mass=BEAM_MASS if is_collision else 0.0,
                    friction=[BEAM_FRICTION, 0.005, 0.0001],
                )
            spec.delete(old)
    else:
        # Mates exist: rename them to LEFT / RIGHT by which side of the robot they are on.
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ys = {j.name: data.xanchor[model.joint(j.name).id][1] for j in hinges[:2]}
        left_name = max(ys, key=ys.get)
        for j in hinges[:2]:
            side = LEFT if j.name == left_name else RIGHT
            # The mate arrows may point opposite ways. Flip any axle that points toward -y, so that a positive
            # speed turns both beams the same way (forward) whichever way the mate was drawn in Onshape.
            if data.xaxis[model.joint(j.name).id][1] < 0:
                j.axis = -np.array(j.axis)
            j.name = side
            j.limited = mujoco.mjtLimited.mjLIMITED_FALSE   # a beam spins round and round: no angle limits
            j.armature, j.frictionloss = 1e-4, 0.0
            beam = j.parent
            beam.name = side
            if beam.mass < 1e-4:   # the beam has no material in Onshape: give it a LEGO beam's mass
                beam.mass = BEAM_MASS
                beam.fullinertia = [1.5e-6, 1.5e-6, 1.5e-6, 0, 0, 0]
            for g in beam.geoms:
                g.rgba, g.material = [1.0, 0.55, 0.1, 1], ""   # orange, to see them turn
                if g.contype != 0 or g.conaffinity != 0:
                    g.name = f"{side}_geom"   # beam_task.py randomizes this geom's friction
                    g.friction[0] = BEAM_FRICTION

    for g in chassis.geoms:   # only the beams grip; the chassis and sphere slide
        g.friction[0] = SLIDE_FRICTION
    if TOTAL_MASS is not None:
        scale = (TOTAL_MASS - 2 * BEAM_MASS) / chassis.mass
        chassis.mass *= scale
        chassis.fullinertia = np.array(chassis.fullinertia) * scale
    return spec


def describe(model):
    """Print what the robot is made of (names, masses, beam length)."""
    print("joints:   ", [model.joint(i).name for i in range(model.njnt)])
    print("bodies:   ", ", ".join(f"{model.body(b).name} {1000 * model.body_mass[b]:.0f} g" for b in range(1, model.nbody)))
    for side in (LEFT, RIGHT):
        b = model.body(side).id
        g = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g])
        print(f"{side}: {2000 * model.geom_aabb[g, 3:].max():.0f} mm long")
