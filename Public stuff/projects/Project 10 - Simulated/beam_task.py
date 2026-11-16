"""mjlab task "Beam-Drive-Straight": teach the two-beam robot to drive in a straight line down +x.

The robot is the Onshape design (onshape/), built by robot_model.py: a chassis, a front sphere that slides, and
two 60 mm beams (the legs), each turned about its centre by one output of the LEGO double motor. A leg moves only
in steps: forward 135 degrees at full speed, then hold (like LEGO's run_for_degrees(135)). The policy decides, 50 times a
second and for each leg, whether that leg starts its next step now. Stepping both legs at the same rate does not drive
straight (check_model.py: the robot drifts 35 degrees left in 5 s), so the policy must learn when to step each leg.

The robot starts at the origin pointing a little off +x and must drive forward at TARGET_SPEED while staying on the
line y = 0 and facing +x. Importing this file registers the task (see train.py, make_video.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

import robot_model as rm
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import (
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# (the motor and mass constants are in robot_model.py, shared with check_model.py)
TARGET_SPEED = 0.15         # m/s; check_model.py measured 0.21 m/s with both legs stepping nonstop
START_YAW = 0.4             # rad; the robot starts pointing up to this far off +x
START_BEAM_ANGLE = 0.3      # rad; each beam starts up to this far from flat (so the beams start a little out of step)
MAX_OFF_LINE = 0.5          # m; the episode ends if the robot wanders this far from y = 0
LINE_WIDTH = 0.05           # m; how sharply stay_on_line falls off with sideways distance
HEADING_WIDTH = 0.15        # rad; how sharply face_forward falls off with heading error
EPISODE_S = 10.0
FRICTION_RANGE = (0.6, 1.2)  # beam-floor friction, randomized per robot copy
MASS_ALPHA_RANGE = (-0.1, 0.1)  # chassis mass scaled by e^(2 alpha): about -18% to +22%
SPAWN_HEIGHT = 0.016        # m; chassis origin height at the start (front sphere just above the floor)
# -------------------------------------------------------------------------------------------------------

_ROBOT = SceneEntityCfg("robot")
_BEAMS = SceneEntityCfg("robot", joint_names=(rm.LEFT, rm.RIGHT))


def _get_spec() -> mujoco.MjSpec:
  spec = rm.build_spec()
  spec.body("chassis").add_freejoint()     # let the robot move around
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_HEIGHT), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # a position servo: it pulls each leg toward the angle the step action gives it, and holds it there
        BuiltinPdActuatorCfg(
          target_names_expr=(rm.LEFT, rm.RIGHT), stiffness=rm.BEAM_KP, damping=rm.BEAM_KD,
          effort_limit=rm.BEAM_TORQUE_LIMIT,
        ),
      ),
    ),
  )


# ---- the 135-degree step action ------------------------------------------------------------------------------


@dataclass(kw_only=True)
class LegStepActionCfg(BaseActionCfg):
  """One number per leg: above 0 means "start your next 135-degree step now" (ignored while a step is running)."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> LegStepAction:
    return LegStepAction(self, env)


class LegStepAction(BaseAction):
  """Same logic as robot_model.StepMotor, for thousands of robots at once.

  goal: the angle where the current step ends. setpoint: moves toward the goal at MAX_BEAM_SPEED every physics step;
  the position servo follows it. A leg can start a new step only when its setpoint has reached its goal.
  """

  def __init__(self, cfg: LegStepActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    # Always [left, right], whatever order the robot file lists the joints in (the LEGO program relies on it).
    order = [self._target_names.index(n) for n in (rm.LEFT, rm.RIGHT)]
    self._target_ids = self._target_ids[order]
    self._target_names = [self._target_names[i] for i in order]
    self._goal = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._setpoint = torch.zeros_like(self._goal)
    self.steps = torch.zeros_like(self._goal)            # steps taken this episode, per leg
    self._needs_sync = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._dt = env.cfg.sim.mujoco.timestep

  def _sync_after_reset(self):
    """After a reset the legs are at new random angles: start from wherever they are."""
    ids = self._needs_sync.nonzero(as_tuple=True)[0]
    if len(ids):
      q = self._entity.data.joint_pos[ids][:, self._target_ids]
      self._goal[ids] = q
      self._setpoint[ids] = q
      self.steps[ids] = 0
      self._needs_sync[ids] = False

  @property
  def stepping(self) -> torch.Tensor:
    """1 while a leg's step is running, else 0."""
    return (self._setpoint < self._goal - 1e-6).float()

  @property
  def remaining(self) -> torch.Tensor:
    """How far (rad) each leg still has to turn to finish its step."""
    return self._goal - self._entity.data.joint_pos[:, self._target_ids]

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._sync_after_reset()
    start = (actions > 0) & (self.stepping == 0)
    self._goal += start * rm.STEP_ANGLE
    self.steps += start
    self._processed_actions = start.float()

  def apply_actions(self) -> None:
    self._setpoint = torch.minimum(self._setpoint + rm.MAX_BEAM_SPEED * self._dt, self._goal)
    self._entity.set_joint_position_target(self._setpoint, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._needs_sync[slice(None) if env_ids is None else env_ids] = True


# ---- what the robot sees, and how it is scored --------------------------------------------------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this robot's start line, in metres."""
  return env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def _beam_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  ids = robot.find_joints((rm.LEFT, rm.RIGHT), preserve_order=True)[0]
  return robot.data.joint_pos[:, ids]


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: the heading, as the Double Motor's gyro reports it (yaw, zeroed at the start)."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def turn_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How fast the robot is turning (rad/s, + = to the left), as the gyro measures it. Shape [num_envs, 1]."""
  return env.scene["robot"].data.root_link_ang_vel_b[:, 2:3]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


def beam_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Each leg's speed (rad/s), [left, right]. Shape [num_envs, 2]."""
  robot = env.scene["robot"]
  ids = robot.find_joints((rm.LEFT, rm.RIGHT), preserve_order=True)[0]
  return robot.data.joint_vel[:, ids]


def beam_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """sin and cos of each beam's angle: where each beam is in its turn. Shape [num_envs, 4]."""
  q = _beam_angles(env)
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def leg_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """For each leg: is a step running (0 or 1), and how far it still has to go (rad). Shape [num_envs, 4]."""
  term = env.action_manager.get_term("legs")
  return torch.cat([term.stepping, term.remaining], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 at TARGET_SPEED along +x, 0 when standing still, less above the target, negative backwards."""
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


# The line and heading terms are bonuses between 0 and 1, not penalties. A penalty that grows with the distance
# (the first version) made drifting off quickly and ending the run cheaper than staying out there: the robot learned
# to give up. A bonus can only be lost, so staying on the line is always worth more.


# Both are also scaled by how fast the robot is moving forward (0 when standing still, 1 at TARGET_SPEED or faster).
# Without that, standing still on the line facing forward collected both bonuses in full, and the policy learned to
# walk less and less.


def _moving(env: ManagerBasedRlEnv) -> torch.Tensor:
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(vx / TARGET_SPEED, 0.0, 1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 on the line y = 0, about 0.37 at 5 cm off, almost 0 beyond 10 cm; times the moving factor."""
  return torch.exp(-torch.square(_off_line(env) / LINE_WIDTH)) * _moving(env)


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 facing +x, about 0.37 at 0.15 rad (9 degrees) off, almost 0 beyond 20 degrees; times the moving factor."""
  return torch.exp(-torch.square(_yaw(env) / HEADING_WIDTH)) * _moving(env)


def bounce(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Up-down speed plus rolling and pitching rate: how much the body is thrown around."""
  robot = env.scene["robot"]
  vz = robot.data.root_link_lin_vel_w[:, 2]
  wxy = robot.data.root_link_ang_vel_b[:, :2]
  return torch.square(vz) + 0.01 * torch.sum(torch.square(wxy), dim=-1)


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 every step the robot is still going. Without it, standing still scores below 0 (the penalties), so tipping
  over early would look better than surviving."""
  return torch.ones(env.num_envs, device=env.device)


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def beam_drive_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # The policy (actor) sees only what the real LEGO car can measure, so the same network can drive it:
  # the Double Motor's gyro (heading and turn rate) and its two motor encoders (leg angles and speeds), plus the
  # step state it keeps track of itself. It does NOT see its speed or how far it is from the line.
  # ORDER MATTERS: lego/lego_drive.py builds the same 13 numbers in this order. (The policy's own last output is
  # left out on purpose: with it, a policy that once said "wait" kept seeing "wait" and could stay stopped forever.)
  actor_terms = {
    "heading": ObservationTermCfg(func=heading),               # 2
    "turn_rate": ObservationTermCfg(func=turn_rate),           # 1
    "beam_phase": ObservationTermCfg(func=beam_phase),         # 4: sin left, sin right, cos left, cos right
    "beam_vel": ObservationTermCfg(func=beam_speed),           # 2: left, right
    "leg_steps": ObservationTermCfg(func=leg_steps),           # 4: stepping left, right, remaining left, right
  }
  # The critic only exists during training, so it may also see what the simulator knows: speed and the line offset.
  critic_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "beam_phase": ObservationTermCfg(func=beam_phase),
    "beam_vel": ObservationTermCfg(func=beam_speed),
    "leg_steps": ObservationTermCfg(func=leg_steps),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "legs": LegStepActionCfg(entity_name="robot", actuator_names=(rm.LEFT, rm.RIGHT), preserve_order=True),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    "reset_beams": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-START_BEAM_ANGLE, START_BEAM_ANGLE),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": _BEAMS,
      },
    ),
    # Domain randomization: every robot copy gets its own beam grip and body mass, so the policy cannot rely on
    # one exact value (the real car's are not known exactly).
    "beam_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(f"{rm.LEFT}_geom", f"{rm.RIGHT}_geom")),
        "operation": "abs",
        "ranges": FRICTION_RANGE,
      },
    ),
    "body_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("chassis",)), "alpha_range": MASS_ALPHA_RANGE},
    ),
  }

  rewards = {
    "alive": RewardTermCfg(func=alive, weight=0.5),
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "bounce": RewardTermCfg(func=bounce, weight=-1.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.5,
      elevation=-20.0,
      azimuth=135.0,
    ),
    # nconmax/njmax: room per robot copy for the mesh contacts (chassis, sphere and beam ends on the floor)
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), nconmax=40, njmax=300),
    decimation=4,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def beam_drive_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},  # gentler first tries: random full-speed reversals flip this light robot,
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="beam_drive_straight",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1500,
  )


register_mjlab_task(
  task_id="Beam-Drive-Straight",
  env_cfg=beam_drive_straight_env_cfg(),
  play_env_cfg=beam_drive_straight_env_cfg(play=True),
  rl_cfg=beam_drive_straight_ppo_runner_cfg(),
)
