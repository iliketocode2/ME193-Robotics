"""Record a trained Beam-Drive-Straight policy: a video, a table, a top view of the path, and policy.npz.

    python make_video.py model_1499.pt          # writes policy.mp4, trajectory.csv, trajectory.png

Made for the cluster, which has no screen (run it with runIt.py, see README.md). The robot starts pointing
START_YAW_TEST radians off the +x axis, so the video shows it correcting its heading and then driving straight.
"""
import glob
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import beam_task  # registers "Beam-Drive-Straight"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "Beam-Drive-Straight"
STEPS = 500             # 10 seconds at 50 policy steps per second
START_YAW_TEST = 0.3    # rad; starting heading error for the video

if len(sys.argv) != 2:
  sys.exit("usage: make_video.py model_1499.pt")
checkpoint = Path(sys.argv[1])

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
env_cfg.events["reset_pose"].params["pose_range"] = {"yaw": (START_YAW_TEST, START_YAW_TEST)}
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder="video", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

# Save the policy network as plain numpy arrays for the LEGO program (lego/lego_drive.py runs it without torch).
ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["actor_state_dict"]
import numpy as np  # noqa: E402
np.savez(
  "policy.npz",
  obs_mean=ck["obs_normalizer._mean"].numpy()[0], obs_std=ck["obs_normalizer._std"].numpy()[0], obs_eps=np.float32(1e-2),
  w0=ck["mlp.0.weight"].numpy(), b0=ck["mlp.0.bias"].numpy(), w1=ck["mlp.2.weight"].numpy(), b1=ck["mlp.2.bias"].numpy(),
  w2=ck["mlp.4.weight"].numpy(), b2=ck["mlp.4.bias"].numpy(),
  leg_order=np.array(env.unwrapped.action_manager.get_term("legs").target_names),
  step_angle_deg=np.float32(np.degrees(beam_task.rm.STEP_ANGLE)), step_speed_rad_s=np.float32(beam_task.rm.MAX_BEAM_SPEED),
)
print(f"wrote policy.npz (inputs: {ck['mlp.0.weight'].shape[1]}, leg order: "
      f"{env.unwrapped.action_manager.get_term('legs').target_names})")

robot = env.unwrapped.scene["robot"]
origin = env.unwrapped.scene.env_origins[0, :2].tolist()
rows = []
legs = env.unwrapped.action_manager.get_term("legs")
print("  second   x (m)   y (m)   heading (deg)   speed (m/s)   steps so far: left, right")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    x, y = (robot.data.root_link_pos_w[0, :2] - robot.data.root_link_pos_w.new_tensor(origin)).tolist()
    w, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    vx = robot.data.root_link_lin_vel_w[0, 0].item()
    steps = legs.steps[0].tolist()
    rows.append(((step + 1) / 50, x, y, math.degrees(yaw), vx, *steps))
    if step % 50 == 49:   # once a second
      print(f"  {(step + 1) / 50:5.0f}  {x:7.3f} {y:7.3f}  {math.degrees(yaw):11.1f}  {vx:12.3f}   "
            + ", ".join(f"{int(n):4d}" for n in steps))
env.close()

# the recorder writes video/<name>.mp4; give it a fixed name
videos = sorted(glob.glob("video/*.mp4"), key=os.path.getmtime)
if videos:
  shutil.move(videos[-1], "policy.mp4")
  shutil.rmtree("video", ignore_errors=True)
  print("wrote policy.mp4")
  try:   # a small GIF too: GitHub shows GIFs inside a README, but not MP4s
    import imageio.v2 as imageio
    frames = [f[::2, ::2] for i, f in enumerate(imageio.get_reader("policy.mp4")) if i % 3 == 0]
    imageio.mimsave("policy.gif", frames, duration=0.1, loop=0)
    print("wrote policy.gif")
  except Exception as e:  # noqa: BLE001
    print(f"(no GIF: {e})")

with open("trajectory.csv", "w") as f:
  f.write("time,x,y,heading_deg,vx,left_steps,right_steps\n")
  for r in rows:
    f.write(",".join(f"{v:.4f}" for v in r) + "\n")

try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  t, xs, ys, hs = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
  fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6))
  a1.plot(xs, ys, color="tab:orange", lw=2, label="robot path")
  a1.axhline(0, color="gray", ls="--", lw=1, label="the line y = 0")
  a1.set_xlabel("x, forward (m)"); a1.set_ylabel("y, sideways (m)"); a1.set_title("Top view")
  a1.set_aspect("equal", adjustable="datalim"); a1.legend(loc="best")
  a2.plot(t, hs, color="tab:blue")
  a2.axhline(0, color="gray", ls="--", lw=1)
  a2.set_xlabel("time (s)"); a2.set_ylabel("heading (deg)"); a2.set_title("Heading error")
  fig.tight_layout(); fig.savefig("trajectory.png", dpi=120)
  print("wrote trajectory.csv and trajectory.png")
except ImportError:
  print("wrote trajectory.csv (install matplotlib for trajectory.png)")
