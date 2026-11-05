# Public stuff

### 👉 Click here to try it

- [Q-Learning Visualizer](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/QLearn/) — train a robot with Q-learning
- [Reward Lab](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/reward-lab/) — design a reward and watch where a robot gets stuck
- [AnnaCode Builder](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/AnnaCode/) — neural networks on LEGO hardware
- [Kernels](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/Kernels/) — explore image convolution
- [LQR & MPC](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/LQR-MPC-web/) — steer a line follower, simulated or on the LEGO robot, with LQR and MPC
- [MQTT Debugging console](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/Debugging/)

The public-facing part of the ME193 Robotics course code (Fall 2026) — the
subset of the class repo that's meant to be shared.

## What's here

- [useful libraries/](useful%20libraries/) — the shared `lelib.py` wrapper
  around `legoeducation`, plus a `main.py` template and a hardware smoke
  test. Start here if you're writing your own code for the LEGO Education
  hardware.
- [Kernels/](Kernels/) — a single-page, no-install tool for exploring image
  convolution: load an image, threshold it, then edit a 3×3 kernel and hit
  "Convolve" to see the effect. [Try it online](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/Kernels/), or open `index.html` directly in a browser.
- [Controls/](Controls/) — a live PD-control demo: a Single Motor acts as a
  hand-turned dial, and a Double Motor drives to match its position. Kp/Kd
  sliders let you watch overshoot and oscillation change in real time.
- [virtualTesting/](virtualTesting/) — run `setup_test_env.py` to spin up a
  disposable virtual environment with `legoeducation` installed and drop
  into a `python3` REPL, no manual venv setup required.
- [Debugging/](Debugging/) — a single-page MQTT console: subscribe to and
  publish on any topic, on any broker, from the browser. [Try it online](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/Debugging/),
  no install needed.
- [mujoco-hpc/](mujoco-hpc/) — run MuJoCo physics simulations on the Tufts
  HPC cluster from your own laptop. `python3 setup.py` checks your login and
  installs everything; `python3 runIt.py --gpu 08_demo.py` runs a bouncing-ball
  simulation on a GPU node and brings the video home. `09_onshape_robot/`
  turns a two-wheeled robot designed in Onshape into a MuJoCo model and drives it.
  Edit `config.sh` with your Tufts username first.
- [LQR-MPC-web/](LQR-MPC-web/) — LQR vs MPC on a line-following robot, simulated or on LEGO hardware, in
  the browser (PyScript, no install; connects to the LEGO robot over Web Bluetooth). [Try it online](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/LQR-MPC-web/).
- [QLearn/](QLearn/) — a browser-based Q-learning visualizer (PyScript +
  Web Bluetooth): train a LEGO Education robot on Silly Walk, Line Follower
  and Maze Solver tasks while watching the Q-table update live. Needs
  Chrome or Edge; no install.
  [Open it online](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/QLearn/).
  Imported from
  [ceeoinnovations/FETLab-Summer-2026](https://github.com/ceeoinnovations/FETLab-Summer-2026/tree/main/QLearn-NewActivities-PyScript).
- [reward-lab/](reward-lab/) — a single-page reward-design exercise:
  a robot greedily takes the move with the best immediate reward, starting
  with "straight-line distance to the goal". Predict where it gets stuck, run
  it, then change the reward and see what happens. [Try it online](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/reward-lab/), no install needed.
