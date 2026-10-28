# MuJoCo on the Tufts HPC cluster

Run MuJoCo simulations on the Tufts cluster from your own laptop: three Python commands, no cluster
knowledge needed. Everything is logged so that, if something breaks, you can send one file to get help.


## Quick start

Run everything on your **laptop**, in a terminal opened **inside this folder** (VSCode: open the
`mujoco-hpc` folder, then Terminal > New Terminal; check with `pwd`). Off campus, connect the Tufts VPN first.

1. Edit `config.sh` once: put your Tufts username in `HPC_USER`.
2. Check the whole setup (about 2 minutes; you type your password / Duo once):
   ```
   python3 setup.py
   ```
   Every step prints PASS or FAIL. All PASS means MuJoCo works on the cluster, on CPU and GPU nodes.
3. Run the demo, a ball that drops and bounces:
   ```
   python3 runIt.py --gpu 08_demo.py
   ```
   It runs on a GPU node, copies the video back to `runs/<name>/bounce.mp4`, and opens it.

Use `python3` on Mac, Linux and WSL. On a Windows PC, use WSL (see [Windows](#windows)).

## The three commands

| Command | What it does |
|---|---|
| `python3 setup.py` | Logs in, copies files, installs MuJoCo into `~/mujoco-env` on the cluster, then tests a quick run, a CPU job and a GPU job. `--skip-gpu` skips the GPU job; `--reinstall` rebuilds the environment. |
| `python3 runIt.py [options] yourscript.py [arguments]` | Copies your script to a fresh folder on the cluster, submits it with Slurm, waits, prints its output, and copies everything it wrote back to `runs/<name>/`. Videos and images open automatically. |
| `python3 status.py` | Shows what is queued, running and recently finished, from any terminal at any time. |

`runIt.py` options (they work before or after the script name; anything else goes to your script):

| Option | Meaning | Default |
|---|---|---|
| `--gpu` | run on a GPU node | CPU node |
| `--time 01:00:00` | maximum run time (the job is stopped after this) | 30 minutes |
| `--mem 8G` | memory | 4G |
| `--cpus 4` | CPU cores | 2 |
| `--with helper.py` | also copy another file next to the script (repeat for more) | none |
| `--partition NAME` | use a different Slurm queue | `batch`, or `gpu` with `--gpu` |
| `--no-open` | do not open videos when done | opens them |

Example: `python3 runIt.py --gpu --time 01:00:00 my_sim.py 10 out.mp4` runs `my_sim.py 10 out.mp4` on a GPU node
for up to an hour. If one of *your* script's arguments has the same name as a `runIt.py` option, put `--` before
your arguments.

`status.py` options:

| Command | Meaning |
|---|---|
| `python3 status.py` | queued / running now, plus jobs from the last 2 days |
| `python3 status.py 6518997` | one job's state and the end of its output |
| `python3 status.py --watch` | refresh every 10 s until Ctrl-C (add a job id to watch one job) |
| `python3 status.py --fetch 6518997` | copy that job's results to `runs/` (if you stopped watching early) |
| `python3 status.py --cancel 6518997` | stop a job |

A job state of `PENDING` with reason `Resources` or `Priority` just means it is waiting for a free node;
GPU jobs can wait a while. Pressing Ctrl-C while `runIt.py` waits stops the *watching*, not the job; use
`status.py` to check on it later.

## The demo: `08_demo.py`

A ball drops onto the floor and bounces: under 50 lines, with comments. The cluster has no screen, so
instead of opening a window the script renders each frame offscreen and saves an MP4. The structure is the
pattern for any MuJoCo script you run here:

1. `XML`: describe the world (floor, ball, camera, light).
2. `MjModel` / `MjData`: build the model and its state.
3. Loop: `mj_step` advances the physics; `renderer.render()` takes a picture.
4. Write the frames to a video file.

To make your own, copy `08_demo.py`, change the `XML` (try a bigger ball, a different `solref` for more or
less bounce, a second body) and run it with `runIt.py`. Whatever your script writes (videos, PNG plots, data
files) comes back to `runs/<name>/`. The CPU and GPU runs give the same physics; the GPU only speeds up
rendering (and GPU-based training such as MJX/JAX later). For an interactive window (`mujoco.viewer`) run
MuJoCo on your laptop instead; the cluster is for long or heavy runs.

## If something goes wrong

Every run writes a log of every command and its output to `logs/`. **Send the newest file in `logs/`**
(the last line of each command's output tells you its name).

| What you see | What it means / what to do |
|---|---|
| `STOP: this script is for your laptop` | You are in a terminal logged in to the cluster (prompt says `login-...`). Use the terminal on your own computer. |
| `can reach ...` FAIL, or login fails | Off campus without the Tufts VPN, or wrong `HPC_USER` / `HPC_HOST` in `config.sh`. |
| Asked for a password again | The shared login lasts 2 hours. Just type it again. |
| `no such file` when running a script | The terminal is not in the `mujoco-hpc` folder. `cd` there. |
| `the MuJoCo environment is missing` | Run `python3 setup.py` first. |
| Job stays `PENDING` | Waiting for a free node (normal for GPUs). `python3 status.py` shows the reason. |
| Job `FAILED` | Read the job output printed by `runIt.py` (or `python3 status.py <JOBID>`); the traceback is usually in the last lines. |
| `TIMEOUT` | The job hit its time limit. Raise `--time`. |
| `PartitionNotFound` / `InvalidQOS` | Wrong queue name. Use `--partition` with a name from `sinfo -s` on the cluster. |
| Quota warnings on the cluster | Home folders are limited (about 28 GB). Delete old folders in `~/mujoco-hpc/runs/` on the cluster. |
| `libEGL warning` lines | Harmless noise from rendering without a display, as long as the next line says `PASS render`. |

## Windows

Windows has no `rsync`, and its built-in `ssh` cannot share one login between commands (you would be asked
for Duo on every step). So use WSL (Ubuntu inside Windows), about 10 minutes once:

1. PowerShell as administrator: `wsl --install`, restart, finish the Ubuntu setup.
2. In the Ubuntu app: `sudo apt update && sudo apt install -y rsync openssh-client python3`
3. Keep this folder inside Ubuntu (for example `~/mujoco-hpc`, not under `/mnt/c`) and run the same commands
   with `python3`. In VSCode, install the "WSL" extension and choose "Connect to WSL".

Running the scripts in plain Windows stops with these instructions.

## What is in this folder

| File | Purpose |
|---|---|
| `config.sh` | Your username and the cluster address. The only thing to edit. |
| `setup.py`, `runIt.py`, `status.py`, `hpc.py` | The commands above, and the shared code they use (ssh, rsync, Slurm, logging). |
| `04_test_mujoco.py` | The smoke test `setup.py` runs on the cluster (physics, plus a rendering check). |
| `05_job.sbatch`, `06_gpu_job.sbatch` | Slurm job scripts for the CPU and GPU tests. |
| `08_demo.py`, `08_demo.sbatch` | The bouncing-ball demo and its job script (for the by-hand route). |
| `02_check_cluster.sh`, `03_setup_env.sh` | Cluster-side survey and installer, run by `setup.py`. |
| `00_...`, `01_...`, `01b_login.sh`, `08b_fetch_results.sh`, `activate.sh` | The by-hand route, below. |
| `logs/`, `runs/`, `results/` | Created as you work: logs, and results copied back from the cluster. |

What the setup found on the cluster (Tufts "Pax"): login node `login.pax.tufts.edu`; Slurm queues `batch`
(CPU) and `gpu` (NVIDIA A100 nodes); Python 3.13 via `module load python/3.13.2` (the system Python 3.9 is too old
for MuJoCo, so the installer loads the newer one automatically); no preinstalled MuJoCo, so each user
gets their own in `~/mujoco-env`.

## Doing it by hand (optional)

The Python commands do all of this for you. The by-hand steps are for understanding what happens, or debugging.
Scripts marked *laptop* run in a terminal on your computer; *cluster* in a terminal logged in to the cluster.

| Step | Where | Command | Proves |
|---|---|---|---|
| 0 | laptop | edit `config.sh` | username/host are right |
| 1 | laptop | `./00_check_connection.sh` | VPN, DNS, SSH, Duo all work |
| 2 | laptop | `./01_push_and_check.sh` | files copy over; runs the survey (step 3) too |
| 2.5 | laptop, 2nd terminal | `./01b_login.sh` | you are logged in, in `~/mujoco-hpc` (VSCode: **+** in the terminal panel) |
| 3 | cluster | `bash 02_check_cluster.sh` | Slurm, queues, Python, modules, disk, pip reachable |
| 4 | cluster | `bash 03_setup_env.sh` | MuJoCo, numpy, imageio, imageio-ffmpeg install into `~/mujoco-env` |
| 5 | cluster | `source ~/mujoco-env/bin/activate`, then `python 04_test_mujoco.py` | MuJoCo imports and simulates |
| 6 | cluster | `sbatch 05_job.sbatch` | same test on a compute node via Slurm |
| 7 | cluster | `sbatch 06_gpu_job.sbatch` | a GPU node is allocated and EGL rendering works |
| 8 | cluster, then **laptop** | `sbatch 08_demo.sbatch`, then `./08b_fetch_results.sh` | the demo renders; the video comes home and opens |

Every new cluster terminal needs `source ~/mujoco-env/bin/activate` first (if `python --version` says 3.9, you
skipped it). Shell scripts (`./...`) must be run from inside `mujoco-hpc`.

Checking a job by hand, on the cluster:
```
squeue -u $USER                  # listed = waiting (PD) or running (R); gone = finished
sacct -j <jobid> -X --format=JobID,State,Elapsed,ExitCode   # COMPLETED = ok; FAILED / TIMEOUT: read the .out file
tail -f mujoco-gpu-<jobid>.out   # watch output live (Ctrl-C stops watching, not the job)
cat mujoco-gpu-<jobid>.out       # the result, once done
```
`sbatch` prints `Submitted batch job <jobid>`; that number is in the output file name. Don't run long
simulations on the login node; submit them with `sbatch` (or `runIt.py`).
