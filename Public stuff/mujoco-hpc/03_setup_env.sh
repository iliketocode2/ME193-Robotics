#!/usr/bin/env bash
# Step 4 (runs ON the cluster login node): make a Python venv with MuJoCo.
# MuJoCo needs a recent Python (3.10+); the cluster's default is 3.9, too old.
# By default this loads the newest python module it can find. To choose one
# yourself, put its name in PY_MODULE (names come from 02_check_cluster.sh).
PY_MODULE=""
VENV="$HOME/mujoco-env"

set -e
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true

if [ -z "$PY_MODULE" ]; then
  # 'module avail' prints to stderr; keep names like python/3.12.1, take the highest >= 3.10
  PY_MODULE=$(module -t avail python 2>&1 | grep -E '^python/3\.[0-9]+' \
    | grep -viE 'rc|alpha|beta' \
    | awk '{ if (match($0, /3\.[0-9]+(\.[0-9]+)?/)) { v=substr($0,RSTART,RLENGTH); split(v,a,"."); if (a[2]>=10) print a[2]*1000+a[3], $0 } }' \
    | sort -n | tail -1 | cut -d' ' -f2 | tr -d ':')
fi
if [ -n "$PY_MODULE" ]; then echo "loading module: $PY_MODULE"; module load "$PY_MODULE"
else echo "No python 3.10+ module found; using system python3 (may be too old)."; fi

python3 --version
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || {
  echo "FAIL: python is older than 3.10. Run 'module avail python', set PY_MODULE in this file, rerun."; exit 1; }

rm -rf "$VENV"            # start clean in case an earlier attempt left a broken venv
python3 -m venv "$VENV"
echo "$PY_MODULE" > "$VENV/python_module"   # so 05_job.sbatch loads the same python
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# --only-binary: never compile MuJoCo from source (needs MUJOCO_PATH and a C++ toolchain).
python -m pip install --only-binary :all: mujoco numpy imageio imageio-ffmpeg
python -c "import mujoco; print('mujoco', mujoco.__version__, 'installed in', '$VENV')"
echo; echo "Next: python 04_test_mujoco.py   (quick, on the login node)"
echo "Then: sbatch 05_job.sbatch        (real test, on a compute node)"
