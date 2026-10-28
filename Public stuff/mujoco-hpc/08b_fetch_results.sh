#!/usr/bin/env bash
# Step 8b (run on your LAPTOP, in the mujoco-hpc folder): copy videos/plots back from the cluster
# into ./results, then open the newest one.
source "$(dirname "$0")/config.sh"
cd "$(dirname "$0")"
mkdir -p results ~/.ssh/sockets
rsync -av -e "ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/%r@%h -o ControlPersist=10m" \
    "$HPC_USER@$HPC_HOST:$HPC_DIR/results/" results/
latest=$(ls -t results/*.mp4 2>/dev/null | head -1)
if [ -n "$latest" ]; then echo "opening $latest"; open "$latest" 2>/dev/null || xdg-open "$latest"; else echo "no .mp4 yet; did the job finish? (squeue -u \$USER on the cluster)"; fi
