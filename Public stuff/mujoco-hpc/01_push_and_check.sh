#!/usr/bin/env bash
# Step 2 (run on your LAPTOP): copy this folder to the cluster and run the
# cluster-side survey (02_check_cluster.sh) on the login node.
source "$(dirname "$0")/config.sh"
cd "$(dirname "$0")"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/%r@%h -o ControlPersist=10m"

mkdir -p ~/.ssh/sockets
rsync -av -e "ssh $SSH_OPTS" --exclude '.DS_Store' --exclude '__pycache__' \
    ./ "$HPC_USER@$HPC_HOST:$HPC_DIR/"
ssh $SSH_OPTS "$HPC_USER@$HPC_HOST" "cd $HPC_DIR && bash 02_check_cluster.sh"
