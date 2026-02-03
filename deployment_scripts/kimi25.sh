#!/bin/bash
set -euxo pipefail

NNODES=${WORLD_SIZE:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODES:-8}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_PATH="/newcpfs/user/zhiyuan/models/moonshotai/Kimi-K2.5"
SERVED_MODEL_NAME="kimi-k2.5"

SERVICE_PORT=8000
DIST_PORT=5000
DIST_INIT_ADDR="${MASTER_ADDR}:${DIST_PORT}"

TP_SIZE=$((NNODES * NGPUS_PER_NODE))

echo "========================================"
echo "SGLang Multi-Node Deployment"
echo "Node Rank: ${NODE_RANK}/${NNODES}"
echo "GPUs per Node: ${NGPUS_PER_NODE}"
echo "Total TP Size: ${TP_SIZE}"
echo "Dist Init Addr: ${DIST_INIT_ADDR}"
echo "========================================"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

if [ ${NODE_RANK} -eq 0 ]; then
    echo "[Head Node] Starting coordinator..."

    sleep 5
else
    echo "[Worker Node] Waiting for head node ${MASTER_ADDR}:${DIST_PORT}..."

    for i in {1..24}; do
        if nc -z ${MASTER_ADDR} ${DIST_PORT} 2>/dev/null; then
            echo "[Worker Node] Head node ready, starting..."
            break
        fi
        echo "[Worker Node] Waiting... ($i/24)"
        sleep 5
    done
fi

exec sglang serve \
    --model-path ${MODEL_PATH} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --tp ${TP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank ${NODE_RANK} \
    --dist-init-addr ${DIST_INIT_ADDR} \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --host 0.0.0.0 \
    --port ${SERVICE_PORT} \
    --enable-metrics \
    --watchdog-timeout 300
