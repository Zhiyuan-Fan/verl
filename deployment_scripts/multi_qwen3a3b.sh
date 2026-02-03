#!/bin/bash
set -euxo pipefail

NNODES=${WORLD_SIZE:-1}
NGPUS_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}

MODEL_PATH="${MODEL_PATH:-/cpfs/user/zhiyuan/models/Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
TP_SIZE=${TP_SIZE:-8}
NGINX_PORT=${NGINX_PORT:-8000}
SVC_PORT=${SVC_PORT:-8001}
IP_DIR="/cpfs/shared/node_ips_${DLC_JOB_ID:-$$}"

NODES_PER_INSTANCE=$((TP_SIZE / NGPUS_PER_NODE))
NUM_INSTANCES=$((NNODES / NODES_PER_INSTANCE))
INSTANCE_ID=$((NODE_RANK / NODES_PER_INSTANCE))
LOCAL_NODE_RANK=$((NODE_RANK % NODES_PER_INSTANCE))
DIST_PORT=$((5000 + INSTANCE_ID))

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

echo "========================================"
echo " RANK=${NODE_RANK} | Instance ${INSTANCE_ID}/${NUM_INSTANCES}"
echo " LocalNodeRank=${LOCAL_NODE_RANK} | NodesPerInst=${NODES_PER_INSTANCE}"
echo " MASTER_ADDR=${MASTER_ADDR}"
echo "========================================"

MY_IP=$(hostname -I | awk '{print $1}')
mkdir -p "${IP_DIR}"
echo "${MY_IP}" > "${IP_DIR}/RANK_${NODE_RANK}"

while [ $(ls "${IP_DIR}"/RANK_* 2>/dev/null | wc -l) -lt ${NNODES} ]; do
    echo "[RANK ${NODE_RANK}] Waiting for all nodes... $(ls ${IP_DIR}/RANK_* 2>/dev/null | wc -l)/${NNODES}"
    sleep 2
done

HEAD_RANK=$((INSTANCE_ID * NODES_PER_INSTANCE))
HEAD_IP=$(cat "${IP_DIR}/RANK_${HEAD_RANK}")
DIST_INIT_ADDR="${HEAD_IP}:${DIST_PORT}"

if [ ${NODE_RANK} -eq 0 ]; then
    NGINX_CONF="/tmp/nginx_kimi.conf"

    {
        echo "events {}"
        echo "http {"
        echo "    upstream kimi {"
        echo "        least_conn;"
        for i in $(seq 0 $((NUM_INSTANCES - 1))); do
            inst_head_rank=$((i * NODES_PER_INSTANCE))
            inst_head_ip=$(cat "${IP_DIR}/RANK_${inst_head_rank}")
            echo "        server ${inst_head_ip}:${SVC_PORT};"
        done
        echo "    }"
        cat <<INNER
    server {
        listen ${NGINX_PORT};
        client_max_body_size 10M;
        location / {
            proxy_pass            http://kimi;
            proxy_http_version    1.1;
            proxy_set_header      Host              \$host;
            proxy_set_header      X-Real-IP         \$remote_addr;
            proxy_set_header      X-Forwarded-For   \$proxy_add_x_forwarded_for;
            proxy_set_header      Connection        "";
            proxy_read_timeout    600s;
            proxy_connect_timeout 10s;
        }
    }
INNER
        echo "}"
    } > "${NGINX_CONF}"

    if ! command -v nginx &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq nginx
    fi
    nginx -s stop 2>/dev/null || true
    sleep 1

    echo "[Nginx config]:" && cat "${NGINX_CONF}"
    nginx -t -c "${NGINX_CONF}" && nginx -c "${NGINX_CONF}"

fi

if [ ${LOCAL_NODE_RANK} -eq 0 ]; then
    echo "[Instance ${INSTANCE_ID}] Head node, starting coordinator..."
    sleep 5
else
    echo "[Instance ${INSTANCE_ID}] Worker, waiting for head ${HEAD_IP}:${DIST_PORT}..."
    for i in {1..24}; do
        if nc -z ${HEAD_IP} ${DIST_PORT} 2>/dev/null; then
            echo "[Instance ${INSTANCE_ID}] Head ready, starting..."
            break
        fi
        echo "[Instance ${INSTANCE_ID}] Waiting... ($i/24)"
        sleep 5
    done
fi

SGLANG_ARGS=(
    --model-path            "${MODEL_PATH}"
    --served-model-name     "${SERVED_MODEL_NAME}"
    --tp                    ${TP_SIZE}
    --nnodes                ${NODES_PER_INSTANCE}
    --node-rank             ${LOCAL_NODE_RANK}
    --dist-init-addr        "${DIST_INIT_ADDR}"
    --trust-remote-code
    --host                  0.0.0.0
    --port                  ${SVC_PORT}
    --enable-metrics
    --watchdog-timeout      300
)

if [ ${NODE_RANK} -eq 0 ]; then
    sglang serve "${SGLANG_ARGS[@]}" &
    SGLANG_PID=$!

    echo "[RANK 0] Waiting for sglang to be ready on port ${SVC_PORT}..."
    while ! curl -sf http://localhost:${SVC_PORT}/v1/models > /dev/null 2>&1; do
        sleep 2
    done

    echo ""
    echo "=============================================="
    echo " [统一入口 → 一般用这个]"
    echo "   curl http://${MY_IP}:${NGINX_PORT}/v1/chat/completions"
    echo ""
    echo " [各 instance 直访]"
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        inst_head_rank=$((i * NODES_PER_INSTANCE))
        inst_head_ip=$(cat "${IP_DIR}/RANK_${inst_head_rank}")
        echo "   instance ${i}: http://${inst_head_ip}:${SVC_PORT}/v1/chat/completions"
    done
    echo "=============================================="
    echo ""

    wait $SGLANG_PID
else
    exec sglang serve "${SGLANG_ARGS[@]}"
fi
