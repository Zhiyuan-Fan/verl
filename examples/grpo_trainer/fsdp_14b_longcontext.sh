set -x

# WandB Login
export WANDB_API_KEY="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"
wandb login --relogin $WANDB_API_KEY

NNODES=${WORLD_SIZE:-1}
NGPUS_PER_NODES=${NGPUS_PER_NODES:-8}

#====================环境变量===================
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-9899}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

# 【修改1】FSDP只需要配置rollout的TP
export INFER_TP=4

project_name='longcontext-rl'
exp_name='Qwen2.5-14B-Instruct-keychain-mixed-fsdp'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 20))
max_response_length=$((1024 * 10))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=64
# 【修改2】去掉固定micro batch，完全依赖动态调度
# train_ppo_micro_batch_size_per_gpu=2  # 删除，让dynamic_bsz接管
infer_ppo_micro_batch_size_per_gpu=2
# Paths
MODEL_PATH=/cpfs/user/zhiyuan/models/Qwen/Qwen2.5-14B-Instruct

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
TRAIN_FILE="['/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/multihop/2wikimultihop_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/keychain/2wikimultihop_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/keychain/hotpotqa_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/keychain/musique_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/multihop/hotpotqa_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/multihop/musique_thinking.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/mathdapo/dapo_math.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/bookruler/book_ruler_multi_key_mixed.parquet','/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/bookruler/book_ruler_multi_value_mixed.parquet']"

# Algorithm
temperature=0.6
top_p=0.95
top_k=-1
val_top_p=0.95

# Performance Related Parameter
use_dynamic_bsz=True
# 【修改3】参考32B脚本的token limit策略


# 创建日志目录
mkdir -p logs

if [ ${RANK} -eq 0 ]; then
    echo ${MASTER_ADDR}
    ray start --head
    echo "Waiting for all ${WORLD_SIZE} nodes to connect..."
    start_time=$(date +%s)
    
    # 等待所有节点连接
    while true; do
        total_nodes=$(ray list nodes 2>/dev/null | grep "Total:" | awk '{print $2}' || echo "0")
        echo "Connected nodes: ${total_nodes}/${WORLD_SIZE}"
        if [ "$total_nodes" -ge "$WORLD_SIZE" ]; then
            echo "All nodes connected! Starting training..."
            break
        fi
        
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -gt 1000 ]; then
            echo "Warning: Not all nodes connected within timeout. Proceeding anyway."
            break
        fi
        sleep 5
    done
    
    # 【修改4】改用ppo_trainer（FSDP版本）
    echo "=== Starting training at $(date) ==="
    python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files='/cpfs/user/zhiyuan/workspace/long-ctx-rl/repos/longctx-rl/loongrl/dataset_synthesis/training_data/multihop/2wikimultihop_thinking.parquet' \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10000 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8 \
    reward_model.reward_manager=naive \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODES}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=25 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    2>&1 | tee logs/${exp_name}_${MASTER_ADDR}_log_${RANK}_$(date +%Y%m%d_%H%M%S).txt
    
    echo "=== Training finished at $(date) ==="

else
    echo "On worker node (RANK=${RANK})"
    echo "Connecting to master at: ${MASTER_ADDR}"

    # 检查head节点是否准备就绪
    head_ready=0
    retry_count=0
    max_retries=600

    echo "Checking if Ray head node is ready..."
    while [ ${head_ready} -eq 0 ] && [ ${retry_count} -lt ${max_retries} ]; do
        if nc -z ${MASTER_ADDR} 6379 2>/dev/null; then
            echo "Ray head node is ready. Connecting..."
            head_ready=1
        else
            retry_count=$((retry_count + 1))
            echo "Ray head node not ready (attempt ${retry_count}/${max_retries}). Retrying in 5 seconds..."
            sleep 5
        fi
    done

    if [ ${head_ready} -eq 0 ]; then
        echo "ERROR: Ray head node not available after ${max_retries} attempts. Exiting."
        exit 1
    fi

    # 连接到Ray head节点
    ray start --address=${MASTER_ADDR}:6379

    # Worker节点等待训练完成
    echo "Connected to head node. Waiting for training to complete..."
    PORT_RAY=$(netstat -ntlp | grep "/ray")
    while [ -n "${PORT_RAY}" ]; do
        sleep 60s
        PORT_RAY=$(netstat -ntlp | grep "/ray")
    done
    echo "Training complete. Stopping Ray..."
    ray stop --force
fi
