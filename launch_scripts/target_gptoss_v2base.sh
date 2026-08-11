#!/bin/bash
#
# RL-Hammer against gpt-oss-20b (v2base, undefended) with Llama-3.1-8B as the easy target.
#
# Modelled on target_meta_secalign_8b.sh, with three changes forced by our target:
#
#   1. gpt-oss is served by `interppi.serve`, not vLLM. For v2base vLLM would work, but the
#      whole point of starting here is a baseline comparable to the static InjecAgent runs
#      (0.19-0.79% ASR), and those went through this server. The gated arms cannot use vLLM
#      at all -- channel-aware attention is the defense -- so this is also the path we have
#      to make work eventually.
#   2. Five replicas of the target behind a proxy, where they give the target one GPU. Our
#      target is 30x slower per GPU than theirs (115 tok/s vs vLLM's 3186 on an 8B), and it
#      is 93% of a step, so the GPUs go there. gpt-oss-20b also takes ~47.5 GiB, so it
#      cannot share a card the way their two 8B targets do.
#   3. --max-batch-size on our server. The reward fans out every rollout at once
#      (ThreadPoolExecutor(max_workers=len(user_inputs))), so unbatched they all queue on
#      _GENERATE_LOCK. 16 is where per-GPU throughput plateaus.
#
# GPU map (8 total): 0-1 attacker GRPO | 2-6 five gpt-oss replicas | 7 llama easy target.
#
# The target model NAME MATTERS. reward_func routes on it: a name containing "/" goes to
# run_target_model, which inlines the whole scratchpad into one *user* message -- trusted by
# our channel tokenizer, so a gated arm would be measured with its defense inert. A name
# without "/" goes to run_gpt_target_model, native tool calling, which is the path our
# InjecAgent eval used. Hence "v2base", never a path.
set -e

export WANDB_PROJECT="rl-hammer-gptoss"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# openai_harmony caches a vocab file here; the default /tmp path belongs to whoever created
# it first on a shared node, and every later user gets a 500 on the first tools= request.
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-${HOME}/.cache/tiktoken-rs}
mkdir -p "${TIKTOKEN_RS_CACHE_DIR}"

LR=1e-5
RUN_NAME=rl_hammer_gptoss_v2base_lora
INTERPPI=${INTERPPI:-${HOME}/prompt_injection_interp}
ATTACKER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct

# --- target 1: gpt-oss-20b v2base, 5 replicas behind one proxy -----------------------
# Five replicas, not one server on five GPUs: --device auto is pipeline parallel (layers
# split, one GPU active at a time), which buys context length, not throughput. Measured, one
# GPU plateaus at ~115 tok/s and batch size beyond 16 adds nothing (79.2s @8, 67.0s @16,
# 68.8s @32, 65.9s @64 for 64 rollouts), so replicas are the only lever left.
#
# The reward keeps one client per target MODEL and has no notion of replicas, so the proxy
# presents the five as a single endpoint. interppi.serve.proxy dispatches to the fewest
# in-flight, which matters because each replica batches internally.
GPTOSS_PORTS=""
for i in 1 2 3 4 5; do
    PORT=$((8010 + i))
    CUDA_VISIBLE_DEVICES=$((i + 1)) uv run --directory "${INTERPPI}" python -m interppi.serve \
        --model openai/gpt-oss-20b --model-name v2base --port ${PORT} \
        --max-model-len 4096 --max-tokens 512 --reasoning-effort low \
        --max-batch-size 16 --batch-wait-ms 40 > logs/serve_v2base_${i}.log 2>&1 &
    GPTOSS_PORTS="${GPTOSS_PORTS} --backend http://localhost:${PORT}"
done

# Proxy on 8010, the single URL the reward sees. It waits for all five replicas before
# listening, so the first step cannot hit a still-loading server.
uv run --directory "${INTERPPI}" python -m interppi.serve.proxy \
    --port 8010 ${GPTOSS_PORTS} > logs/proxy_v2base.log 2>&1 &

# --- target 2: Llama-3.1-8B-Instruct, vLLM, GPU 7 ------------------------------------
# One replica is plenty: measured 3186 tok/s against gpt-oss's 115, so the easy target is
# 4.9% of the step. Two 8B models would fit on this card if that ever changes.
# Kept on vLLM: it is only here to supply reward signal on the easy target, and vLLM's
# continuous batching serves it far faster than our server would.
CUDA_VISIBLE_DEVICES=7 ${VLLM_PY:-python} -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8011 \
    --gpu_memory_utilization 0.90 > logs/serve_llama.log 2>&1 &

until curl -s http://localhost:8010/v1/models > /dev/null; do sleep 5; done
until curl -s http://localhost:8011/v1/models > /dev/null; do sleep 5; done
echo "both targets up"

# --- attacker: GRPO on GPUs 0-1 -------------------------------------------------------
# 2/5/1 comes from measuring the three phases at 80 rollouts: attacker generation 2.0s,
# gpt-oss target 84.6s (93% of the step), llama easy target 4.5s. The target dominates, so
# it gets the GPUs.
#
# 20 per_device x 2 processes x 2 accum = 80 = 10 goals x 8 rollouts. The paper's batch is 8
# goals; 80 is chosen instead because it divides evenly over 5 replicas at 16 each, which is
# exactly where per-GPU throughput plateaus. --num_generations 8 keeps the paper's
# rollouts-per-goal, which is the part that sets GRPO's advantage variance.
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
    train.py \
    --attacker_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
    --target_model_name_or_path "v2base;meta-llama/Llama-3.1-8B-Instruct" \
    --target_model_url "http://localhost:8010/v1;http://localhost:8011/v1" \
    --reward_functions InjecAgentToolCallingReward \
    --dataset data/InjecAgent/dataset/train.json \
    --attn_implementation flash_attention_2 \
    --num_generations 8 \
    --num_iterations 1 \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 40 \
    --bf16 True \
    --beta 0.0 \
    --warmup_ratio 0.03 \
    --gradient_checkpointing True \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --use_peft True \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir checkpoints/${RUN_NAME} \
    --report_to wandb \
    --run_name ${RUN_NAME}
