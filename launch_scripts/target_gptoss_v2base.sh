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
#   2. The two targets get a GPU each. Theirs share GPU 7 at gpu_memory_utilization 0.45,
#      which works for two 8B models; gpt-oss-20b alone takes ~47.5 GiB of an 80 GiB card.
#   3. --max-batch-size on our server. The reward fans out every rollout at once
#      (ThreadPoolExecutor(max_workers=len(user_inputs))), so without batching all 192 land
#      on _GENERATE_LOCK and serialize. Measured: 44.5 tok/s serial -> 161.7 tok/s at 16.
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

# --- target 1: gpt-oss-20b v2base, our server, GPU 6 ---------------------------------
# max-batch-size 16 is what we measured; higher is untested and the eager prefill score
# tensor is quadratic in padded length, so raise it with --max-batch-tokens in hand.
CUDA_VISIBLE_DEVICES=6 uv run --directory "${INTERPPI}" python -m interppi.serve \
    --model openai/gpt-oss-20b --model-name v2base --port 8010 \
    --max-model-len 4096 --max-tokens 512 \
    --max-batch-size 16 --batch-wait-ms 40 > logs/serve_v2base.log 2>&1 &

# --- target 2: Llama-3.1-8B-Instruct, vLLM, GPU 7 ------------------------------------
# Kept on vLLM: it is only here to supply reward signal on the easy target, and vLLM's
# continuous batching serves it far faster than our server would.
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8011 \
    --gpu_memory_utilization 0.90 > logs/serve_llama.log 2>&1 &

until curl -s http://localhost:8010/v1/models > /dev/null; do sleep 5; done
until curl -s http://localhost:8011/v1/models > /dev/null; do sleep 5; done
echo "both targets up"

# --- attacker: GRPO on 6 GPUs ---------------------------------------------------------
# 16 per_device x 6 processes x 2 accum = 192 sequences / 32 generations = 6 goals per step.
# At 243 completion tokens per call (measured over all 1054 InjecAgent cases, of which 53%
# is reasoning) that is ~58k target tokens per step, ~6 min on one target GPU at 161 tok/s.
# If that proves too slow the first lever is --num_generations 8, the paper's InjecAgent
# setting, which cuts target load 4x; the second is a second v2base replica.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
accelerate launch \
    train.py \
    --attacker_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
    --target_model_name_or_path "v2base;meta-llama/Llama-3.1-8B-Instruct" \
    --target_model_url "http://localhost:8010/v1;http://localhost:8011/v1" \
    --reward_functions InjecAgentToolCallingReward \
    --dataset data/InjecAgent/dataset/train.json \
    --attn_implementation flash_attention_2 \
    --num_generations 32 \
    --num_iterations 1 \
    --per_device_train_batch_size 16 \
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
