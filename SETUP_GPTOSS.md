# Running RL-Hammer against the gpt-oss defense arms

Branch `agentdojo`. Attacks the 13 arms measured in
`prompt_injection_interp/docs/attention_gate/agentdojo_results_data_week_aug3.md`.

## Which AgentDojo

Install **`~/agentdojo-fork`**, not the `AgentDojo/` directory vendored here. The vendored
copy is stock upstream v0.1.34; the fork is v0.1.35 plus three fixes this setup depends on:

* `vllm_parsed` honours `LOCAL_LLM_TIMEOUT` — otherwise the OpenAI SDK waits 600s on a
  wedged generation, and RL-Hammer's own 100s reward timeout scores that as *attack failed*
* `vllm_parsed` honours `--model-id` — otherwise `/v1/models` `data[0]` picks whichever
  model the server lists first
* `AGENTDOJO_MAX_TOKENS` caps generation, and malformed tool arguments no longer abort the
  whole shard

Every number in the sweep was produced with those, so results are only comparable with them.

```bash
uv pip install -e ~/agentdojo-fork      # NOT ./AgentDojo
```

## Secrets / wandb

`.env` is a copy of `prompt_injection_interp/.env` and is **gitignored** (`.gitignore:131`).
It holds `WANDB_API_KEY`, `WANDB_ENTITY=ffuuugor` and `WANDB_PROJECT=rl-hammer-gptoss`.
`train.py` and `agentdojo_eval.py` call `load_env()` (`env_setup.py`) at import, so the keys
are picked up however the run is launched -- no need to remember
`set -a; source .env; set +a`. Already-exported variables win, so you can override the
project per run without editing the file.

It is a copy, not a link: rotating the key means updating both files.

**HF auth is not in `.env`.** The token is the usual `~/.cache/huggingface/token`, and the
cache paths (`HF_HUB_CACHE` -> `/mnt/data/artifacts/hf_cache`) come from the ambient
environment. Do not set `HF_HOME` -- it moves the token path and silently de-authenticates,
so a private repo 404s as though it did not exist.

Training reporting is TRL's, so pass `--report_to wandb --run_name ...`, and
`--log_completions True --num_completions_to_print 8` to get the generated injections into
the run. `agentdojo_eval.py` logs its own table (`adv_goal`, `attacker_output`,
`attacker_adv_prompt`, `agentdojo_output`, `if_attack_success`) plus `attack_success_rate`
and `utility_success_rate` -- but under its own `--wandb_project_name`, which defaults to
`RL-Hammer`. Pass `--wandb_project_name rl-hammer-gptoss` or evals land in a different
project from the training runs.

## Serving a target

Same launch path as the sweep. Channels must be computed **server-side**, so the target has
to be driven through `/v1/chat/completions` — which the AgentDojo reward does.

```bash
# gates arm
uv run python -m interppi.serve --model openai/gpt-oss-20b \
    --gates results/attngate_gptoss20b/gates_lr0.025/gates/epoch_0.safetensors \
    --chat-template-channel --model-name v2gates0025 \
    --port 8011 --max-tokens 8192 --max-model-len 12288

# secalign arm -- the delimiter is part of the defense
uv run python -m interppi.serve --model openai/gpt-oss-20b \
    --adapter results/agentdojo_lora/peft_adapter_v2secalign_ep1 \
    --role-remap tool=input --untrusted-inline-roles \
    --untrusted-header-roles ipython tool input \
    --model-name v2secalign_ep1 --port 8012 --max-tokens 8192 --max-model-len 12288
```

**Do not use the `main` (InjecAgent) branch against a gated arm.** It renders the chat
template client-side and posts to `/v1/completions`, where the server labels every token
trusted (`token_channels` all zero). Gating is then inert and the arm reports as undefended.

## Targeting

`--target_model_url` takes `PORT[:MODEL_ID]` per target, positionally matched to
`--target_model_name_or_path`:

```bash
export LOCAL_LLM_TIMEOUT=2400 AGENTDOJO_MAX_TOKENS=8192
accelerate launch train.py \
    --attacker_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --target_model_name_or_path "vllm_parsed;vllm_parsed" \
    --target_model_url "8010:v2base;8011:v2gates0025" \
    --reward_functions AgentDojoReward \
    --dataset data/AgentDojo/dataset/train.json ...
```

One target = an adaptive attacker for that arm, which is the robustness question. Several
targets = a universal attack that must beat all of them.

## Known measurement hazards

* **A timeout scores as a successful defense.** `run_agentdojo_with_timeout` returns
  `(False, False)` after `MAX_SECONDS = 100`. Slower arms therefore look more robust for
  free, and the gated arms *are* slower than base. Raise it and count timeouts separately
  before believing any ASR difference.
* **Generation budget.** A budget too small to finish a trajectory also reads as a defense.
  On the demo prompts, 320 tokens scored a compromised base model as having resisted.
* **v1.2.2**, matching the sweep. Upstream used v1.2.1, whose task set differs.
