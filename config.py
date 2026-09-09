# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig


@dataclass
class LocalGRPOConfig(GRPOConfig):
    dataset: str = field(default=None)
    epsilon: float = field(default=0.5)

    attacker_model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct"
    )
    model_dtype: Optional[str] = field(default="bfloat16")
    target_model_name_or_path: Optional[str] = field(default="")
    max_completion_length: int = field(default=1024)
    save_total_limit: int = field(default=10)
    seed: int = field(default=1024)
    max_grad_norm: float = field(default=0.2)
    loss_type: str = field(default="bnpo")

    reward_functions: list[str] = field(
        default_factory=lambda: ["InjecAgentToolCallingReward"]
    )
    target_model_url: str = field(default="http://localhost:8000/v1")
    soft_rewards: bool = field(default=True)
    target_model_max_completion_length: int = field(default=512)
    target_model_temperature: float = field(default=None)
    # None = do NOT send the field; the target runs at its own default. Set it explicitly for
    # any run whose effort matters, with a value the TARGET supports: gpt-oss takes low|medium|
    # high, gpt-5.6-luna takes none|low|medium|high|xhigh|max. The old default "minimal" was
    # silently accepted by luna, which does not support it and treated it as low (measured by
    # reasoning tokens, sep09), so a run labelled minimal was in fact a low-effort run.
    reasoning_effort: Optional[str] = field(default=None)
    model_wise_reward_weights: Optional[list[float]] = field(default=None)


@dataclass
class EvalConfig:
    validation_data_path: str = None
    val_batch_size: int = 16
    max_new_tokens: int = 1024
    val_max_new_tokens: int = 512

    attacker_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    attacker_base_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    target_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    # EvalConfig lacked this while LocalGRPOConfig has it, so a target that is an
    # OpenAI-compatible endpoint could be trained against but not evaluated against.
    target_model_url: str = field(default="http://localhost:8000/v1")
    # None = do NOT send the field; the target runs at its own default. Set it explicitly for
    # any run whose effort matters, with a value the TARGET supports: gpt-oss takes low|medium|
    # high, gpt-5.6-luna takes none|low|medium|high|xhigh|max. The old default "minimal" was
    # silently accepted by luna, which does not support it and treated it as low (measured by
    # reasoning tokens, sep09), so a run labelled minimal was in fact a low-effort run.
    reasoning_effort: Optional[str] = field(default=None)
    attacker_model_dtype: str = "bfloat16"
    target_model_dtype: str = "bfloat16"
    temperature: float = None

    enable_wandb: bool = False
    wandb_project_name: str = "RL-Hammer"
    run_name: str = "test"
    output_dir: str = "outputs/test"
    save_name: str = "default"
