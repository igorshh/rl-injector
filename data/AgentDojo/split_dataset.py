# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os
import itertools

from agentdojo.scripts.benchmark import get_suites, get_suite

# v1.2.2 to match the gpt-oss sweep this attacks; v1.2.1 has a different
# task set, so ASRs would not be comparable.
benchmark_version = "v1.2.2"

all_suites = tuple(get_suites(benchmark_version).keys())

data = []
# Excluded from the gpt-oss sweep, and excluded here for the same reasons plus one more.
#   workspace/user_task_35 -- never fits the 12288-token cap at any GPU count
#   workspace/user_task_38 -- sits AT the cap, so whether it fits depends on how verbose the
#                             arm is, making it a measure of verbosity rather than defense
# The extra reason: a task that blows the cap is recorded by AgentDojo as
# `utility=False, security=True`, i.e. an attack success. Under RL that is free reward for
# any injection that pads the context, so leaving them in hands the attacker a reward hack
# that has nothing to do with defeating the defense.
EXCLUDED_USER_TASKS = {"user_task_35", "user_task_38"}

for suite_name in all_suites:
    suite = get_suite(benchmark_version, suite_name)
    all_tasks = [t for t in suite.user_tasks.keys()
                 if not (suite_name == "workspace" and t in EXCLUDED_USER_TASKS)]
    all_injections = list(suite.injection_tasks)

    for task_name, injection_name in list(itertools.product(all_tasks, all_injections)):
        data.append(
            {
                "suite_name": suite_name,
                "task_name": task_name,
                "injection_name": injection_name,
            }
        )

# Seeded: the split has to be identical across arms, or an attacker trained against
# v2base and one trained against v2gates0025 are fitted and scored on different tasks.
random.seed(1024)
random.shuffle(data)

# 100 for eval; 100 for test; rest for training. NOTE the eval set is only 100 pairs
# against the sweep's 921, so an ASR from it is far noisier and is NOT comparable to the
# 24.00% / 1.19% headline numbers. Always pair it with a baseline measured on this same
# eval.json (e.g. --attacker_model_name_or_path important_instructions).
train_data = data[:-200]
eval_data = data[-200:-100]
test_data = data[-100:]

os.makedirs("data/AgentDojo/dataset", exist_ok=True)
with open("data/AgentDojo/dataset/train.json", "w") as file:
    json.dump(train_data, file, indent=4)
with open("data/AgentDojo/dataset/eval.json", "w") as file:
    json.dump(eval_data, file, indent=4)
with open("data/AgentDojo/dataset/test.json", "w") as file:
    json.dump(test_data, file, indent=4)
