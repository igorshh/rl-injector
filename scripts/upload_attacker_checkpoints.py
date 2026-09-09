#!/usr/bin/env python3
"""Upload RL-Hammer attacker LoRA checkpoints to the Hugging Face ``imperial-cpg`` org.

For every run directory under ``--checkpoints`` the highest-numbered ``checkpoint-N`` is
uploaded to ``imperial-cpg/<run_dir_name>`` as a private repo, keeping only what is needed
to load the adapter (PEFT config + weights, tokenizer, chat template, model card,
trainer_state for provenance). Optimizer/scheduler/RNG state is not uploaded.

    python scripts/upload_attacker_checkpoints.py --checkpoints /path/to/checkpoints [--dry-run]
    python scripts/upload_attacker_checkpoints.py --checkpoints ... --only rl_hammer_v2base_lora

Auth: the token at ~/.cache/huggingface/token (``huggingface-cli login``) must belong to a
member of imperial-cpg with write access.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

ORG = "imperial-cpg"
KEEP = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "trainer_state.json",
    "training_args.bin",
]


def last_checkpoint(run_dir: Path) -> Path | None:
    cks = [(int(m.group(1)), p) for p in run_dir.glob("checkpoint-*")
           if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir()]
    return max(cks)[1] if cks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run dir names to upload (default: all)")
    ap.add_argument("--skip", nargs="*", default=[], help="run dir names to skip")
    ap.add_argument("--public", action="store_true", help="create public repos (default private)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    who = api.whoami()
    assert any(o.get("name") == ORG for o in who.get("orgs", [])), f"{who.get('name')} is not in {ORG}"
    runs = sorted(p for p in Path(a.checkpoints).iterdir() if p.is_dir() and not p.name.startswith("_"))
    for run in runs:
        if a.only and run.name not in a.only:
            continue
        if run.name in a.skip:
            continue
        ck = last_checkpoint(run)
        if ck is None:
            print(f"[skip] {run.name}: no checkpoint-* dir")
            continue
        files = [ck / f for f in KEEP if (ck / f).exists()]
        size_gb = sum(f.stat().st_size for f in files) / 2**30
        repo_id = f"{ORG}/{run.name}"
        print(f"{repo_id}  <-  {ck.relative_to(run.parent)}  ({len(files)} files, {size_gb:.2f} GB)", flush=True)
        if a.dry_run:
            continue
        api.create_repo(repo_id, private=not a.public, exist_ok=True, repo_type="model")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(ck),
            allow_patterns=KEEP,
            commit_message=f"Upload {run.name} {ck.name} (adapter + tokenizer, no optimizer state)",
        )
        print(f"[done] https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
