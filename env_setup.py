"""Load `.env` into the environment for wandb / HF auth.

Called from the entry points (`train.py`, `agentdojo_eval.py`) so a run picks the keys up
however it was launched, rather than depending on the caller having remembered
`set -a; source .env; set +a`.

Deliberately dependency-free: this branch ships no requirements file, and python-dotenv is
not guaranteed to be in the env. A missing `.env` is not an error -- the keys may already be
exported, which is the normal case under sbatch.

Existing environment variables win. An explicitly exported WANDB_PROJECT should not be
silently overridden by a stale value in the file.

HF auth is deliberately NOT handled here. The token lives at ~/.cache/huggingface/token and
the cache paths (HF_HUB_CACHE, ...) come from the ambient environment. Do not set HF_HOME:
it relocates the token path and silently de-authenticates, turning a private repo into a
404 that reads like the repo does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path: str | Path | None = None, override: bool = False, quiet: bool = False) -> list[str]:
    """Set variables from a `.env` file. Returns the names it set (never the values)."""
    p = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not p.exists():
        if not quiet:
            print(f"[env] no {p} -- relying on the ambient environment")
        return []

    applied = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Tolerate `export FOO=bar` and quoted values.
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)

    if not quiet:
        skipped = "" if override else " (already-set vars left alone)"
        print(f"[env] {p.name}: set {applied or 'nothing'}{skipped}")
        entity = os.environ.get("WANDB_ENTITY", "<unset>")
        project = os.environ.get("WANDB_PROJECT", "<unset>")
        has_key = "yes" if os.environ.get("WANDB_API_KEY") else "NO -- wandb will run offline or prompt"
        print(f"[env] wandb entity={entity} project={project} api_key={has_key}")
    return applied
