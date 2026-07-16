"""Config loading.

Two sources, split by what changes and what doesn't:

    .env         where the models are. Machine-specific, not in git.
    config.toml  who she is and what she says. The same everywhere.

Paths resolve relative to the project root, never the cwd, so ./yap works from
anywhere.
"""

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"
ENV_PATH = ROOT / ".env"

load_dotenv(ENV_PATH)


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def _env_path(name, must_exist=True):
    """Read a path from .env, failing loudly and early if it's wrong.

    Worth being strict here: a bad model path is not an error to transformers --
    it treats it as a hub repo id and dies much later with 'Repo id must be in
    the form namespace/repo_name', which says nothing about the real problem.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise SystemExit(
            f"{name} is not set.\n"
            f"    cp .env.example .env    # then point it at your models"
        )
    path = Path(raw).expanduser()
    if must_exist and not path.exists():
        raise SystemExit(f"{name} in {ENV_PATH} points at\n    {path}\nwhich does not exist.")
    return path


def _resolve_snapshot(path):
    """Turn an HF cache repo root into its snapshot dir.

    Given a `models--org--name` root, return the snapshot inside it. A path that
    is already a model dir is returned untouched. Resolving at load time rather
    than hardcoding a hash means a re-download doesn't silently break things.
    """
    if (path / "config.json").exists():
        return str(path)

    snapshots = sorted((path / "snapshots").glob("*/")) if (path / "snapshots").is_dir() else []
    usable = [s for s in snapshots if (s / "config.json").exists()]
    if not usable:
        raise SystemExit(
            f"No usable model under {path}.\n"
            f"Expected {path}/config.json or {path}/snapshots/<hash>/config.json"
        )
    return str(usable[-1])


def load(path=None):
    """Load config.toml + .env as nested attributes: cfg.voice.reference etc."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "rb") as f:
        cfg = _ns(tomllib.load(f))

    # Triple-quoted TOML blocks keep their newlines and indentation.
    cfg.writer.system_prompt = cfg.writer.system_prompt.strip()
    cfg.writer.opening = cfg.writer.opening.strip()
    cfg.voice.description = " ".join(cfg.voice.description.split())
    cfg.voice.audition_text = " ".join(cfg.voice.audition_text.split())

    # From .env, not config.toml: these differ per machine.
    cfg.llm.model_path = str(_env_path("LLM_MODEL"))
    cfg.voice.model_dir = _resolve_snapshot(_env_path("TTS_MODEL_DIR"))
    cfg.voice.design_model_dir = _resolve_snapshot(_env_path("TTS_DESIGN_MODEL_DIR"))
    # Not must_exist: ./yap gives a better error, pointing at the build script.
    cfg.llm.server_binary = _env_path("LLAMA_SERVER", must_exist=False)

    cfg.voice.reference = str(ROOT / cfg.voice.reference)
    cfg.root = ROOT
    return cfg
