"""Read params.yaml, the single place for all pipeline settings (DVC tracks it per stage)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
