"""Load and hash the immutable reference contract and operational policy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import content_hash


CONFIG_DIR = Path(__file__).with_name("config")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    value = load_json(path or CONFIG_DIR / "reference_contract.json")
    if value.get("reported_experiment_activation") is not False:
        raise ValueError("monitoring must remain outside the reported experiments")
    return value, content_hash(value)


def load_policy(path: Path | None = None) -> tuple[dict[str, Any], str]:
    value = load_json(path or CONFIG_DIR / "policy_v1.json")
    if value.get("classification") != "operational_policy":
        raise ValueError("monitoring thresholds must be labeled operational_policy")
    return value, content_hash(value)