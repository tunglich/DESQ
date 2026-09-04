"""Immutable schemas and canonical hashing for monitoring artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceArtifact:
    stage: int
    path: str
    sha256: str
    stock_id: str | None = None
    aspect: str | None = None
    evidence_role: str = "monitoring_input"


@dataclass(frozen=True)
class MetricObservation:
    name: str
    value: float | None
    sample_count: int
    stock_id: str
    stage: int
    aspect: str | None = None
    status: str = "valid"


@dataclass(frozen=True)
class MonitoringSnapshot:
    as_of_date: str
    observation_start: str
    observation_end: str
    mature_label_cutoff: str
    paper_contract_hash: str
    policy_hash: str
    evaluator_hash: str
    sources: tuple[SourceArtifact, ...] = field(default_factory=tuple)
    metrics: tuple[MetricObservation, ...] = field(default_factory=tuple)
    market: str = "TW"
    schema_version: str = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def snapshot_id(self) -> str:
        return content_hash(self.payload())

    def write(self, root: Path) -> Path:
        destination = root / self.snapshot_id / "snapshot.json"
        serialized = canonical_json(self.payload()) + "\n"
        if destination.exists():
            if destination.read_text(encoding="ascii") != serialized:
                raise RuntimeError(f"immutable snapshot collision: {destination}")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=False)
        destination.write_text(serialized, encoding="ascii")
        return destination


@dataclass(frozen=True)
class CandidateStep:
    name: str
    stages: tuple[int, ...]
    stocks: tuple[str, ...]
    aspects: tuple[str, ...]
    changed_parameters: tuple[str, ...]
    mature_data_only: bool = True
    validation_scope: str = "sealed_validation"


@dataclass(frozen=True)
class CandidatePlan:
    decision_level: int
    decision_hash: str
    steps: tuple[CandidateStep, ...]
    dry_run: bool = True
    executable: bool = False
    schema_version: str = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def plan_id(self) -> str:
        return content_hash(self.payload())


@dataclass(frozen=True)
class AlarmMemory:
    stock_id: str
    previous: tuple[str, ...]
    current: tuple[str, ...]


@dataclass(frozen=True)
class MetricDeltaMemory:
    stock_id: str
    precision: float
    sharpe: float
    disagreement: float
    drift: float
    drawdown: float | None = None
    turnover: float | None = None


@dataclass(frozen=True)
class ResearchMemoryRecord:
    previous_batch_id: str
    current_batch_id: str
    window_start: str
    window_end: str
    alarms: tuple[AlarmMemory, ...]
    regime_signatures: tuple[str, ...]
    candidate_plan_id: str
    candidate_parameters: tuple[str, ...]
    promotion_status: str
    promoted: bool | None
    metric_deltas: tuple[MetricDeltaMemory, ...]
    source_equation: int = 43
    schema_version: str = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def record_id(self) -> str:
        return content_hash(self.payload())