"""Read-only adapters from current pipeline artifacts to monitoring metrics."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import fmean

from .schemas import MetricObservation, SourceArtifact, file_sha256


ASPECT_NAMES = {
    "fundamental": "fundamental",
    "macro": "macro",
    "moment": "momentum",
    "tech_trend": "trend",
    "trade": "trade",
}
REQUIRED_STAGE2_COLUMNS = {"Date", "y_true_20", "prob_down", "prob_up", "source"}


@dataclass(frozen=True)
class Stage2Result:
    source: SourceArtifact
    metrics: tuple[MetricObservation, ...]
    observation_start: str
    observation_end: str
    mature_label_cutoff: str
    status: str


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left, mean_right = fmean(left), fmean(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    variance_left = sum((value - mean_left) ** 2 for value in left)
    variance_right = sum((value - mean_right) ** 2 for value in right)
    denominator = math.sqrt(variance_left * variance_right)
    return covariance / denominator if denominator else None


def _balanced_accuracy(labels: list[int], predictions: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    true_positives = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    true_negatives = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    return 0.5 * (true_positives / positives + true_negatives / negatives)


def adapt_stage2_prediction(path: Path, stock_id: str, aspect: str, as_of_date: date,
                            label_horizon: int = 20, mature_window: int = 60) -> Stage2Result:
    normalized_aspect = ASPECT_NAMES.get(aspect)
    if normalized_aspect is None:
        raise ValueError(f"unknown Stage 2 aspect: {aspect}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_STAGE2_COLUMNS <= set(reader.fieldnames or ()):
            missing = sorted(REQUIRED_STAGE2_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    parsed: list[tuple[date, int, float, float, str]] = []
    seen_dates: set[date] = set()
    for row in rows:
        row_date = date.fromisoformat(row["Date"])
        if row_date > as_of_date:
            continue
        if row_date in seen_dates:
            raise ValueError(f"{path}: duplicate date {row_date}")
        seen_dates.add(row_date)
        label = int(row["y_true_20"])
        prob_down, prob_up = float(row["prob_down"]), float(row["prob_up"])
        if label not in (0, 1) or not (0.0 <= prob_up <= 1.0 and 0.0 <= prob_down <= 1.0):
            raise ValueError(f"{path}: invalid label/probability on {row_date}")
        if not math.isclose(prob_up + prob_down, 1.0, abs_tol=1e-5):
            raise ValueError(f"{path}: probabilities do not sum to one on {row_date}")
        parsed.append((row_date, label, prob_down, prob_up, row["source"].strip()))
    parsed.sort(key=lambda item: item[0])
    if len(parsed) <= label_horizon:
        mature: list[tuple[date, int, float, float, str]] = []
        cutoff = ""
    else:
        eligible = parsed[:-label_horizon]
        mature = eligible[-mature_window:]
        cutoff = eligible[-1][0].isoformat()
    sample_count = len(mature)
    status = "valid" if sample_count >= mature_window else "insufficient_data"
    observation_start = mature[0][0].isoformat() if mature else ""
    observation_end = mature[-1][0].isoformat() if mature else ""
    labels = [item[1] for item in mature]
    probabilities = [item[3] for item in mature]
    predictions = [int(value >= 0.5) for value in probabilities]
    predicted_positive = sum(predictions)
    precision = (sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
                 / predicted_positive if predicted_positive else None)
    values = {
        "precision": precision,
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "brier_score": fmean((probability - label) ** 2
                              for probability, label in zip(probabilities, labels)) if mature else None,
        "probability_label_correlation": _correlation(probabilities, labels),
    }
    metrics = tuple(MetricObservation(name, value, sample_count, stock_id, 2,
                                      normalized_aspect,
                                      status if value is not None else "unavailable")
                    for name, value in values.items())
    source_modes = sorted({item[4] for item in mature})
    evidence_role = "stage2_prediction:" + (",".join(source_modes) if source_modes else "none")
    source = SourceArtifact(2, path.as_posix(), file_sha256(path), stock_id,
                            normalized_aspect, evidence_role)
    return Stage2Result(source, metrics, observation_start, observation_end, cutoff, status)