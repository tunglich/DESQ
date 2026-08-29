"""Generate revised-paper Table 9 directly from shipped feature CSV headers."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "features"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tables"
LABEL_PREFIX = "y_"

GROUPS = (
    ("Fundamental", "fundamental", 16,
     "Valuation ratios and revenue, income, margin, and EPS growth."),
    ("Price Trend (Tech-trend)", "tech_trend", 21,
     "Trend-following technicals, rolling alpha, and OHLCV."),
    ("Momentum", "moment", 13,
     "Oscillators, multi-window acceleration, VPT, and rolling beta."),
    ("Float (Chip / Trade flow)", "trade", 13,
     "Institutional holdings, costs, balances, borrowing, and net buying."),
    ("Macro", "macro", 15,
     "Rates, commodities, global indices, volatility, FX, and derivatives."),
)


def feature_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    return [name for name in header[1:] if name and not name.startswith(LABEL_PREFIX)]


def inventory(feature_dir: Path = FEATURE_DIR) -> tuple[
        list[dict[str, str | int]], list[dict[str, str]]]:
    rows: list[dict[str, str | int]] = []
    drift_rows: list[dict[str, str]] = []
    for paper_group, file_prefix, expected_count, description in GROUPS:
        paths = sorted(feature_dir.glob(f"{file_prefix}_*.csv"))
        if not paths:
            raise FileNotFoundError(f"no feature files for {file_prefix} in {feature_dir}")
        columns_by_path = {path: feature_columns(path) for path in paths}
        common = set.intersection(*(set(columns) for columns in columns_by_path.values()))
        canonical = [name for name in columns_by_path[paths[0]] if name in common]
        if len(canonical) != expected_count:
            raise ValueError(
                f"{file_prefix}: expected {expected_count} common features, found {len(canonical)}"
            )
        canonical_set = set(canonical)
        for path, columns in columns_by_path.items():
            extras = [name for name in columns if name not in canonical_set]
            missing = [name for name in canonical if name not in columns]
            if extras or missing:
                drift_rows.append({
                    "feature_group": paper_group,
                    "source_file": path.name,
                    "extra_features": ", ".join(extras),
                    "missing_features": ", ".join(missing),
                })
        rows.append({
            "feature_group": paper_group,
            "file_prefix": file_prefix,
            "feature_count": expected_count,
            "features": ", ".join(canonical),
            "description": description,
            "source_file_count": len(paths),
            "evidence_status": "reproduced_with_schema_drift" if any(
                drift["feature_group"] == paper_group for drift in drift_rows
            ) else "reproduced",
        })
    if sum(int(row["feature_count"]) for row in rows) != 78:
        raise ValueError("feature groups do not sum to 78")
    return rows, drift_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str | int]]) -> None:
    lines = [
        "# Table 9. Taxonomy of the 78 features",
        "",
        "Generated from every shipped `features/<group>_<ticker>.csv` header.",
        "",
        "| Feature group | # | Features | Description | Files checked | Status |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_group']} | {row['feature_count']} | {row['features']} | "
            f"{row['description']} | {row['source_file_count']} | {row['evidence_status']} |"
        )
    lines.append(
        f"| **Total** | **{sum(int(row['feature_count']) for row in rows)}** |  | "
        "Five domain-informed groups. |  | **reproduced** |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex(path: Path, rows: list[dict[str, str | int]]) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Taxonomy of the 78 features.}",
        r"\label{tab:feature_taxonomy}",
        r"\begin{tabular}{lrlp{8.5cm}}",
        r"\toprule",
        "Feature group & \\# & Features & Description \\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['feature_group']))} & {row['feature_count']} & "
            f"{latex_escape(str(row['features']))} & {latex_escape(str(row['description']))} \\\\"
        )
    lines.extend([
        r"\midrule",
        "Total & 78 & -- & Five domain-informed groups. \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows, drift_rows = inventory(args.feature_dir)
    write_csv(args.output_dir / "table9_feature_taxonomy.csv", rows)
    write_csv(
        args.output_dir / "table9_schema_drift.csv",
        drift_rows,
        ["feature_group", "source_file", "extra_features", "missing_features"],
    )
    write_markdown(args.output_dir / "table9_feature_taxonomy.md", rows)
    write_latex(args.output_dir / "table9_feature_taxonomy.tex", rows)
    print(f"Table 9: {sum(int(row['feature_count']) for row in rows)} features; "
          f"checked {sum(int(row['source_file_count']) for row in rows)} files; "
          f"{len(drift_rows)} schema drifts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())