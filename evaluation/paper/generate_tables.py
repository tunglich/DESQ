"""Generate and audit revised-paper Tables 3-10.

Reported-only tables are deterministic transcriptions of the authoritative PDF,
not empirical reproductions. Validation reports show which rows can be rebuilt
from shipped raw artifacts and where legacy artifacts disagree with the paper.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

import generate_table9 as feature_taxonomy


PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parents[1]
SOURCES = PAPER_DIR / "sources"
TABLES = PAPER_DIR / "tables"
VALIDATION = PAPER_DIR / "validation"
RF_ANNUAL = 0.0442
ANN = 252

TABLE_SPECS = (
    (3, "walk_forward", "Walk-forward validation precision"),
    (4, "module_ablation", "Module ablation on the TWSE Top-50"),
    (5, "horizon", "Signal-horizon sensitivity"),
    (6, "cross_market", "Cross-market portfolio back-test statistics"),
    (7, "uncertainty", "Uncertainty and statistical reliability"),
    (8, "regime", "Regime-conditional performance"),
    (10, "top50_flooding", "Per-stock OOS returns and flooding ablations"),
)


def sha256(path: Path) -> str:
    canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical_bytes).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                     ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def render_markdown(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    lines = [f"# {title}", "",
             "Evidence status is row-specific; `reported_only` is a PDF transcription.", "",
             "| " + " | ".join(fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(markdown_escape(row[field]) for field in fields) + " |"
                 for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def render_latex(path: Path, number: int, title: str, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    columns = "l" * len(fields)
    lines = [r"\begin{landscape}", r"\begin{longtable}{" + columns + "}",
             f"\\caption{{{latex_escape(title)}}}\\label{{tab:paper_{number}}} " + r"\\",
             r"\toprule", " & ".join(latex_escape(field) for field in fields) + r" \\",
             r"\midrule", r"\endfirsthead", r"\toprule",
             " & ".join(latex_escape(field) for field in fields) + r" \\", r"\midrule",
             r"\endhead"]
    lines.extend(" & ".join(latex_escape(row[field]) for field in fields) + r" \\"
                 for row in rows)
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def validate_statuses(rows: list[dict[str, str]], source: Path) -> None:
    allowed = {"reported_only", "reported_only_ticker_mismatch", "mixed",
               "reproducible_from_shipped_nav"}
    unknown = sorted({row.get("evidence_status", "") for row in rows} - allowed)
    if unknown:
        raise ValueError(f"{source.name}: unknown evidence status: {unknown}")


def _stats(cumulative_pct: pd.Series) -> dict[str, float]:
    wealth = 1.0 + cumulative_pct.astype(float) / 100.0
    returns = wealth.pct_change().dropna()
    years = len(wealth) / ANN
    total = wealth.iloc[-1] / wealth.iloc[0] - 1.0
    ann_return = (1.0 + total) ** (1.0 / years) - 1.0
    ann_vol = returns.std() * math.sqrt(ANN)
    downside = returns[returns < 0].std() * math.sqrt(ANN)
    annual_mean = returns.mean() * ANN - RF_ANNUAL
    drawdown = (wealth / wealth.cummax() - 1.0).min()
    return {
        "total_return_pct": total * 100.0,
        "annual_return_pct": ann_return * 100.0,
        "annual_volatility_pct": ann_vol * 100.0,
        "sharpe_tbill": annual_mean / ann_vol,
        "sortino_tbill": (ann_return - RF_ANNUAL) / downside,
        "max_drawdown_pct": drawdown * 100.0,
        "calmar": ann_return / abs(drawdown),
    }


def validate_table6(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    files = {
        "Dow 30": (ROOT / "us/baselines/combined/dow30_comparison.csv", "^DJI"),
        "S&P 100": (ROOT / "us/baselines/combined/sp100_comparison.csv", "^OEX"),
        "NASDAQ": (ROOT / "us/baselines/combined/ndx100_comparison.csv", "^NDX"),
    }
    method_columns = {"DRL Ensemble": "DRL Ensemble", "DSR": "Dynamic Stock Recommendation",
                      "MACE": "MACE"}
    report: list[dict[str, object]] = []
    for row in rows:
        if row["evidence_status"] != "reproducible_from_shipped_nav":
            continue
        path, benchmark_column = files[row["universe"]]
        frame = pd.read_csv(path)
        method = row["method"]
        column = benchmark_column if method.endswith("(benchmark)") else method_columns[method]
        actual = _stats(frame[column])
        benchmark_total = _stats(frame[benchmark_column])["total_return_pct"]
        actual["excess_return_pct"] = actual["total_return_pct"] - benchmark_total
        for metric in ("total_return_pct", "excess_return_pct", "annual_return_pct",
                       "annual_volatility_pct", "sharpe_tbill", "sortino_tbill",
                       "max_drawdown_pct", "calmar"):
            if row[metric] == "":
                continue
            expected = float(row[metric])
            difference = actual[metric] - expected
            report.append({"universe": row["universe"], "method": method,
                           "metric": metric, "paper_value": expected,
                           "recomputed_value": round(actual[metric], 6),
                           "difference": round(difference, 6),
                           "pass_0_02": abs(difference) <= 0.02})
    core_metrics = {"total_return_pct", "excess_return_pct", "annual_return_pct",
                    "annual_volatility_pct", "max_drawdown_pct", "calmar"}
    failures = [item for item in report
                if item["metric"] in core_metrics and not item["pass_0_02"]]
    if failures:
        raise ValueError(f"Table 6 core shipped-NAV validation failed: {failures[:3]}")
    return report


def _compound_selected(returns: pd.Series, mask: pd.Series) -> float:
    return ((1.0 + returns.loc[mask]).prod() - 1.0) * 100.0


def validate_table8(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    path = ROOT / "evaluation/backtest_portfolio_tw50.csv"
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    model_wealth = 1.0 + frame["Model_CumRet"]
    benchmark_wealth = 1.0 + frame["Benchmark_CumRet"]
    model_returns = model_wealth.pct_change().fillna(model_wealth.iloc[0] - 1.0)
    benchmark_returns = benchmark_wealth.pct_change().fillna(benchmark_wealth.iloc[0] - 1.0)
    correction_returns = pd.Series(False, index=frame.index)
    calculations = {"Full window": (
        len(frame), (model_wealth.iloc[-1] - 1) * 100, (benchmark_wealth.iloc[-1] - 1) * 100)}
    for row in rows:
        if not row["regime"].startswith("Corr."):
            continue
        segment = frame.loc[row["start_date"]:row["end_date"]]
        days = len(segment) - 1
        model_return = ((1.0 + segment["Model_CumRet"].iloc[-1]) /
                        (1.0 + segment["Model_CumRet"].iloc[0]) - 1.0) * 100.0
        benchmark_return = ((1.0 + segment["Benchmark_CumRet"].iloc[-1]) /
                            (1.0 + segment["Benchmark_CumRet"].iloc[0]) - 1.0) * 100.0
        calculations[row["regime"]] = (days, model_return, benchmark_return)
        correction_returns.loc[(frame.index > segment.index[0]) &
                               (frame.index <= segment.index[-1])] = True
    calculations["Up-trend (pooled)"] = (
        int((~correction_returns).sum()),
        _compound_selected(model_returns, ~correction_returns),
        _compound_selected(benchmark_returns, ~correction_returns),
    )
    report = []
    for row in rows:
        actual = calculations[row["regime"]]
        report.append({"regime": row["regime"], "legacy_source": path.name,
                       "paper_days": int(row["days"]), "legacy_days": int(actual[0]),
                       "paper_desq_pct": float(row["desq_return_pct"]),
                       "legacy_rule_trader_pct": round(float(actual[1]), 6),
                       "paper_benchmark_pct": float(row["benchmark_return_pct"]),
                       "legacy_benchmark_pct": round(float(actual[2]), 6),
                       "evidence_status": "legacy_diagnostic_mismatch"})
    return report


def validate_table10(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    universe = set(pd.read_csv(ROOT / "tw50_top50.csv", dtype={"stock_id": str})["stock_id"])
    report = []
    for row in rows:
        if row["ticker"] == "SUMMARY":
            continue
        ticker = row["ticker"].split(".")[0]
        transaction_expected = (Decimal(row["round_trips"]) * Decimal("0.585")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        transaction_ok = abs(float(transaction_expected) - float(row["transaction_cost_pct"])) <= 0.001
        arithmetic_ok = all(
            abs((float(row[return_col]) - float(row["buy_hold_return_pct"]))
                - float(row[excess_col])) <= 0.011
            for return_col, excess_col in (
                ("dynamic_return_pct", "dynamic_excess_pct"),
                ("static_return_pct", "static_excess_pct"),
                ("no_flood_return_pct", "no_flood_excess_pct"),
            )
        )
        report.append({"ticker": row["ticker"], "in_repo_universe": ticker in universe,
                       "transaction_cost_ok": transaction_ok, "return_arithmetic_ok": arithmetic_ok})
    if not all(item["transaction_cost_ok"] and item["return_arithmetic_ok"] for item in report):
        raise ValueError("Table 10 arithmetic validation failed")
    return report


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"schema_version": 1, "tables": {}}

    for number, slug, title in TABLE_SPECS:
        source = SOURCES / f"table{number}_{slug}_reported.csv"
        rows = read_rows(source)
        validate_statuses(rows, source)
        output_csv = TABLES / f"table{number}_{slug}.csv"
        shutil.copyfile(source, output_csv)
        render_markdown(TABLES / f"table{number}_{slug}.md", f"Table {number}. {title}", rows)
        render_latex(TABLES / f"table{number}_{slug}.tex", number, title, rows)
        statuses = Counter(row["evidence_status"] for row in rows)
        manifest["tables"][str(number)] = {
            "title": title, "rows": len(rows), "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(source), "evidence_status_counts": dict(statuses),
        }
        if number == 6:
            write_rows(VALIDATION / "table6_shipped_nav_check.csv", validate_table6(rows))
        elif number == 8:
            write_rows(VALIDATION / "table8_legacy_nav_discrepancy.csv", validate_table8(rows))
        elif number == 10:
            write_rows(VALIDATION / "table10_arithmetic_universe_check.csv", validate_table10(rows))

    taxonomy_rows, drift_rows = feature_taxonomy.inventory()
    feature_taxonomy.write_csv(TABLES / "table9_feature_taxonomy.csv", taxonomy_rows)
    feature_taxonomy.write_csv(TABLES / "table9_schema_drift.csv", drift_rows,
                               ["feature_group", "source_file", "extra_features", "missing_features"])
    feature_taxonomy.write_markdown(TABLES / "table9_feature_taxonomy.md", taxonomy_rows)
    feature_taxonomy.write_latex(TABLES / "table9_feature_taxonomy.tex", taxonomy_rows)
    manifest["tables"]["9"] = {
        "title": "Taxonomy of the 78 features", "rows": len(taxonomy_rows),
        "source": "features/*.csv", "source_file_count": 250,
        "schema_drift_rows": len(drift_rows), "evidence_status_counts": {
            "reproduced": 4, "reproduced_with_schema_drift": 1},
    }

    manifest_path = PAPER_DIR / "tables_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Generated and audited revised-paper Tables 3-10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())