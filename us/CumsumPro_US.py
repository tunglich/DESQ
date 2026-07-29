"""US version of CumsumPro.py.

Generate probability-CUSUM series per US ticker and write
``cumSum_prob_{t_warmup}/cusum_{ticker}.csv`` for ``t_warmup`` in {6, 12, 15}.

Differences vs the Taiwan original (CumsumPro.py):
1. Ticker universe is discovered from ``experiment/ATT_*_<TICKER>`` directories.
2. OHLCV is fetched via :func:`feature._us_data.load_price_frames` (yfinance +
   on-disk cache under ``feature/_raw/``); no SQL Server / DES_model dependency.
3. Runs the detector three times with ``t_warmup ∈ [6, 12, 15]`` and writes
   each output set to its own ``cumSum_prob_{t}`` directory.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from feature._us_data import load_price_frames  # noqa: E402
from prob_cusum.prob_cusum import CusumDetector  # noqa: E402


EXPERIMENT_ROOT = _THIS_DIR / "experiment"
T_WARMUPS = [6, 12, 15]
_ATT_DIR_RE = re.compile(r"^ATT_(?:fundamental|moment|tech_trend|macro)_([A-Z0-9.\-]+)$")


def discover_tickers(experiment_root: Path = EXPERIMENT_ROOT) -> list[str]:
    """Return sorted unique tickers found under ``experiment/ATT_<model>_<TICKER>``."""
    if not experiment_root.exists():
        raise FileNotFoundError(f"experiment root not found: {experiment_root}")
    tickers: set[str] = set()
    for entry in experiment_root.iterdir():
        if not entry.is_dir():
            continue
        match = _ATT_DIR_RE.match(entry.name)
        if match:
            tickers.add(match.group(1))
    return sorted(tickers)


def compute_cusum_series(close: pd.Series, t_warmup: int) -> pd.Series:
    """Run :class:`CusumDetector` over a Close-price series."""
    detector = CusumDetector(t_warmup=t_warmup)
    values = close.to_numpy()
    probs = np.empty(values.shape[0], dtype=float)
    for i, v in enumerate(values):
        probs[i] = detector.predict_next(v)[0]
    return pd.Series(probs, index=close.index)


def main() -> None:
    tickers = discover_tickers()
    if not tickers:
        print("[ERR] no tickers discovered under experiment/")
        return
    print(f"[INFO] discovered {len(tickers)} tickers: {tickers}")

    print("[INFO] loading Close prices via feature/_us_data.load_price_frames ...")
    frames = load_price_frames(tickers)
    close_wide = frames["Close"]

    for t_warmup in T_WARMUPS:
        out_dir = _THIS_DIR / f"cumSum_prob_{t_warmup}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] t_warmup={t_warmup} -> {out_dir}")
        for ticker in tickers:
            try:
                if ticker not in close_wide.columns:
                    print(f"  [WARN] {ticker}: missing in price frame, skip")
                    continue
                series = close_wide[ticker].dropna()
                if series.empty:
                    print(f"  [WARN] {ticker}: empty Close series, skip")
                    continue
                probs_ = compute_cusum_series(series, t_warmup)
                probs_.to_csv(out_dir / f"cusum_{ticker}.csv", header=False)
                print(f"  [OK]   {ticker}: {len(probs_)} rows")
            except Exception as exc:
                print(f"  [ERR]  {ticker}: {exc}")
                continue


if __name__ == "__main__":
    main()
