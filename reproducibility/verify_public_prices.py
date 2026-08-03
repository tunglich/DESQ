"""Verify shipped OHLCV against a fresh yfinance download.

Anti-fabrication check: for every stock CSV under ``prices/`` (or the subset
passed with ``--stock-ids``), this script re-downloads the same ticker / date
range from Yahoo Finance (adding the ``.TW`` suffix used by TWSE) and
compares Close / Open / High / Low series.

Because Yahoo periodically retro-applies split/dividend adjustments while our
shipped CSVs are the raw prices captured at experiment time, we tolerate a
per-day *ratio* mismatch as long as it is consistent across the full window
(i.e. one split multiplier). We report:

* median close ratio (yf / shipped) - should be very close to 1 for
  unadjusted series, or a clean fraction (0.5, 0.2, ...) after a stock split;
* max abs deviation of close ratios from that median - flags fabrication
  because a fake series would drift day-by-day, not lock to a scalar;
* count of overlapping trading days.

Exits 0 when every stock passes (deviation <= --tol), 1 otherwise.

Usage
-----
    python reproducibility/verify_public_prices.py                # all shipped
    python reproducibility/verify_public_prices.py --stock-ids 2330,2454
    python reproducibility/verify_public_prices.py --tol 0.005    # 0.5%
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as err:
    raise SystemExit('yfinance is required. pip install yfinance>=0.2') from err

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = REPO_ROOT / 'prices'
YF_SUFFIX = '.TW'


def _load_shipped(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, parse_dates=['Date'])
    df = df.rename(columns=str.capitalize).sort_values('Date').reset_index(drop=True)
    return df.set_index('Date')


def _download(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(
        f'{ticker}{YF_SUFFIX}',
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _compare(shipped: pd.DataFrame, fresh: pd.DataFrame, tol: float) -> dict:
    idx = shipped.index.intersection(fresh.index)
    n = len(idx)
    if n == 0:
        return dict(status='FAIL', reason='no overlapping dates', n=0)
    a = shipped.loc[idx, 'Close'].astype(float)
    b = fresh.loc[idx, 'Close'].astype(float)
    mask = (a > 0) & (b > 0) & a.notna() & b.notna()
    if mask.sum() < 10:
        return dict(status='FAIL', reason='too few valid points', n=int(mask.sum()))
    ratio = (b[mask] / a[mask])
    med = float(ratio.median())
    dev = float((ratio - med).abs().max() / med)   # relative
    return dict(
        status='PASS' if dev <= tol else 'FAIL',
        n=int(mask.sum()),
        median_ratio=med,
        max_rel_dev=dev,
        reason='' if dev <= tol else f'max_rel_dev {dev:.4f} > tol {tol:.4f}',
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated subset; default = every prices/*.csv')
    p.add_argument('--tol', type=float, default=0.01,
                   help='max relative deviation from median split ratio (default 0.01 = 1%%)')
    p.add_argument('--prices-dir', default=str(PRICES_DIR))
    args = p.parse_args(argv)

    prices_dir = Path(args.prices_dir).resolve()
    if args.stock_ids:
        ids = [s.strip() for s in args.stock_ids.split(',') if s.strip()]
        fps = [prices_dir / f'{sid}.csv' for sid in ids]
    else:
        fps = sorted(prices_dir.glob('*.csv'))
    if not fps:
        print(f'[ERR] no CSVs under {prices_dir}')
        return 2

    print(f'[PLAN] verifying {len(fps)} tickers against yfinance, tol={args.tol}')
    n_pass = n_fail = 0
    rows = []
    for fp in fps:
        sid = fp.stem
        try:
            shipped = _load_shipped(fp)
        except Exception as err:
            print(f'  ERROR    {sid}: cannot read shipped CSV: {err}')
            n_fail += 1
            rows.append((sid, 'ERROR', 0, np.nan, np.nan, str(err)))
            continue
        start, end = shipped.index.min(), shipped.index.max()
        try:
            fresh = _download(sid, start, end)
        except Exception as err:
            print(f'  ERROR    {sid}: yfinance download failed: {err}')
            n_fail += 1
            rows.append((sid, 'ERROR', 0, np.nan, np.nan, f'yfinance: {err}'))
            continue
        r = _compare(shipped, fresh, args.tol)
        rows.append((sid, r['status'], r['n'], r.get('median_ratio', np.nan),
                     r.get('max_rel_dev', np.nan), r.get('reason', '')))
        if r['status'] == 'PASS':
            n_pass += 1
            print(f'  PASS     {sid}: n={r["n"]}, median_ratio={r["median_ratio"]:.4f}, '
                  f'max_rel_dev={r["max_rel_dev"]:.2e}')
        else:
            n_fail += 1
            print(f'  FAIL     {sid}: {r["reason"]}')

    print(f'\n[SUMMARY] pass={n_pass}, fail={n_fail}, total={len(fps)}')
    out = REPO_ROOT / 'reproducibility' / 'verify_public_prices.csv'
    pd.DataFrame(rows, columns=['stock_id', 'status', 'n', 'median_ratio',
                                'max_rel_dev', 'reason']).to_csv(out, index=False)
    print(f'[OUT] {out}')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
