"""fetch_prices.py

Convenience helper: download OHLCV for the TW-50 constituents from Yahoo
Finance into ``./prices/<stock_id>.csv``, in the format that
``tw50_des.py`` expects.

Usage
-----
    python fetch_prices.py --top50
    python fetch_prices.py --stock-ids 2330,2454
    python fetch_prices.py --top50 --start 2019-01-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    import yfinance as yf
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        'yfinance is required. Install with: pip install yfinance>=0.2'
    ) from err

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PRICES_DIR = REPO_ROOT / 'prices'
DEFAULT_START = '2019-01-01'
DEFAULT_END = '2026-03-31'
YF_SUFFIX = '.TW'   # TWSE main-board convention on Yahoo Finance


def load_top50_ids() -> list[str]:
    fp = REPO_ROOT / 'tw50_top50.csv'
    if not fp.exists():
        raise SystemExit(f'tw50_top50.csv not found at {fp}')
    return pd.read_csv(fp, dtype={'stock_id': str})['stock_id'].tolist()


def parse_stock_ids(arg_stock_ids: str | None, use_top50: bool) -> list[str]:
    if use_top50:
        return load_top50_ids()
    if not arg_stock_ids:
        raise SystemExit('Provide --stock-ids or --top50.')
    return [s.strip() for s in arg_stock_ids.split(',') if s.strip()]


def fetch_one(stock_id: str, start: str, end: str, dest_dir: Path,
              overwrite: bool) -> str:
    dst = dest_dir / f'{stock_id}.csv'
    if dst.exists() and not overwrite:
        return f'{stock_id}: skip (exists)'
    ticker = f'{stock_id}{YF_SUFFIX}'
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df is None or df.empty:
        return f'{stock_id}: EMPTY (yfinance returned nothing for {ticker})'
    # yfinance returns a MultiIndex columns when multiple tickers; single
    # ticker returns a plain frame but recent versions may still nest.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename_axis('Date').reset_index()
    needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for c in needed:
        if c not in df.columns:
            return f'{stock_id}: MISSING column {c}'
    df = df[needed]
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    dest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, lineterminator='\n')
    return f'{stock_id}: OK ({len(df)} rows -> {dst.name})'


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--top50', action='store_true')
    p.add_argument('--start', default=DEFAULT_START)
    p.add_argument('--end', default=DEFAULT_END)
    p.add_argument('--dest-dir', type=Path, default=DEFAULT_PRICES_DIR)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--sleep', type=float, default=0.4,
                   help='seconds to sleep between downloads (default: 0.4)')
    args = p.parse_args(argv)

    stock_ids = parse_stock_ids(args.stock_ids, args.top50)
    print(f'[PLAN] {len(stock_ids)} stock(s), {args.start}..{args.end}, '
          f'dest={args.dest_dir}, overwrite={args.overwrite}')

    ok = 0
    fail = 0
    for i, sid in enumerate(stock_ids):
        try:
            msg = fetch_one(sid, args.start, args.end, args.dest_dir, args.overwrite)
        except Exception as exc:  # noqa: BLE001
            msg = f'{sid}: EXCEPTION {exc}'
        print(f'  [{i+1:>3}/{len(stock_ids)}] {msg}')
        if 'OK' in msg or 'skip' in msg:
            ok += 1
        else:
            fail += 1
        if i + 1 < len(stock_ids):
            time.sleep(args.sleep)

    print(f'\n[SUMMARY] ok={ok}  fail={fail}')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
