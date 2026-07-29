"""Build DQN input CSVs from the DESQ pipeline output.

Original setup (tunglich/Market-Timing-DQN)
-------------------------------------------
The reference repo shipped one CSV per (symbol, window) in the schema:

    <DATE>, <DES>, <OPEN>, <HIGH>, <LOW>, <CLOSE>

where `<DES>` was a binary {0, 1} label derived from **ground-truth**
forward-return direction, and `window` was a per-symbol accuracy setting.

This variant
-------------
Here `<DES>` is derived from the **DESQ pipeline output** (the KNORA-E
aggregated probability produced by `tw50_des.py`) instead. DESQ produces
exactly one DES signal per stock, so the per-symbol accuracy `window` no
longer applies and is dropped everywhere. Everything else in the DQN
pipeline is unchanged.

    <DES> = 1  if  prob_up > threshold  else 0

Input files
-----------
    ../artifacts/des/pred/DES_<sym>.csv
        Two-column CSV: (index=Date, DES=prob_up in [0, 1]).
        Emitted by tw50_des.py at the end of Stage 3.

    ../prices/<sym>.csv    (user-supplied)
        Columns: Date, Open, High, Low, Close, Volume

Outputs
-------
    ./data/<sym>_all.csv                (one file per symbol)
        Columns: <DATE>, <DES>, <OPEN>, <HIGH>, <LOW>, <CLOSE>
    ./data/tw50_2023-12-29.csv          (copy of the top-50 list, if present)

Coverage note
-------------
`tw50_des.py` predicts on the test window only (2024-01-01 .. 2026-03-31).
The original DQN did 5-fold walk-forward on 2005-2023 pre-test history,
which is NOT available in this file. If you need DQN training history,
extend `tw50_dflood.py` to also emit in-sample predictions for the training
years, then re-run this script.

Usage
-----
    python build_dqn_data.py --stock-ids 2330
    python build_dqn_data.py --top50 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = REPO_ROOT.parent  # tw50_pipeline / DESQ root

DEFAULT_DES_PRED_DIR = PARENT_ROOT / 'artifacts' / 'des' / 'pred'
DEFAULT_PRICES_DIR = PARENT_ROOT / 'prices'
DEFAULT_DEST_DIR = REPO_ROOT / 'data'
DEFAULT_THRESHOLD = 0.5
KEEP_COLS = ('<DATE>', '<DES>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>')


def load_top50_ids() -> list[str]:
    fp = PARENT_ROOT / 'tw50_top50.csv'
    if not fp.exists():
        raise SystemExit(
            f'tw50_top50.csv not found at {fp}. Provide --stock-ids instead.'
        )
    df = pd.read_csv(fp, dtype={'stock_id': str})
    return df['stock_id'].tolist()


def parse_stock_ids(arg_stock_ids: str | None, use_top50: bool) -> list[str]:
    if use_top50:
        return load_top50_ids()
    if not arg_stock_ids:
        raise SystemExit('Provide --stock-ids or --top50.')
    return [s.strip() for s in arg_stock_ids.split(',') if s.strip()]


def load_des_probs(stock_id: str, des_dir: Path) -> pd.Series:
    fp = des_dir / f'DES_{stock_id}.csv'
    if not fp.exists():
        raise FileNotFoundError(
            f'DES prediction missing at {fp}. Run tw50_des.py first.'
        )
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    if 'DES' not in df.columns:
        # Fallback: first non-index column is treated as the probability.
        df.columns = ['DES'] + list(df.columns[1:])
    s = pd.Series(df['DES'].astype(float).to_numpy(),
                  index=pd.to_datetime(df.index), name='DES')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s


def load_prices(stock_id: str, prices_dir: Path) -> pd.DataFrame:
    fp = prices_dir / f'{stock_id}.csv'
    if not fp.exists():
        raise FileNotFoundError(
            f'Price file missing at {fp}. Add it before running.'
        )
    df = pd.read_csv(fp, parse_dates=['Date'])
    for col in ('Open', 'High', 'Low', 'Close'):
        if col not in df.columns:
            raise KeyError(f'{fp}: missing required column {col}')
    df = df[~df['Date'].duplicated(keep='last')].sort_values('Date').set_index('Date')
    return df[['Open', 'High', 'Low', 'Close']].astype(float)


def build_one_csv(stock_id: str, prices: pd.DataFrame, des_prob: pd.Series,
                   threshold: float) -> pd.DataFrame:
    """Inner-join prices with DES on Date, binarize DES, project schema."""
    joined = prices.join(des_prob, how='inner')
    if joined.empty:
        raise RuntimeError(
            f'{stock_id}: no overlapping dates between DES and prices '
            f'(DES range {des_prob.index.min().date()}..{des_prob.index.max().date()}, '
            f'prices range {prices.index.min().date()}..{prices.index.max().date()}).'
        )
    joined['<DES>'] = (joined['DES'] > threshold).astype(int)
    out = pd.DataFrame({
        '<DATE>': joined.index.strftime('%Y-%m-%d'),
        '<DES>': joined['<DES>'].to_numpy(),
        '<OPEN>': joined['Open'].to_numpy(),
        '<HIGH>': joined['High'].to_numpy(),
        '<LOW>': joined['Low'].to_numpy(),
        '<CLOSE>': joined['Close'].to_numpy(),
    })
    return out[list(KEEP_COLS)]


def write_csv(df: pd.DataFrame, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, lineterminator='\n')
    return True


def copy_constituents(dest_dir: Path, overwrite: bool) -> None:
    src = PARENT_ROOT / 'tw50_top50.csv'
    if not src.exists():
        return
    dst = dest_dir / 'tw50_2023-12-29.csv'
    if dst.exists() and not overwrite:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f'Copied constituents -> {dst.relative_to(REPO_ROOT)}')


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--top50', action='store_true')
    p.add_argument('--des-dir', type=Path, default=DEFAULT_DES_PRED_DIR)
    p.add_argument('--prices-dir', type=Path, default=DEFAULT_PRICES_DIR)
    p.add_argument('--dest-dir', type=Path, default=DEFAULT_DEST_DIR)
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                   help='DES probability threshold to binarize (default: 0.5)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)

    stock_ids = parse_stock_ids(args.stock_ids, args.top50)

    print(f'[PLAN] stocks={len(stock_ids)}  threshold={args.threshold}  '
          f'overwrite={args.overwrite}')
    print(f'       des_dir={args.des_dir}')
    print(f'       prices_dir={args.prices_dir}')
    print(f'       dest_dir={args.dest_dir}')

    copy_constituents(args.dest_dir, args.overwrite)

    n_written = 0
    n_skipped = 0
    fails: list[str] = []
    for sym in stock_ids:
        try:
            des = load_des_probs(sym, args.des_dir)
            px = load_prices(sym, args.prices_dir)
            df = build_one_csv(sym, px, des, args.threshold)
        except (FileNotFoundError, RuntimeError, KeyError) as err:
            fails.append(f'{sym}: {err}')
            continue

        head_date = df['<DATE>'].iloc[0]
        tail_date = df['<DATE>'].iloc[-1]
        des_pos = int((df['<DES>'] == 1).sum())
        print(f'  {sym}: {len(df)} rows  {head_date}..{tail_date}  '
              f'DES=1: {des_pos}/{len(df)}')
        dst = args.dest_dir / f'{sym}_all.csv'
        if write_csv(df, dst, overwrite=args.overwrite):
            n_written += 1
        else:
            n_skipped += 1

    print(f'\n[SUMMARY] wrote={n_written}  skipped_existing={n_skipped}  '
          f'failed={len(fails)}')
    if fails:
        print('Failures:')
        for f in fails:
            print(f'  - {f}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
