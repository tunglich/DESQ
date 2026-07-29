"""tw50_des.py

Stage 3 of the TW-50 pipeline.

Dynamic Ensemble Selection (KNORA-E) over the 5 aspect probabilities emitted
by Stage 2 (tw50_dflood.py), followed by a signal-pattern-driven backtest.

This stage intentionally does NOT use CUSUM anywhere:
    - No CUSUM as input feature.
    - No CUSUM-based buy/sell gating in the backtest.
    - Aggregated DES probability is used directly for signal generation.

Sentiment is also excluded (5 aspects, not 6).

Usage
-----
    python tw50_des.py --stock-ids 2330 --no-show
    python tw50_des.py --top50

Inputs
------
    ./artifacts/dflood/pred/<stock>_<aspect>.csv
        columns: Date, y_true_20, prob_up, prob_down
    ./prices/<stock_id>.csv    (user-supplied)
        columns: Date, Open, High, Low, Close, Volume

Outputs
-------
    ./artifacts/des/pred/DES_<stock>.csv, RF_<stock>.csv
    ./artifacts/des/models/DES_<stock>.pkl, RF_<stock>.pkl
    ./artifacts/des/backtest/<stock>_equity.csv
    ./artifacts/des/backtest/<stock>_backtest.png       (if matplotlib available)
    ./artifacts/des/backtest/summary.csv                (rolling summary of runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

try:
    from deslib.des.knora_e import KNORAE
except ImportError as err:  # pragma: no cover - install-time issue
    raise SystemExit(
        'deslib is required for tw50_des.py. Install with: pip install deslib==0.3.7'
    ) from err

# Optional plotting.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

import joblib

from tw50_flood import (
    REPO_ROOT,
    TEST_END,
    TEST_START,
    TRAIN_END,
    load_top50_ids,
    parse_stock_ids,
)


# =============================================================================
# Constants
# =============================================================================

# Sentiment is excluded per requirement.
FEATURE_ORDER = ('fundamental', 'trade', 'tech_trend', 'moment', 'macro')

DFLOOD_PRED_DIR = Path(os.environ.get('DFLOOD_ROOT', REPO_ROOT / 'artifacts' / 'dflood')) / 'pred'
PRICES_DIR = Path(os.environ.get('PRICES_DIR', REPO_ROOT / 'prices'))

DES_ROOT = Path(os.environ.get('DES_ROOT', REPO_ROOT / 'artifacts' / 'des'))
DES_PRED_DIR = DES_ROOT / 'pred'
DES_MODEL_DIR = DES_ROOT / 'models'
DES_BT_DIR = DES_ROOT / 'backtest'
for _p in (DES_PRED_DIR, DES_MODEL_DIR, DES_BT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# DES training window: uses the last chunk of the train period up to TRAIN_END.
DES_TRAIN_START = '2020-01-01'
DES_TRAIN_END = TRAIN_END

# Signal-pattern parameters (same defaults as legacy DES script).
DEFAULT_LONG = 3
DEFAULT_SHORT = 3
DEFAULT_S2L = 3
DEFAULT_L2S = 3
DEFAULT_THRESHOLD = 0.5

# Trading model constants.
BUY_FEE = 0.001425      # brokerage fee
SELL_FEE_TAX = 0.004425  # brokerage + 0.3% transaction tax
INITIAL_CASH = 50_000_000
SHARES_PER_LOT = 1000


# =============================================================================
# Load per-aspect predictions -> stack into X_all
# =============================================================================


def load_aspect_predictions(stock_id: str) -> pd.DataFrame:
    """Load the 5 aspect prediction CSVs and stack their prob_up columns.

    Missing aspects are skipped (not raised). The result has one column per
    present aspect and a datetime index sorted ascending.
    """
    frames = []
    present = []
    for aspect in FEATURE_ORDER:
        fp = DFLOOD_PRED_DIR / f'{stock_id}_{aspect}.csv'
        if not fp.exists():
            print(f'[WARN] {stock_id}/{aspect}: prediction file missing at {fp}')
            continue
        df = pd.read_csv(fp, parse_dates=['Date'])
        df = df[~df['Date'].duplicated(keep='last')].sort_values('Date')
        s = pd.Series(df['prob_up'].to_numpy(), index=df['Date'], name=aspect)
        frames.append(s)
        present.append(aspect)
    if not frames:
        raise RuntimeError(f'{stock_id}: no aspect predictions found.')
    X_all = pd.concat(frames, axis=1)
    X_all.index.name = 'Date'
    X_all = X_all.ffill().bfill().fillna(0.5).astype(np.float64)
    return X_all[present]


def load_labels(stock_id: str, index: pd.DatetimeIndex) -> pd.Series:
    """Read y_20 label from the fundamental CSV, reindexed to the DES matrix."""
    fp = REPO_ROOT / 'features' / f'fundamental_{stock_id}.csv'
    if not fp.exists():
        raise FileNotFoundError(f'{stock_id}: fundamental CSV missing at {fp}')
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    if 'y_20' not in df.columns:
        raise KeyError(f'{stock_id}: y_20 column missing in {fp}')
    y = df['y_20'].reindex(index)
    y = y.ffill().bfill().fillna(0).astype(int).clip(0, 1)
    return y


def load_prices(stock_id: str) -> pd.DataFrame | None:
    """Load user-supplied price data. Returns None if the file is missing."""
    fp = PRICES_DIR / f'{stock_id}.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=['Date'])
    df = df[~df['Date'].duplicated(keep='last')].sort_values('Date').set_index('Date')
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        if c not in df.columns:
            raise KeyError(f'{fp}: missing column {c}')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(np.float64)


# =============================================================================
# Base classifier + KNORAE
# =============================================================================


def find_best_rf(X_train: pd.DataFrame, y_train: pd.Series,
                  n_iter: int = 20, cv: int = 5, random_state: int = 42) -> RandomForestClassifier:
    """RandomizedSearchCV over a compact RF grid; returns the best estimator."""
    grid = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_depth': [None, 10, 20, 40, 80],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
    }
    uniq, counts = np.unique(y_train, return_counts=True)
    counts_w = (1.0 / counts) * len(y_train)
    class_weights = dict(zip(uniq, counts_w))

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight=class_weights, random_state=random_state),
        param_distributions=grid,
        n_iter=n_iter,
        cv=cv,
        verbose=0,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train.to_numpy(), y_train.to_numpy())
    return search.best_estimator_


def train_or_load_des(stock_id: str, X_train: pd.DataFrame, y_train: pd.Series,
                       force: bool) -> tuple[RandomForestClassifier, KNORAE, dict]:
    des_path = DES_MODEL_DIR / f'DES_{stock_id}.pkl'
    rf_path = DES_MODEL_DIR / f'RF_{stock_id}.pkl'
    if not force and des_path.exists() and rf_path.exists():
        try:
            base_clf = joblib.load(rf_path)
            model = joblib.load(des_path)
            print(f'[{stock_id}] loaded cached RF + KNORAE')
            return base_clf, model, {'rf_path': str(rf_path), 'des_path': str(des_path)}
        except Exception as err:
            print(f'[{stock_id}] cache load failed ({err}); retraining.')

    print(f'[{stock_id}] fitting RandomForest ...')
    base_clf = find_best_rf(X_train, y_train)
    print(f'[{stock_id}] fitting KNORA-E ...')
    model = KNORAE(pool_classifiers=base_clf, k=10, DFP=True)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    joblib.dump(base_clf, rf_path)
    joblib.dump(model, des_path)
    return base_clf, model, {'rf_path': str(rf_path), 'des_path': str(des_path)}


# =============================================================================
# Signal patterns -> buy / sell series (NO CUSUM)
# =============================================================================


def _match_tail(arr: np.ndarray, i: int, pattern: list[int]) -> bool:
    n = len(pattern)
    if i < n - 1:
        return False
    return np.array_equal(arr[i - (n - 1): i + 1], np.asarray(pattern, dtype=arr.dtype))


def build_signals(prob_up: pd.Series, threshold: float,
                   long: int, short: int, s2l: int, l2s: int
                   ) -> tuple[pd.Series, pd.Series]:
    """Return (sig_buy, sig_sell) as +1/-1 pulses.

    Buy patterns:
        [0]*s2l + [1]*long        (standard reversal)
        [0, 1] + [1]*long         (fast up-flip)
        [1, 0] + [1]*long         (fast down-then-up)

    Sell pattern:
        [1]*l2s + [0]*short        (standard reversal down)
    """
    bin_signal = (prob_up > threshold).astype(int)
    arr = bin_signal.to_numpy()
    n = len(arr)
    buy_pats = [
        [0] * s2l + [1] * long,
        [0, 1] + [1] * long,
        [1, 0] + [1] * long,
    ]
    sell_pats = [
        [1] * l2s + [0] * short,
    ]
    sig_buy = np.zeros(n, dtype=int)
    sig_sell = np.zeros(n, dtype=int)
    for i in range(n):
        if any(_match_tail(arr, i, p) for p in buy_pats):
            sig_buy[i] = 1
        if any(_match_tail(arr, i, p) for p in sell_pats):
            sig_sell[i] = -1
    return (
        pd.Series(sig_buy, index=prob_up.index, name='sig_buy'),
        pd.Series(sig_sell, index=prob_up.index, name='sig_sell'),
    )


# =============================================================================
# Backtest (no CUSUM gating)
# =============================================================================


def backtest(stock_id: str, prob_up: pd.Series, stock_price: pd.DataFrame,
              threshold: float, long: int, short: int, s2l: int, l2s: int
              ) -> dict:
    """Simulate open-price fills. Buy/sell decisions driven purely by DES signal."""
    joined_idx = prob_up.index.intersection(stock_price.index)
    if len(joined_idx) < 5:
        raise RuntimeError(f'{stock_id}: joined DES/price index too short ({len(joined_idx)})')
    prob = prob_up.loc[joined_idx]
    px = stock_price.loc[joined_idx]

    sig_buy, sig_sell = build_signals(prob, threshold, long, short, s2l, l2s)

    n = len(joined_idx)
    cash = np.zeros(n)
    shares = np.zeros(n)
    asset = np.zeros(n)
    cost = np.zeros(n)
    buy_act = np.zeros(n, dtype=int)
    sell_act = np.zeros(n, dtype=int)

    cash[0] = INITIAL_CASH
    asset[0] = cash[0] + shares[0] * px['Close'].iloc[0] * SHARES_PER_LOT

    open_arr = px['Open'].to_numpy()
    close_arr = px['Close'].to_numpy()
    sb = sig_buy.to_numpy()
    ss = sig_sell.to_numpy()
    acc_buy = 0
    acc_sell = 0

    for i in range(1, n):
        # Buy: prior day flagged buy, currently flat.
        if sb[i - 1] == 1 and shares[i - 1] == 0:
            lots = int(cash[i - 1] // (open_arr[i] * SHARES_PER_LOT))
            if lots > 0:
                shares[i] = lots
                cost[i] = lots * open_arr[i] * BUY_FEE * SHARES_PER_LOT
                cash[i] = cash[i - 1] - cost[i] - lots * open_arr[i] * SHARES_PER_LOT
                asset[i] = cash[i] + shares[i] * close_arr[i] * SHARES_PER_LOT
                buy_act[i] = 1
                acc_buy += 1
                continue
        # Sell: prior day flagged sell, currently long.
        if ss[i - 1] == -1 and shares[i - 1] > 0:
            cost[i] = shares[i - 1] * open_arr[i] * SELL_FEE_TAX * SHARES_PER_LOT
            shares[i] = 0
            cash[i] = shares[i - 1] * open_arr[i] * SHARES_PER_LOT - cost[i] + cash[i - 1]
            asset[i] = cash[i] + shares[i] * close_arr[i] * SHARES_PER_LOT
            sell_act[i] = 1
            acc_sell += 1
            continue
        # Hold.
        cash[i] = cash[i - 1]
        shares[i] = shares[i - 1]
        asset[i] = cash[i] + shares[i] * close_arr[i] * SHARES_PER_LOT

    ret_model = asset / np.maximum(np.roll(asset, 1), 1e-9) - 1.0
    ret_model[0] = 0.0
    ret_stock = close_arr / np.roll(close_arr, 1) - 1.0
    ret_stock[0] = 0.0

    cum_model = np.cumprod(1.0 + ret_model) - 1.0
    cum_stock = np.cumprod(1.0 + ret_stock) - 1.0

    equity = pd.DataFrame({
        'Date': joined_idx,
        'prob_up': prob.to_numpy(),
        'sig_buy': sb,
        'sig_sell': ss,
        'buy_action': buy_act,
        'sell_action': sell_act,
        'cash': cash,
        'shares': shares,
        'asset': asset,
        'ret_model': ret_model,
        'ret_stock': ret_stock,
        'cum_model': cum_model,
        'cum_stock': cum_stock,
    })

    total_ret_model = float(cum_model[-1])
    total_ret_stock = float(cum_stock[-1])

    return {
        'equity': equity,
        'acc_buy': int(acc_buy),
        'acc_sell': int(acc_sell),
        'total_ret_model': total_ret_model,
        'total_ret_stock': total_ret_stock,
        'excess_ret': total_ret_model - total_ret_stock,
    }


def plot_backtest(stock_id: str, equity: pd.DataFrame, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(equity['Date'], equity['cum_model'], label='DES strategy', color='tab:blue')
    ax1.plot(equity['Date'], equity['cum_stock'], label='Buy & hold', color='tab:gray', alpha=0.7)
    ax1.set_ylabel('Cumulative return')
    ax1.set_title(f'{stock_id} — DES backtest ({equity["Date"].iloc[0].date()} .. {equity["Date"].iloc[-1].date()})')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2.plot(equity['Date'], equity['prob_up'], label='DES prob_up', color='tab:orange')
    ax2.axhline(0.5, color='k', ls='--', lw=0.6, alpha=0.5)
    buys = equity[equity['buy_action'] == 1]
    sells = equity[equity['sell_action'] == 1]
    ax2.scatter(buys['Date'], buys['prob_up'], marker='^', c='green', label='buy', s=40)
    ax2.scatter(sells['Date'], sells['prob_up'], marker='v', c='red', label='sell', s=40)
    ax2.set_ylabel('DES prob_up')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# End-to-end runner for one stock
# =============================================================================


def run_one(stock_id: str, *, threshold: float, long: int, short: int,
             s2l: int, l2s: int, force: bool) -> dict:
    print(f'\n=== DES: {stock_id} ===')
    X_all = load_aspect_predictions(stock_id)
    y_all = load_labels(stock_id, X_all.index)

    X_train = X_all.loc[DES_TRAIN_START:DES_TRAIN_END]
    y_train = y_all.loc[DES_TRAIN_START:DES_TRAIN_END]
    X_test = X_all.loc[TEST_START:TEST_END]
    if len(X_train) < 100:
        raise RuntimeError(f'{stock_id}: DES train slice too short ({len(X_train)})')
    if len(X_test) < 20:
        raise RuntimeError(f'{stock_id}: DES test slice too short ({len(X_test)})')

    base_clf, model, paths = train_or_load_des(stock_id, X_train, y_train, force=force)

    prob_des = pd.Series(
        model.predict_proba(X_all.to_numpy())[:, 1],
        index=X_all.index, name='DES',
    )
    prob_rf = pd.Series(
        base_clf.predict_proba(X_all.to_numpy())[:, 1],
        index=X_all.index, name='RF',
    )
    prob_des.to_csv(DES_PRED_DIR / f'DES_{stock_id}.csv', header=True)
    prob_rf.to_csv(DES_PRED_DIR / f'RF_{stock_id}.csv', header=True)

    # AGG_DES is the aggregated DES probability. No CUSUM blending applied.
    prob_des_test = prob_des.loc[TEST_START:TEST_END]

    prices = load_prices(stock_id)
    if prices is None:
        print(f'[{stock_id}] no price CSV at {PRICES_DIR / (stock_id + ".csv")}; '
              f'skipping backtest.')
        return {
            'stock_id': stock_id,
            'has_prices': False,
            'n_test_days': int(len(prob_des_test)),
            'paths': paths,
        }

    price_test = prices.loc[TEST_START:TEST_END]
    bt = backtest(stock_id, prob_des_test, price_test,
                  threshold=threshold, long=long, short=short, s2l=s2l, l2s=l2s)
    equity_path = DES_BT_DIR / f'{stock_id}_equity.csv'
    bt['equity'].to_csv(equity_path, index=False)
    print(f'[{stock_id}] cum_model={bt["total_ret_model"]:.3f}, '
          f'cum_stock={bt["total_ret_stock"]:.3f}, '
          f'excess={bt["excess_ret"]:.3f}, buys={bt["acc_buy"]}, sells={bt["acc_sell"]}')

    if _HAVE_MPL:
        plot_backtest(stock_id, bt['equity'], DES_BT_DIR / f'{stock_id}_backtest.png')

    return {
        'stock_id': stock_id,
        'has_prices': True,
        'n_test_days': int(len(prob_des_test)),
        'acc_buy': bt['acc_buy'],
        'acc_sell': bt['acc_sell'],
        'total_ret_model': bt['total_ret_model'],
        'total_ret_stock': bt['total_ret_stock'],
        'excess_ret': bt['excess_ret'],
        'paths': paths,
    }


# =============================================================================
# CLI
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--top50', action='store_true', help='use all stocks from tw50_top50.csv')
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--long', type=int, default=DEFAULT_LONG)
    p.add_argument('--short', type=int, default=DEFAULT_SHORT)
    p.add_argument('--s2l', type=int, default=DEFAULT_S2L, help='short-to-long transition days')
    p.add_argument('--l2s', type=int, default=DEFAULT_L2S, help='long-to-short transition days')
    p.add_argument('--force', action='store_true', help='retrain even if cached models exist')
    p.add_argument('--no-show', action='store_true',
                   help='(kept for CLI compat; plots are always saved to disk without display)')
    args = p.parse_args(argv)

    stock_ids = parse_stock_ids(args.stock_ids, args.top50)
    print(f'[PLAN] DES over stocks={stock_ids}, threshold={args.threshold}, '
          f'long={args.long}, short={args.short}, s2l={args.s2l}, l2s={args.l2s}')

    rows = []
    for sid in stock_ids:
        try:
            row = run_one(sid,
                          threshold=args.threshold,
                          long=args.long, short=args.short,
                          s2l=args.s2l, l2s=args.l2s,
                          force=args.force)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {sid}: {exc}')
            import traceback
            traceback.print_exc()

    if rows:
        summary_rows = []
        for r in rows:
            summary_rows.append({
                'stock_id': r['stock_id'],
                'has_prices': r.get('has_prices', False),
                'n_test_days': r.get('n_test_days', 0),
                'acc_buy': r.get('acc_buy', 0),
                'acc_sell': r.get('acc_sell', 0),
                'total_ret_model': r.get('total_ret_model', np.nan),
                'total_ret_stock': r.get('total_ret_stock', np.nan),
                'excess_ret': r.get('excess_ret', np.nan),
            })
        pd.DataFrame(summary_rows).to_csv(DES_BT_DIR / 'summary.csv', index=False)
        print(f'\n[SUMMARY] {len(summary_rows)} rows -> {DES_BT_DIR / "summary.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
