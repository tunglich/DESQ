"""Deterministic backtest of a trained DQN checkpoint on the held-out test span.

Test window (inclusive): 2024-01-02 ~ 2026-03-30

Reports:
  * model cumulative return %  (env reward sum, matches training reward scale)
  * Buy&Hold cumulative return % over the same span, after TW retail costs

Usage:
    python src/backtest.py --symbol 2330
    python src/backtest.py --symbol 2330 --model trained_models/2330_all.data
    python src/backtest.py --all --out backtest_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib import data, environ, models  # noqa: E402

TEST_START = pd.Timestamp("2024-01-02")
TEST_END = pd.Timestamp("2026-03-30")


def slice_test_csv(csv_path: Path, tmp_path: Path,
                   start: pd.Timestamp = TEST_START, end: pd.Timestamp = TEST_END) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    ok = (df["<OPEN>"] > 0) & (df["<HIGH>"] > 0) & (df["<LOW>"] > 0) & (df["<CLOSE>"] > 0)
    df = df.loc[ok].copy()
    mask = (df["<DATE>"] >= start) & (df["<DATE>"] <= end)
    df = df.loc[mask].sort_values("<DATE>").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{csv_path.name}: no rows in test window {start.date()}~{end.date()}")
    out = df.copy()
    out["<DATE>"] = out["<DATE>"].dt.strftime("%Y-%m-%d")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(tmp_path, index=False, lineterminator="\n")
    return df


def buy_and_hold_pct(df: pd.DataFrame, commission_buy: float, commission_sell: float) -> float:
    first_open = float(df["<OPEN>"].iloc[0])
    last_close = float(df["<CLOSE>"].iloc[-1])
    ret_pct = 100.0 * (last_close - first_open) / first_open
    ret_pct -= commission_buy  # entry cost as %
    ret_pct -= commission_sell  # exit cost as %
    return ret_pct


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_hash(bars_count: int, commission_buy: float,
                   commission_sell: float, epsilon: float) -> str:
    payload = json.dumps({
        "bars_count": bars_count,
        "commission_buy": commission_buy,
        "commission_sell": commission_sell,
        "epsilon": epsilon,
        "reset_on_close": False,
        "reward_on_close": False,
        "state_1d": True,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def run_backtest(model_path: Path, test_csv: Path,
                 commission_buy: float, commission_sell: float,
                 bars_count: int = 10, device: str = "cpu",
                 epsilon: float = 0.0, capture_trace: bool = False) -> dict:
    prices = {"TW": data.load_relative(str(test_csv))}
    env = environ.StocksEnv(
        prices,
        commission=commission_buy,
        commission_buy=commission_buy,
        commission_sell=commission_sell,
        reset_on_close=False,
        reward_on_close=False,
        state_1d=True,
        volumes=False,
        random_ofs_on_reset=False,
    )

    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)
    net = models.DQNConv1DLarge(obs_shape, n_actions).to(device)
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    n_buy = n_close = n_skip = 0
    nav = 1.0
    trace_rows: list[dict] = []
    trace_frame = pd.read_csv(test_csv) if capture_trace else None
    checkpoint_hash = sha256_file(model_path) if capture_trace else ""
    eval_hash = evaluator_hash(bars_count, commission_buy, commission_sell, epsilon)
    with torch.no_grad():
        while True:
            obs_v = torch.tensor(np.array([obs], dtype=np.float32)).to(device)
            q = net(obs_v)
            action_idx = int(q.max(dim=1)[1].item())
            if epsilon > 0.0 and np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            if action_idx == 0:
                n_skip += 1
            elif action_idx == 1:
                n_buy += 1
            else:
                n_close += 1
            offset_before = env._state._offset
            position_before = bool(env._state.have_position)
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            position_after = bool(env._state.have_position)
            total_reward += float(reward)
            nav *= 1.0 + float(reward) / 100.0
            steps += 1
            if trace_frame is not None:
                cost_pct = 0.0
                if action_idx == 1 and not position_before:
                    cost_pct = commission_buy
                elif action_idx == 2 and position_before:
                    cost_pct = commission_sell
                trace_rows.append({
                    "date": trace_frame["<DATE>"].iloc[offset_before],
                    "action": environ.Actions(action_idx).name.lower(),
                    "position_before": int(position_before),
                    "position_after": int(position_after),
                    "reward_pct": float(reward),
                    "cost_pct": cost_pct,
                    "nav": nav,
                    "des_probability": float(trace_frame["<DES>"].iloc[offset_before]),
                    "checkpoint_sha256": checkpoint_hash,
                    "evaluator_sha256": eval_hash,
                })
            if terminated or truncated:
                break
    return {
        "model_pct": total_reward,
        "steps": steps,
        "n_buy": n_buy,
        "n_close": n_close,
        "n_skip": n_skip,
        "trace": trace_rows,
        "evaluator_sha256": eval_hash,
    }


def load_top50_symbols(data_dir: Path) -> list[str]:
    cons = data_dir / "tw50_2023-12-29.csv"
    if not cons.is_file():
        raise SystemExit(f"missing constituent list: {cons}")
    with cons.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[1].strip() for row in reader if row and row[0].strip().isdigit()]


def resolve_model_path(models_dir: Path, symbol: str) -> Path:
    p = models_dir / f"{symbol}_all.data"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol")
    ap.add_argument("--all", action="store_true", help="Backtest every TW50 symbol")
    ap.add_argument("--model", type=Path, default=None, help="Override checkpoint path")
    ap.add_argument("--models-dir", type=Path, default=REPO_ROOT / "trained_models")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--tmp-dir", type=Path, default=REPO_ROOT / "saves" / "_backtest_tmp")
    ap.add_argument("--bars", type=int, default=10)
    ap.add_argument("--commission-buy", type=float, default=0.1425)
    ap.add_argument("--commission-sell", type=float, default=0.4425)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="Write summary CSV")
    ap.add_argument("--trace-dir", type=Path, default=None,
                    help="Write deterministic per-symbol daily monitoring traces")
    args = ap.parse_args()

    device = "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"

    jobs: list[str] = []
    if args.all:
        jobs.extend(load_top50_symbols(args.data_dir))
    else:
        if not args.symbol:
            ap.error("--symbol is required unless --all is set")
        jobs.append(args.symbol)

    rows: list[dict] = []
    for sym in jobs:
        csv_path = args.data_dir / f"{sym}_all.csv"
        tmp_test = args.tmp_dir / f"{sym}_all_test.csv"
        if not csv_path.is_file():
            print(f"[skip] {sym}: missing {csv_path}")
            continue
        try:
            df = slice_test_csv(csv_path, tmp_test)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sym}: {e}")
            continue

        model_path = args.model if (args.model and not args.all) else None
        if model_path is None:
            try:
                model_path = resolve_model_path(args.models_dir, sym)
            except FileNotFoundError as e:
                print(f"[skip] {sym}: no model at {e}")
                continue

        try:
            res = run_backtest(model_path, tmp_test,
                               commission_buy=args.commission_buy,
                               commission_sell=args.commission_sell,
                               bars_count=args.bars, device=device,
                               epsilon=args.epsilon,
                               capture_trace=args.trace_dir is not None)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {sym}: {e}")
            continue

        bh_pct = buy_and_hold_pct(df, args.commission_buy, args.commission_sell)
        n_rows = len(df)
        row = {
            "symbol": sym,
            "test_start": df["<DATE>"].iloc[0].strftime("%Y-%m-%d"),
            "test_end": df["<DATE>"].iloc[-1].strftime("%Y-%m-%d"),
            "n_rows": n_rows,
            "model_pct": round(res["model_pct"], 4),
            "bh_pct": round(bh_pct, 4),
            "excess_pct": round(res["model_pct"] - bh_pct, 4),
            "n_buy": res["n_buy"], "n_close": res["n_close"], "n_skip": res["n_skip"],
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "evaluator_sha256": res["evaluator_sha256"],
        }
        rows.append(row)
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = args.trace_dir / f"{sym}_daily_trace.csv"
            pd.DataFrame(res["trace"]).to_csv(trace_path, index=False, lineterminator="\n")
            print(f"  trace: {trace_path}")
        print(f"  {sym}: model={row['model_pct']:+7.2f}%  BH={row['bh_pct']:+7.2f}%  "
              f"excess={row['excess_pct']:+7.2f}%  rows={n_rows}  "
              f"buys={row['n_buy']} closes={row['n_close']}")

    if args.out and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote summary: {args.out}  ({len(rows)} rows)")
        model_avg = float(np.mean([r["model_pct"] for r in rows]))
        bh_avg = float(np.mean([r["bh_pct"] for r in rows]))
        print(f"Mean model_pct={model_avg:+.2f}%   Mean bh_pct={bh_avg:+.2f}%   "
              f"Mean excess={model_avg - bh_avg:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
