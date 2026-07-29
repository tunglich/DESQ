"""Abbade & Costa 2026 MACE backtest — parametric over universe.

Usage:
    python mace_backtest.py dow30
    python mace_backtest.py sp100
    python mace_backtest.py ndx100

Reuses 5 SB3 agents pre-trained per universe and re-runs them under two cost
models (flat baseline vs Almgren-Chriss). See env_mace.py for AC details.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- FinRL path plumbing --------------------------------------------------
FINRL_ROOT = Path(r"d:\US_stock\FinRL")
WS = Path(r"d:\US_stock")
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))
sys.path.insert(0, str(WS / "baselines" / "mi_abbade"))

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3   # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402
from env_mace import MACEStockTradingEnv                # noqa: E402

# --- Universe registry -----------------------------------------------------
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

UNIVERSES: Dict[str, Dict] = {
    "dow30": {
        "label":            "Dow 30",
        # NOTE: The active `train_data.csv` / `trade_data.csv` in FinRL root
        # were overwritten with an older 2005-2021 / 2022-2023 pair for a
        # separate experiment. Use the `_2005_2023.bak.csv` / `_2024_20260330.bak.csv`
        # snapshots that correspond to the 2024-2026 backtest here.
        "train_file":       FINRL_ROOT / "train_data_2005_2023.bak.csv",
        "trade_file":       FINRL_ROOT / "trade_data_2024_20260330.bak.csv",
        "trained_dir":      FINRL_ROOT / "trained_models",
        "extra_env_kwargs": {"turbulence_threshold": 70, "risk_indicator_col": "vix"},
        "des_equity":       WS / "backtest_portfolio_US" / "equity_dow30_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":     "backtest_dow30_2024_20260330",
        "index_ticker":     "^DJI",
        "extra_benchmarks": {"DIA": "DIA"},   # cap-weighted Dow ETF
    },
    "sp100": {
        "label":            "S&P 100",
        "train_file":       FINRL_ROOT / "sp100_train.pkl",
        "trade_file":       FINRL_ROOT / "sp100_trade.pkl",
        "trained_dir":      FINRL_ROOT / "sp100_variantA_trained_models",
        "extra_env_kwargs": {},
        "des_equity":       WS / "backtest_portfolio_US" / "equity_sp100_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":     "backtest_sp100_2024_20260330",
        "index_ticker":     "^OEX",
        "extra_benchmarks": {"OEF": "OEF"},   # iShares S&P 100 ETF
    },
    "ndx100": {
        "label":            "Nasdaq 100",
        "train_file":       FINRL_ROOT / "ndx100_train.pkl",
        "trade_file":       FINRL_ROOT / "ndx100_trade.pkl",
        "trained_dir":      FINRL_ROOT / "ndx100_variantA_trained_models",
        "extra_env_kwargs": {},
        "des_equity":       WS / "backtest_portfolio_US" / "equity_ndx100_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":     "backtest_ndx100_2024_20260330",
        "index_ticker":     "^NDX",
        "extra_benchmarks": {"QQEW": "QQEW", "QQQ": "QQQ"},
    },
}

TRADE_START = pd.Timestamp("2024-01-02")
TRADE_END   = pd.Timestamp("2026-03-30")
INITIAL_AMOUNT = 1_000_000

BASELINE_BUY_COST = 0.0005    # 0.05 % (roundtrip 0.10 %)
BASELINE_SELL_COST = 0.0005

AC_ALPHA = 0.5
AC_BETA = 1.0
AC_SPREAD_BPS = 5.0
AC_HALF_LIFE = 5.0

AGENT_CLS = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "td3": TD3, "sac": SAC}


# --- Helpers --------------------------------------------------------------
def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.index = df.date.factorize()[0]
    return df


def load_trained_agents(trained_dir: Path) -> Dict[str, object]:
    out = {}
    for name, cls in AGENT_CLS.items():
        path = trained_dir / f"agent_{name}.zip"
        if not path.exists():
            print(f"  [skip] {name.upper()}: not trained ({path})")
            continue
        try:
            out[name] = cls.load(str(path.with_suffix("")))
            print(f"  loaded {name.upper()} from {path.name}")
        except Exception as exc:
            print(f"  [err] failed to load {name.upper()}: {exc}")
    return out


def compute_adv_and_sigma(train_df: pd.DataFrame, tickers: list[str]) \
        -> tuple[pd.Series, pd.Series]:
    tail = train_df.groupby("tic").tail(60)
    adv, sigma = {}, {}
    for tic, sub in tail.groupby("tic"):
        sub = sub.sort_values("date")
        adv[tic] = float(sub["volume"].tail(20).mean())
        lret = np.log(sub["close"] / sub["close"].shift(1)).dropna()
        sigma[tic] = float(lret.tail(20).std())
    fb_adv = float(np.median(list(adv.values())))
    fb_sig = float(np.median(list(sigma.values())))
    adv_s = pd.Series({t: adv.get(t, fb_adv) for t in tickers})
    sig_s = pd.Series({t: sigma.get(t, fb_sig) for t in tickers})
    return adv_s, sig_s


def _rollout(model, env) -> pd.Series:
    obs, _ = env.reset()
    n_days = len(env.df.index.unique())
    for _ in range(n_days):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    dates = list(env.date_memory)
    values = list(env.asset_memory)
    if len(dates) == len(values) + 1:
        dates = dates[: len(values)]
    elif len(values) == len(dates) + 1:
        values = values[: len(dates)]
    return pd.Series(values, index=pd.to_datetime(dates), name="account_value")


def rollout_agent(model, name: str, trade_df: pd.DataFrame,
                  adv: pd.Series, sigma: pd.Series,
                  cost_mode: str, extra_env_kwargs: Dict) -> dict:
    stock_dim = trade_df["tic"].nunique()
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    tickers = trade_df.loc[0, "tic"].tolist() \
        if hasattr(trade_df.loc[0, "tic"], "tolist") \
        else [trade_df.loc[0, "tic"]]
    common = dict(
        df=trade_df, stock_dim=stock_dim, hmax=100,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0] * stock_dim,
        state_space=state_space, action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        reward_scaling=1e-4,
    )
    if cost_mode == "baseline":
        env = StockTradingEnv(
            **common, **extra_env_kwargs,
            buy_cost_pct=[BASELINE_BUY_COST] * stock_dim,
            sell_cost_pct=[BASELINE_SELL_COST] * stock_dim,
        )
    else:  # ac
        env = MACEStockTradingEnv(
            **common, **extra_env_kwargs,
            buy_cost_pct=[0.0] * stock_dim,
            sell_cost_pct=[0.0] * stock_dim,
            adv={t: float(adv[t]) for t in tickers},
            sigma={t: float(sigma[t]) for t in tickers},
            alpha_ac=AC_ALPHA, beta_ac=AC_BETA,
            spread_bps=AC_SPREAD_BPS, perm_half_life=AC_HALF_LIFE,
        )

    t0 = time.time()
    print(f"    [{name} / {cost_mode}] rolling out ...", flush=True)
    equity = _rollout(model, env)
    dt = time.time() - t0
    out = dict(
        name=name, cost_mode=cost_mode, equity=equity,
        total_cost=float(env.cost), trades=int(env.trades), elapsed=dt,
    )
    if isinstance(env, MACEStockTradingEnv):
        out["daily_cost"]     = pd.Series(env.mace_daily_cost)
        out["daily_turnover"] = pd.Series(env.mace_daily_turnover)
        out["daily_pov"]      = pd.Series(env.mace_daily_pov)
    return out


def _metrics(equity: pd.Series) -> dict:
    s = equity.dropna()
    if len(s) < 2:
        return dict(final=float("nan"), ret_pct=float("nan"),
                    sharpe=float("nan"), mdd=float("nan"))
    r = s.pct_change().dropna()
    return dict(
        final=float(s.iloc[-1]),
        ret_pct=float(s.iloc[-1] / s.iloc[0] - 1) * 100,
        sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        mdd=float(((s - s.cummax()) / s.cummax()).min()) * 100,
    )


# --- Main ------------------------------------------------------------------
def main(universe_key: str) -> int:
    t_all = time.time()
    if universe_key not in UNIVERSES:
        print(f"unknown universe {universe_key!r}", file=sys.stderr)
        return 2
    U = UNIVERSES[universe_key]
    out_dir = WS / "baselines" / "mi_abbade" / U["out_dir_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== MACE backtest — universe={U['label']!r} ===")
    print(f"  train: {U['train_file']}")
    print(f"  trade: {U['trade_file']}")
    print(f"  models: {U['trained_dir']}")

    train = _load_df(U["train_file"])
    trade = _load_df(U["trade_file"])
    print(f"  train: {len(train):,} rows / {train['tic'].nunique()} tickers "
          f"/ {train['date'].min()} ~ {train['date'].max()}")
    print(f"  trade: {len(trade):,} rows / {trade['tic'].nunique()} tickers "
          f"/ {trade['date'].min()} ~ {trade['date'].max()}")

    trade_tickers = sorted(trade["tic"].unique())
    adv, sigma = compute_adv_and_sigma(train, trade_tickers)
    print(f"  ADV median={adv.median():,.0f}   sigma median={sigma.median():.4f}")

    agents = load_trained_agents(U["trained_dir"])
    if not agents:
        print(f"[ERR] no trained agents in {U['trained_dir']}")
        return 1

    print(f"\n=== Rolling out {len(agents)} agents × 2 cost models ===")
    results: Dict[tuple, dict] = {}
    for name, model in agents.items():
        for cost_mode in ("baseline", "ac"):
            try:
                results[(name, cost_mode)] = rollout_agent(
                    model, name, trade, adv, sigma, cost_mode,
                    U["extra_env_kwargs"],
                )
            except Exception as exc:
                print(f"    [{name} / {cost_mode}] FAILED: {exc}")
                import traceback; traceback.print_exc()

    if not results:
        print("[ERR] no successful rollouts")
        return 1

    # Save per-config CSVs
    metric_rows = []
    for (name, mode), r in results.items():
        eq = r["equity"]; eq.name = f"equity_{name}_{mode}"
        eq.to_csv(out_dir / f"equity_{name}_{mode}.csv", header=True)
        m = _metrics(eq)
        metric_rows.append({
            "agent": name.upper(), "cost_model": mode,
            "final_$": m["final"], "return_%": m["ret_pct"],
            "sharpe": m["sharpe"], "max_dd_%": m["mdd"],
            "total_cost_$": r["total_cost"], "trades": r["trades"],
        })
        if "daily_cost" in r:
            pd.DataFrame({
                "day": np.arange(len(r["daily_cost"])),
                "cost_$": r["daily_cost"].values,
                "turnover_$": r["daily_turnover"].values,
                "pov": r["daily_pov"].values,
            }).to_csv(out_dir / f"daily_costs_{name}_{mode}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # Benchmarks (yfinance)
    import yfinance as yf
    def _yf(sym: str) -> pd.Series:
        d = yf.download(sym, start=TRADE_START,
                        end=TRADE_END + pd.Timedelta(days=1),
                        progress=False, auto_adjust=True)
        if d is None or d.empty:
            return pd.Series(dtype=float)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        return d["Close"].rename(sym)

    bench_index = _yf(U["index_ticker"])
    extra_bench = {n: _yf(sym) for n, sym in U["extra_benchmarks"].items()}

    des = None
    if U["des_equity"].exists():
        des_df = pd.read_csv(U["des_equity"], index_col=0, parse_dates=True)
        des = des_df["portfolio_equity"].loc[TRADE_START:TRADE_END]

    # Chart 1: baseline vs AC per agent
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = {"a2c": "#1F77B4", "ppo": "#8CC63F", "ddpg": "#FF7F0E",
               "td3": "#9467BD", "sac": "#C44E52"}
    for (name, mode), r in results.items():
        ls = "-" if mode == "ac" else "--"
        ax.plot(r["equity"].index, r["equity"].values,
                label=f"{name.upper()} ({mode})",
                color=palette.get(name, "#333"),
                linestyle=ls, linewidth=1.4, alpha=0.85)
    for name, b in extra_bench.items():
        if not b.empty:
            b2 = INITIAL_AMOUNT * b / b.iloc[0]
            ax.plot(b2.index, b2.values, label=name, linewidth=1.4,
                    linestyle=":", color="#666666")
    if not bench_index.empty:
        b2 = INITIAL_AMOUNT * bench_index / bench_index.iloc[0]
        ax.plot(b2.index, b2.values, label=U["index_ticker"],
                linewidth=1.5, linestyle="-.", color="black")
    ax.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title(f"{U['label']} — Abbade 2026 MACE (baseline vs AC) "
                 f"{TRADE_START.date()} ~ {TRADE_END.date()}")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(out_dir / "baseline_vs_ac.png", dpi=140); plt.close()
    print(f"[OK] wrote {out_dir / 'baseline_vs_ac.png'}")

    # Chart 2: paper vs DES (use best-AC agent + PPO baseline)
    fig, ax = plt.subplots(figsize=(14, 8))
    best_ac = max(
        ((k, r) for k, r in results.items() if k[1] == "ac"),
        key=lambda kv: _metrics(kv[1]["equity"])["ret_pct"],
    )
    (best_name, _), best_r = best_ac
    ppo_base = results.get(("ppo", "baseline"))
    if ppo_base is not None:
        ax.plot(ppo_base["equity"].index, ppo_base["equity"].values,
                label="PPO (baseline)", color="#8CC63F", linewidth=1.8)
    ax.plot(best_r["equity"].index, best_r["equity"].values,
            label=f"{best_name.upper()} (AC)",
            color=palette.get(best_name, "#C44E52"), linewidth=1.8)
    if des is not None and not des.empty:
        des2 = INITIAL_AMOUNT * des / des.iloc[0]
        ax.plot(des2.index, des2.values, label="DES (ours)",
                color="#1F77B4", linewidth=2.6)
    for name, b in extra_bench.items():
        if not b.empty:
            b2 = INITIAL_AMOUNT * b / b.iloc[0]
            ax.plot(b2.index, b2.values, label=name, linewidth=1.2,
                    linestyle=":", color="#666666", alpha=0.85)
    if not bench_index.empty:
        b2 = INITIAL_AMOUNT * bench_index / bench_index.iloc[0]
        ax.plot(b2.index, b2.values, label=U["index_ticker"],
                linewidth=1.4, linestyle="-.", color="black")
    ax.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title(f"{U['label']} — MACE Paper vs DES (Ours)  "
                 f"{TRADE_START.date()} ~ {TRADE_END.date()}")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(out_dir / "paper_vs_des.png", dpi=140); plt.close()
    print(f"[OK] wrote {out_dir / 'paper_vs_des.png'}")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Abbade 2026 MACE — {U['label']} reproduction\n")
        f.write(f"Trade period: {TRADE_START.date()} ~ {TRADE_END.date()}\n")
        f.write(f"Initial capital: ${INITIAL_AMOUNT:,}\n")
        f.write(f"AC params: alpha={AC_ALPHA} beta={AC_BETA} "
                f"spread_bps={AC_SPREAD_BPS} half_life={AC_HALF_LIFE}\n\n")
        f.write(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        f.write("\n")
    print(f"[OK] wrote {out_dir / 'summary.txt'}")
    print(f"[DONE] elapsed {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mace_backtest.py <dow30|sp100|ndx100>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1].lower()))
