"""Phase C — Abbade & Costa 2026 MACE stock-trading env reproduction on NDX 100.

We reuse the five SB3 agents already trained inside our FinRL fork
(`ndx100_variantA_trained_models/agent_{a2c,ppo,ddpg,td3,sac}.zip`, trained on
2015-01-02 ~ 2023-12-29 with a flat 10 bps baseline) and re-backtest them from
2024-01-02 to 2026-03-30 under **two cost models**:

    * Baseline    : flat 10 bps buy + 10 bps sell (matches training)
    * AC          : Almgren-Chriss non-linear impact with permanent-impact
                    exponential decay (τ½ = 5 days) — see env_mace.py.

For each agent × cost-model we compute the daily portfolio-value curve,
per-day trading cost, POV, turnover, then aggregate into a report and
overlay our DES (market-weighted) SP100 curve as a comparison.

Outputs (under baselines/mi_abbade/backtest_2024_20260330/):
    equity_<agent>_<cost>.csv     ...  daily portfolio equity per config
    metrics.csv                   ...  summary table (return, Sharpe, MDD, cost)
    daily_costs_<agent>_<cost>.csv .. trading cost / POV / turnover per day
    baseline_vs_des.png           ...  DES vs best AC agent(s) vs QQEW / QQQ
    baseline_vs_ac.png            ...  same 5 agents under baseline vs AC
    summary.txt                   ...  text summary
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

# --- FinRL fork path plumbing ---------------------------------------------
FINRL_ROOT = Path(r"d:\US_stock\FinRL")
os.chdir(FINRL_ROOT)
sys.path.insert(0, str(FINRL_ROOT))
sys.path.insert(0, str(Path(r"d:\US_stock\baselines\mi_abbade")))

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402
from env_mace import MACEStockTradingEnv  # noqa: E402

# --- Configuration ---------------------------------------------------------
WS = Path(r"d:\US_stock")
OUT_DIR = WS / "baselines" / "mi_abbade" / "backtest_2024_20260330"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODEL_DIR = FINRL_ROOT / "ndx100_variantA_trained_models"
DES_EQUITY = WS / "backtest_portfolio_US" / "equity_ndx100_market_2024-01-02_2026-03-31.csv"

TRADE_START = pd.Timestamp("2024-01-02")
TRADE_END = pd.Timestamp("2026-03-30")
INITIAL_AMOUNT = 1_000_000
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30",
              "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

BASELINE_BUY_COST = 0.001   # 10 bps
BASELINE_SELL_COST = 0.001  # 10 bps

# AC reference parameters.
AC_ALPHA = 0.5
AC_BETA = 1.0
AC_SPREAD_BPS = 5.0
AC_HALF_LIFE = 5.0

AGENT_CLS = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "td3": TD3, "sac": SAC}


# --- Helpers ---------------------------------------------------------------
def load_trained_agents() -> Dict[str, object]:
    out = {}
    for name, cls in AGENT_CLS.items():
        path = TRAINED_MODEL_DIR / f"agent_{name}.zip"
        if not path.exists():
            print(f"  [skip] {name.upper()}: not trained ({path})")
            continue
        try:
            out[name] = cls.load(str(path.with_suffix("")))
            print(f"  loaded {name.upper()} from {path.name}")
        except Exception as exc:
            print(f"  [err] failed to load {name.upper()}: {exc}")
    return out


def compute_adv_and_sigma(train_df: pd.DataFrame, trade_df: pd.DataFrame,
                          tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    """ADV20 & sigma20 estimated on the *train* tail (last 60 trading days
    before TRADE_START) — a stable, out-of-sample estimate.
    """
    tail = train_df.groupby("tic").tail(60)
    adv, sigma = {}, {}
    for tic, sub in tail.groupby("tic"):
        sub = sub.sort_values("date")
        adv[tic] = float(sub["volume"].tail(20).mean())
        lret = np.log(sub["close"] / sub["close"].shift(1)).dropna()
        sigma[tic] = float(lret.tail(20).std())
    # Coerce to same order as `tickers`
    fallback_adv = float(np.median(list(adv.values())))
    fallback_sig = float(np.median(list(sigma.values())))
    adv_s = pd.Series({t: adv.get(t, fallback_adv) for t in tickers})
    sig_s = pd.Series({t: sigma.get(t, fallback_sig) for t in tickers})
    return adv_s, sig_s


def _rollout(model, env) -> tuple[pd.Series, pd.Series | None]:
    """Deterministic rollout of a trained SB3 model in `env`.

    We roll out **manually** (without DummyVecEnv auto-reset) so that
    env.cost / env.trades / env.mace_daily_* are preserved after the last
    step for post-hoc inspection.

    Returns (asset_value_series indexed by date, actions df or None).
    """
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
        # occasional off-by-one; align lengths
        dates = dates[: len(values)]
    elif len(values) == len(dates) + 1:
        values = values[: len(dates)]
    s = pd.Series(values, index=pd.to_datetime(dates), name="account_value")
    return s, None


def rollout_agent(model, name: str, trade_df: pd.DataFrame,
                  adv: pd.Series, sigma: pd.Series,
                  cost_mode: str) -> dict:
    """Run one (agent, cost-mode) config; return dict of results."""
    stock_dim = trade_df["tic"].nunique()
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim

    common = dict(
        df=trade_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0] * stock_dim,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        reward_scaling=1e-4,
    )

    tickers = list(trade_df.loc[0, "tic"]) \
        if isinstance(trade_df.loc[0, "tic"], (list, np.ndarray)) \
        else trade_df.loc[0, :]["tic"].tolist()

    if cost_mode == "baseline":
        env = StockTradingEnv(
            **common,
            buy_cost_pct=[BASELINE_BUY_COST] * stock_dim,
            sell_cost_pct=[BASELINE_SELL_COST] * stock_dim,
        )
    elif cost_mode == "ac":
        env = MACEStockTradingEnv(
            **common,
            buy_cost_pct=[0.0] * stock_dim,   # AC-model handles all costs
            sell_cost_pct=[0.0] * stock_dim,
            adv={t: float(adv[t]) for t in tickers},
            sigma={t: float(sigma[t]) for t in tickers},
            alpha_ac=AC_ALPHA, beta_ac=AC_BETA,
            spread_bps=AC_SPREAD_BPS, perm_half_life=AC_HALF_LIFE,
        )
    else:
        raise ValueError(cost_mode)

    t0 = time.time()
    print(f"    [{name} / {cost_mode}] rolling out ...", flush=True)
    equity, _ = _rollout(model, env)
    dt = time.time() - t0

    out = {
        "name": name,
        "cost_mode": cost_mode,
        "equity": equity,
        "total_cost": float(env.cost),
        "trades": int(env.trades),
        "elapsed": dt,
    }
    if isinstance(env, MACEStockTradingEnv):
        out["daily_cost"] = pd.Series(env.mace_daily_cost)
        out["daily_turnover"] = pd.Series(env.mace_daily_turnover)
        out["daily_pov"] = pd.Series(env.mace_daily_pov)
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
def main() -> int:
    t_all = time.time()
    print("=== Loading data ===")
    train = pd.read_pickle(FINRL_ROOT / "ndx100_train.pkl")
    trade = pd.read_pickle(FINRL_ROOT / "ndx100_trade.pkl")
    train.index = train.date.factorize()[0]
    trade.index = trade.date.factorize()[0]
    print(f"  train: {len(train):,} rows / {train['tic'].nunique()} tickers "
          f"/ {train['date'].min()} ~ {train['date'].max()}")
    print(f"  trade: {len(trade):,} rows / {trade['tic'].nunique()} tickers "
          f"/ {trade['date'].min()} ~ {trade['date'].max()}")

    trade_tickers = sorted(trade["tic"].unique())
    print("\n=== Computing ADV20 / sigma20 (from train tail) ===")
    adv, sigma = compute_adv_and_sigma(train, trade, trade_tickers)
    print(f"  ADV summary : median={adv.median():.0f}  min={adv.min():.0f}  max={adv.max():.0f}")
    print(f"  sigma summary: median={sigma.median():.4f}  "
          f"min={sigma.min():.4f}  max={sigma.max():.4f}")

    print("\n=== Loading trained SB3 agents ===")
    agents = load_trained_agents()
    if not agents:
        print("[ERR] no trained agents available under "
              f"{TRAINED_MODEL_DIR} — aborting.")
        return 1

    print(f"\n=== Rolling out {len(agents)} agents × 2 cost models ===")
    results: Dict[tuple, dict] = {}
    for name, model in agents.items():
        for cost_mode in ("baseline", "ac"):
            key = (name, cost_mode)
            try:
                results[key] = rollout_agent(model, name, trade, adv, sigma, cost_mode)
            except Exception as exc:
                print(f"    [{name} / {cost_mode}] FAILED: {exc}")
                import traceback
                traceback.print_exc()

    if not results:
        print("[ERR] no successful rollouts")
        return 1

    # --- Save per-config CSVs ---
    metric_rows = []
    for (name, mode), r in results.items():
        eq = r["equity"]
        eq.name = f"equity_{name}_{mode}"
        eq.to_csv(OUT_DIR / f"equity_{name}_{mode}.csv", header=True)
        m = _metrics(eq)
        metric_rows.append({
            "agent": name.upper(), "cost_model": mode,
            "final_$": m["final"], "return_%": m["ret_pct"],
            "sharpe": m["sharpe"], "max_dd_%": m["mdd"],
            "total_cost_$": r["total_cost"], "trades": r["trades"],
        })
        if "daily_cost" in r:
            df = pd.DataFrame({
                "day": np.arange(len(r["daily_cost"])),
                "cost_$": r["daily_cost"].values,
                "turnover_$": r["daily_turnover"].values,
                "pov": r["daily_pov"].values,
            })
            df.to_csv(OUT_DIR / f"daily_costs_{name}_{mode}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False,
                             float_format=lambda x: f"{x:,.3f}"))

    # --- DES + benchmark for the overlay chart ---
    print("\n=== Loading benchmarks ===")
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

    qqew = _yf("QQEW")   # equal-weighted NDX 100 ETF (Abbade benchmark)
    qqq  = _yf("QQQ")    # cap-weighted NDX 100 ETF
    ndx  = _yf("^NDX")   # NDX index

    des = None
    if DES_EQUITY.exists():
        des_df = pd.read_csv(DES_EQUITY, index_col=0, parse_dates=True)
        des = des_df["portfolio_equity"].loc[TRADE_START:TRADE_END]

    # --- Chart 1: baseline vs AC per agent -------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = {
        "a2c":  "#1F77B4",
        "ppo":  "#8CC63F",
        "ddpg": "#FF7F0E",
        "td3":  "#9467BD",
        "sac":  "#C44E52",
    }
    for (name, mode), r in results.items():
        ls = "-" if mode == "ac" else "--"
        ax.plot(r["equity"].index, r["equity"].values,
                label=f"{name.upper()} ({mode})",
                color=palette.get(name, "#333"),
                linestyle=ls, linewidth=1.4, alpha=0.85)
    for b, colour in ((qqew, "black"), (qqq, "#666666"), (ndx, "#999999")):
        if not b.empty:
            b2 = INITIAL_AMOUNT * b / b.iloc[0]
            ax.plot(b2.index, b2.values, label=b.name, color=colour,
                    linewidth=1.4, linestyle=":")
    ax.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"NDX 100 — 5 SB3 agents: Baseline 10 bps (dashed) vs. "
                 f"Almgren-Chriss (solid)   {TRADE_START.date()} ~ {TRADE_END.date()}")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio value ($, start $1M)")
    ax.legend(loc="upper left", fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_vs_ac.png", dpi=150); plt.close()
    print(f"[OK] wrote {OUT_DIR / 'baseline_vs_ac.png'}")

    # --- Chart 2: DES vs best AC agent vs QQEW ---------------------------
    fig, ax = plt.subplots(figsize=(13, 7))
    ac_rows = metrics[metrics["cost_model"] == "ac"]
    if not ac_rows.empty:
        best_ac_agent = ac_rows.sort_values("sharpe", ascending=False).iloc[0]["agent"].lower()
        best_ac = results[(best_ac_agent, "ac")]["equity"]
        ax.plot(best_ac.index, best_ac.values,
                label=f"{best_ac_agent.upper()} (AC — best OOS Sharpe)",
                color="#C44E52", linewidth=2.4)
    # Also show optimized-PPO/TD3 style: PPO under the baseline's best cell.
    if ("ppo", "baseline") in results:
        eq_ppo_b = results[("ppo", "baseline")]["equity"]
        ax.plot(eq_ppo_b.index, eq_ppo_b.values,
                label="PPO (baseline 10 bps)",
                color="#8CC63F", linewidth=1.6)
    if des is not None and not des.empty:
        des_r = INITIAL_AMOUNT * des / des.iloc[0]
        ax.plot(des_r.index, des_r.values, label="DES (ours, market-weighted)",
                color="#1F77B4", linewidth=2.4)
    for b, colour, name in ((qqew, "black", "QQEW"), (qqq, "#666666", "QQQ")):
        if not b.empty:
            b2 = INITIAL_AMOUNT * b / b.iloc[0]
            ax.plot(b2.index, b2.values, label=name, color=colour,
                    linewidth=1.4, linestyle=":")
    ax.axhline(INITIAL_AMOUNT, color="grey", linewidth=0.8,
               linestyle="--", label="Initial $1M")
    ax.set_title(f"NDX 100 — Abbade & Costa 2026 MACE vs Ours (DES)   "
                 f"{TRADE_START.date()} ~ {TRADE_END.date()}")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=10); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_vs_des.png", dpi=150); plt.close()
    print(f"[OK] wrote {OUT_DIR / 'baseline_vs_des.png'}")

    # --- Summary text ---
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Abbade & Costa 2026 MACE (Almgren-Chriss) — NDX 100 reproduction\n")
        f.write(f"Trade period : {TRADE_START.date()} ~ {TRADE_END.date()}\n")
        f.write(f"Universe     : {trade['tic'].nunique()} tickers (static, FinRL cache)\n")
        f.write(f"Agents       : {list(agents.keys())} (trained on 2015-2023 flat 10bps)\n")
        f.write(f"AC params    : α={AC_ALPHA}  β={AC_BETA}  ε={AC_SPREAD_BPS} bps  τ½={AC_HALF_LIFE} days\n")
        f.write(f"Initial cap  : ${INITIAL_AMOUNT:,}\n\n")
        f.write("Metrics (per agent × cost model):\n")
        f.write(metrics.to_string(index=False,
                                   float_format=lambda x: f"{x:,.3f}"))
        f.write("\n")
    print(f"[OK] wrote {OUT_DIR / 'summary.txt'}")
    print(f"[DONE] total elapsed {(time.time() - t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
