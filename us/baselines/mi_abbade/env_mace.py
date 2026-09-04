"""MACE (Market-Adjusted Cost Execution) stock trading environment.

Reimplementation of Abbade & Costa (2026) "Realistic Market Impact Modeling
for Reinforcement Learning Trading Environments" (arXiv 2603.29086v2) — the
MACE stock-trading environment, adapted to run inside our existing FinRL
StockTradingEnv trained-agent pipeline.

We reuse the pre-trained SB3 agents from `ndx100_variantA_trained_models/`
(trained with the FinRL flat 10 bps baseline) and re-backtest them in a
MACE environment that swaps the flat 10 bps cost for the Almgren-Chriss
non-linear model with permanent-impact exponential decay.

Almgren-Chriss cost decomposition for a trade of `x` shares at price `P`:

    C_perm   = 0.5 * α * σ * (|x|/V) * |x| * P    (permanent, reflects info)
    C_spread = ε * |x| * P                        (half-spread, e.g. 5 bps)
    C_temp   = β * σ * (|x|/V) * |x| * P          (temporary depth cost)

The permanent price shift on the mid-quote is:
    ΔP = α * σ * (x/V) * P     (signed by trade direction)
which then decays each day as ΔP_t = ΔP_{t-1} * (1 - λ) with λ = 1 - 2^(-1/τ½).

Default parameters follow the reference method:
    α = 0.5   (permanent impact prefactor)
    β = 1.0   (temporary impact prefactor)
    ε = 5e-4  (5 bps half-spread)
    τ½ = 5 trading days (large-cap decay half-life)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Make sure our FinRL fork is importable.
_FINRL_ROOT = Path(r"d:\US_stock\FinRL")
if str(_FINRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINRL_ROOT))

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class MACEStockTradingEnv(StockTradingEnv):
    """FinRL StockTradingEnv with Almgren-Chriss non-linear impact costs.

    Additional parameters
    ---------------------
    adv          : dict[str, float] or ndarray[stock_dim]
                   average-daily-volume (shares) per ticker, index-aligned to
                   the same tic order as `df`.  Constant across time (fine for
                   our short 2024–2026 test window).
    sigma        : dict[str, float] or ndarray[stock_dim]
                   daily return volatility per ticker.
    alpha_ac     : permanent-impact prefactor (default 0.5)
    beta_ac      : temporary-impact prefactor (default 1.0)
    spread_bps   : half-spread in bps (default 5.0 → 0.0005)
    perm_half_life : permanent-impact decay half-life in days (default 5)
    log_costs    : if True, keep per-step lists of costs / turnover / POV
                   available on the env instance for post-hoc inspection.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *args,
        adv: Sequence[float] | dict | None = None,
        sigma: Sequence[float] | dict | None = None,
        alpha_ac: float = 0.5,
        beta_ac: float = 1.0,
        spread_bps: float = 5.0,
        perm_half_life: float = 5.0,
        log_costs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Sanity: infer ticker order from the underlying DataFrame's first
        # trading day (StockTradingEnv relies on stable ordering across days).
        first_day = self.df.loc[0, :]
        if isinstance(first_day, pd.Series):
            first_day = first_day.to_frame().T
        self._tics: list[str] = list(first_day["tic"].values)
        assert len(self._tics) == self.stock_dim, (
            f"stock_dim mismatch: len(self._tics)={len(self._tics)}, "
            f"self.stock_dim={self.stock_dim}"
        )

        def _as_arr(x):
            if x is None:
                raise ValueError("adv and sigma must be supplied for MACE env")
            if isinstance(x, dict):
                return np.array([float(x[t]) for t in self._tics], dtype=float)
            return np.asarray(x, dtype=float).reshape(-1)

        self._adv = _as_arr(adv)
        self._sigma = _as_arr(sigma)
        assert len(self._adv) == self.stock_dim
        assert len(self._sigma) == self.stock_dim

        self._alpha_ac = float(alpha_ac)
        self._beta_ac = float(beta_ac)
        self._spread = float(spread_bps) / 10_000.0
        self._perm_lambda = 1.0 - 2.0 ** (-1.0 / float(perm_half_life))

        # Permanent price shift per ticker (in $/share).  Persists day-to-day
        # and decays exponentially; applied on top of `state[index+1]` for
        # the purpose of cost accounting but does NOT overwrite historical
        # prices used in valuation.
        self._perm_shift = np.zeros(self.stock_dim, dtype=float)

        # Logging
        self._log_costs = log_costs
        self.mace_daily_cost: list[float] = []
        self.mace_daily_turnover: list[float] = []
        self.mace_daily_pov: list[float] = []
        # Buffers accumulated within a single day
        self._day_cost = 0.0
        self._day_turnover_notional = 0.0
        self._day_pov_num = 0.0
        self._day_pov_den = 0.0

    # ------------------------------------------------------------------ #
    #  Cost engine                                                       #
    # ------------------------------------------------------------------ #
    def _ac_costs(self, index: int, shares: float, price: float
                  ) -> tuple[float, float]:
        """Return (total_cost, signed_perm_shift) for a trade of `shares`
        at reference price `price`.

        `shares` is signed: positive for buy, negative for sell.
        """
        if shares == 0 or price <= 0:
            return 0.0, 0.0
        abs_x = abs(float(shares))
        adv = max(float(self._adv[index]), 1.0)
        sig = max(float(self._sigma[index]), 1e-6)
        pov = abs_x / adv  # participation of daily volume (fraction)

        c_spread = self._spread * abs_x * price
        c_perm   = 0.5 * self._alpha_ac * sig * pov * abs_x * price
        c_temp   =        self._beta_ac  * sig * pov * abs_x * price
        total    = c_spread + c_perm + c_temp

        # Signed permanent price shift ($ / share)
        perm_shift = np.sign(shares) * self._alpha_ac * sig * pov * price

        # Accumulate per-day metrics
        self._day_cost += total
        self._day_turnover_notional += abs_x * price
        self._day_pov_num += pov * abs_x
        self._day_pov_den += abs_x
        return total, perm_shift

    # ------------------------------------------------------------------ #
    #  Overrides of the trade actions                                    #
    # ------------------------------------------------------------------ #
    def _sell_stock(self, index, action):
        """Sell — same logic as parent, but cost/permanent impact via AC."""
        # Respect the parent's turbulence-flag path
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            # Turbulence liquidation — fall back to parent flat-cost logic
            return super()._sell_stock(index, action)

        # tradable flag column
        if self.state[index + 2 * self.stock_dim + 1] == True:  # noqa: E712
            return 0

        price = self.state[index + 1]
        if price <= 0:
            return 0

        held = self.state[index + self.stock_dim + 1]
        if held <= 0:
            return 0

        sell_num_shares = float(min(abs(action), held))
        signed_shares = -sell_num_shares  # negative for sell

        cost, perm_shift = self._ac_costs(index, signed_shares, price)
        # Sell proceeds = price * shares - cost.  Impact reduces realized price.
        proceeds = price * sell_num_shares - cost
        self.state[0] += proceeds
        self.state[index + self.stock_dim + 1] -= sell_num_shares
        self.cost += cost
        self.trades += 1
        # Sells push mid down (perm_shift is negative)
        self._perm_shift[index] += perm_shift
        return sell_num_shares

    def _buy_stock(self, index, action):
        """Buy — same logic as parent, but cost/permanent impact via AC."""
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            return 0

        if self.state[index + 2 * self.stock_dim + 1] == True:  # noqa: E712
            return 0

        price = self.state[index + 1]
        if price <= 0:
            return 0

        # Iteratively find how many shares we can afford *including* AC costs.
        # AC cost is quadratic in shares, so a linear approximation is fine;
        # then we clip.
        cash = self.state[0]
        if cash <= 0 or action <= 0:
            return 0

        # First pass: ignore AC to bound `buy_num_shares` by cash.
        max_shares = cash / (price * (1.0 + self._spread))
        buy_num_shares = float(min(max_shares, action))
        if buy_num_shares <= 0:
            return 0

        # AC cost with this size; if it eats into cash, scale back.
        cost, perm_shift = self._ac_costs(index, +buy_num_shares, price)
        outlay = price * buy_num_shares + cost
        # Rare guard — scale down if we overshot cash.
        if outlay > cash:
            scale = cash / outlay
            buy_num_shares *= scale
            # Reset per-day accumulators for this trade and recompute
            self._day_cost -= cost
            self._day_turnover_notional -= abs(buy_num_shares / scale) * price
            self._day_pov_num -= (abs(buy_num_shares / scale)
                                  * (abs(buy_num_shares / scale) / max(self._adv[index], 1.0)))
            self._day_pov_den -= abs(buy_num_shares / scale)
            cost, perm_shift = self._ac_costs(index, +buy_num_shares, price)
            outlay = price * buy_num_shares + cost

        self.state[0] -= outlay
        self.state[index + self.stock_dim + 1] += buy_num_shares
        self.cost += cost
        self.trades += 1
        # Buys push mid up
        self._perm_shift[index] += perm_shift
        return buy_num_shares

    # ------------------------------------------------------------------ #
    #  Daily decay hook                                                  #
    # ------------------------------------------------------------------ #
    def step(self, actions):
        # Flush yesterday's per-day accumulator into log buffers at the
        # start of each step (i.e. per completed day).
        if self._log_costs and (self._day_pov_den > 0 or self._day_turnover_notional > 0):
            self.mace_daily_cost.append(self._day_cost)
            self.mace_daily_turnover.append(self._day_turnover_notional)
            self.mace_daily_pov.append(
                self._day_pov_num / self._day_pov_den if self._day_pov_den > 0 else 0.0
            )
        self._day_cost = 0.0
        self._day_turnover_notional = 0.0
        self._day_pov_num = 0.0
        self._day_pov_den = 0.0

        # Permanent-impact decay applied once per day
        self._perm_shift *= (1.0 - self._perm_lambda)

        return super().step(actions)

    def reset(self, *args, **kwargs):
        self._perm_shift[:] = 0.0
        self._day_cost = 0.0
        self._day_turnover_notional = 0.0
        self._day_pov_num = 0.0
        self._day_pov_den = 0.0
        self.mace_daily_cost.clear()
        self.mace_daily_turnover.clear()
        self.mace_daily_pov.clear()
        return super().reset(*args, **kwargs)
