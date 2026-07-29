import gymnasium as gym
import gymnasium.spaces
from gymnasium.utils import seeding

import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.15
DEFAULT_COMMISSION_BUY_PERC = 0.10
DEFAULT_COMMISSION_SELL_PERC = 0.30


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True,
                 use_sharpe=False, sharpe_eta=0.05, idle_penalty=0.0,
                 commission_buy_perc=None, commission_sell_perc=None,
                 sent=False):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.commission_buy_perc = float(
            commission_perc if commission_buy_perc is None else commission_buy_perc)
        self.commission_sell_perc = float(
            commission_perc if commission_sell_perc is None else commission_sell_perc)
        assert self.commission_buy_perc >= 0.0
        assert self.commission_sell_perc >= 0.0
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        # Optional sentiment channel (e.g. <SENT> column in CSV).
        self.sent = bool(sent)
        # Differential Sharpe Ratio (Moody & Saffell 1998) reward shaping
        self.use_sharpe = bool(use_sharpe)
        self.sharpe_eta = float(sharpe_eta)
        # Per-step penalty applied when agent is flat (no position).
        self.idle_penalty = float(idle_penalty)
    # The `offset` argument lets each training episode start at a different
    # timestep instead of always from the beginning of the series.
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)  # named tuple
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
        # EMAs of return (A) and squared return (B) used by Differential Sharpe Ratio
        self._sharpe_A = 0.0
        self._sharpe_B = 0.0

    @property
    def shape(self):
        # [d, h, l, c, (v,) (s,)] * bars + position_flag + rel_profit (since open)
        per_bar = 4
        if self.volumes:
            per_bar += 1
        if self.sent:
            per_bar += 1
        return (per_bar * self.bars_count + 1 + 1, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.DES[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
            if self.sent:
                res[shift] = self._prices.sent[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        # Cost components are kept separate from market P&L so DSR shaping
        # only transforms the return-driven signal.
        cost = 0.0
        pnl_pct = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            cost -= self.commission_buy_perc
        elif action == Actions.Close and self.have_position:
            cost -= self.commission_sell_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                pnl_pct += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            pnl_pct += 100.0 * (close - prev_close) / prev_close

        if self.use_sharpe:
            # DSR: reward each step is the contribution of the latest return
            # to the running estimate of Sharpe ratio.
            eta = self.sharpe_eta
            delta_a = pnl_pct - self._sharpe_A
            delta_b = pnl_pct ** 2 - self._sharpe_B
            var = self._sharpe_B - self._sharpe_A ** 2
            if var > 1e-6:
                shaped = (self._sharpe_B * delta_a - 0.5 * self._sharpe_A * delta_b) / (var ** 1.5)
            else:
                shaped = pnl_pct  # warm-up: not enough variance estimate yet
            self._sharpe_A += eta * delta_a
            self._sharpe_B += eta * delta_b
            reward = shaped + cost
        else:
            reward = pnl_pct + cost

        if not self.have_position and self.idle_penalty > 0.0:
            reward -= self.idle_penalty

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        # 4 base channels (DES + rel H/L/C) + optional volume + optional sent + 2 (position, rel_profit)
        ch = 4 + (1 if self.volumes else 0) + (1 if self.sent else 0) + 2
        return (ch, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.DES[self._offset - ofs:self._offset + 1]
        res[1] = self._prices.high[self._offset-ofs:self._offset+1]
        res[2] = self._prices.low[self._offset-ofs:self._offset+1]
        res[3] = self._prices.close[self._offset-ofs:self._offset+1]
        dst = 4
        if self.volumes:
            res[dst] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst += 1
        if self.sent:
            res[dst] = self._prices.sent[self._offset-ofs:self._offset+1]
            dst += 1
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=True,
                 use_sharpe=False, sharpe_eta=0.05, idle_penalty=0.0,
                 commission_buy=None, commission_sell=None, sent=False):
        assert isinstance(prices, dict)
        self._prices = prices
        state_kwargs = dict(reward_on_close=reward_on_close, volumes=volumes,
                            use_sharpe=use_sharpe, sharpe_eta=sharpe_eta,
                            idle_penalty=idle_penalty,
                            commission_buy_perc=commission_buy,
                            commission_sell_perc=commission_sell,
                            sent=sent)
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, **state_kwargs)
        else:
            self._state = State(bars_count, commission, reset_on_close, **state_kwargs)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.integers(prices.high.shape[0] - bars * 10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return self._state.encode(), info

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
