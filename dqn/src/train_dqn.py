"""DQN trainer for the public TW50 walk-forward benchmark.

Configuration is intentionally narrow compared to the internal
``d:/DRL/train_model.py``:

* Features       : ``<DES> + <OHLC>``  (no volume, no sentiment)
* State encoder  : 1-D CNN (``DQNConv1DLarge``) with ``bars_count=10``
* Reward         : realised P&L (no Sharpe shaping, no idle penalty)
* Commissions    : buy 0.1425 % / sell 0.4425 %
* Learning       : Double DQN with prioritised replay (PER) + n-step return
* Validation     : 5-fold contiguous walk-forward (see ``src/walk_forward.py``)
* Test           : 2024-01-02 ~ 2026-03-30 (see ``src/backtest.py``)

Usage:
    python src/train_dqn.py --symbol 2330 --fold 0 --hours 1.5
    python src/train_dqn.py --symbol 2330 --fold all --hours 1.5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib import common, data, environ, models, validation  # noqa: E402
from src.walk_forward import build_all_folds, load_prefiltered, split_folds, write_fold  # noqa: E402

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4


def default_cfg() -> dict:
    return {
        # data (filled in per-fold)
        "train_csv": None,
        "val_csv": None,
        # DQN hyper-parameters
        "gamma": 0.99,
        "lr": 1e-4,
        "batch_size": 128,
        "bars_count": 10,
        "replay_size": 30000,
        "replay_initial": 1000,
        "reward_steps": 2,
        "target_net_sync": 5000,
        "epsilon_start": 1.0,
        "epsilon_final": 0.05,
        "epsilon_steps": 100_000,
        "epsilon_ratio": 0.8,
        "beta_ratio": 0.9,
        # environment
        "commission_buy": 0.1425,
        "commission_sell": 0.4425,
        "reset_on_close": False,
        "reward_on_close": False,
        "state_1d": True,
        "volumes": False,
        "time_limit": 200,
        # runtime
        "n_envs": 4,
        "cuda": True,
        "train_hours": 1.5,
        "eval_every_step": 1000,
        "validation_every_step": 1000,
        "checkpoint_every_step": 30_000,
        "states_to_evaluate": 1000,
        "adaptive_fps_measure_delay": 2500,
        # output
        "saves_path": None,
        "run_name": "tw_dqn",
        "seed": None,
    }


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size: int, prob_alpha: float = 0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer: list = []
        self.priorities = np.zeros((buf_size,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)

    def populate(self, count: int) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[: self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities) -> None:
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def calc_loss(batch, batch_weights, net, tgt_net, gamma: float, device: str = "cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones.astype(bool)).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_actions = net(next_states_v).argmax(dim=1, keepdim=True)
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected = next_state_values * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected) ** 2
    return losses_v.mean(), losses_v + 1e-5


def _make_env(prices_dict, cfg: dict, *, for_train: bool):
    e = environ.StocksEnv(
        prices_dict,
        bars_count=cfg["bars_count"],
        commission=float(cfg["commission_buy"]),
        commission_buy=float(cfg["commission_buy"]),
        commission_sell=float(cfg["commission_sell"]),
        reset_on_close=cfg["reset_on_close"],
        reward_on_close=cfg["reward_on_close"],
        state_1d=cfg["state_1d"],
        volumes=cfg["volumes"],
    )
    if for_train:
        return gym.wrappers.TimeLimit(e, max_episode_steps=cfg["time_limit"])
    return e


def train_one_fold(cfg: dict) -> dict:
    if not cfg.get("train_csv") or not cfg.get("val_csv"):
        raise ValueError("cfg must include train_csv and val_csv")

    if cfg.get("seed") is not None:
        configure_seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg.get("cuda", True) and torch.cuda.is_available() else "cpu")
    saves_path = Path(cfg["saves_path"])
    saves_path.mkdir(parents=True, exist_ok=True)
    print(f"[train] saves_path={saves_path}  device={device}")

    train_prices = {"TW": data.load_relative(str(cfg["train_csv"]))}
    val_prices = {"TW": data.load_relative(str(cfg["val_csv"]))}

    n_envs = max(1, int(cfg.get("n_envs", 1)))
    if n_envs > 1:
        train_env = gym.vector.SyncVectorEnv([
            (lambda: _make_env(train_prices, cfg, for_train=True)) for _ in range(n_envs)
        ])
        obs_shape = train_env.single_observation_space.shape
        n_actions = int(train_env.single_action_space.n)
    else:
        train_env = _make_env(train_prices, cfg, for_train=True)
        obs_shape = train_env.observation_space.shape
        n_actions = int(train_env.action_space.n)

    env_val = _make_env(val_prices, cfg, for_train=False)

    net = models.DQNConv1DLarge(obs_shape, n_actions).to(device)
    print(net)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=cfg["epsilon_start"])
    epsilon_tracker = common.EpsilonTracker(selector, dict(
        epsilon_start=cfg["epsilon_start"],
        epsilon_final=cfg["epsilon_final"],
        epsilon_frames=cfg["epsilon_steps"],
    ))
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    if n_envs > 1:
        exp_source = ptan.experience.VectorExperienceSourceFirstLast(
            train_env, agent, cfg["gamma"], steps_count=cfg["reward_steps"])
    else:
        exp_source = ptan.experience.ExperienceSourceFirstLast(
            train_env, agent, cfg["gamma"], steps_count=cfg["reward_steps"])

    buffer = PrioReplayBuffer(exp_source, cfg["replay_size"], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    writer = SummaryWriter(logdir=str(saves_path / "tb"), comment=f"-{cfg['run_name']}")

    frame_idx = 0
    beta = BETA_START
    beta_frames = cfg["epsilon_steps"]
    eval_states = None
    best_val_reward = None
    best_val_checkpoint = None
    train_deadline = (time.time() + float(cfg["train_hours"]) * 3600) if cfg.get("train_hours") else None
    fps_measure_start_time = None
    fps_measure_start_frame = None
    adaptive_applied = False

    try:
        with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                epsilon_tracker.frame(frame_idx)
                beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / beta_frames)

                new_rewards = exp_source.pop_rewards_steps()
                if new_rewards:
                    writer.add_scalar("beta", beta, frame_idx)
                    reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon)

                if len(buffer) < cfg["replay_initial"]:
                    continue

                if eval_states is None:
                    print("Initial buffer populated, start training")
                    eval_batch, _, __ = buffer.sample(cfg["states_to_evaluate"], beta)
                    eval_states = np.asarray([np.array(t.state, copy=False) for t in eval_batch])
                    if cfg.get("train_hours"):
                        fps_measure_start_time = time.time()
                        fps_measure_start_frame = frame_idx

                if (cfg.get("train_hours") and fps_measure_start_time is not None
                        and not adaptive_applied
                        and frame_idx >= fps_measure_start_frame + int(cfg["adaptive_fps_measure_delay"])):
                    elapsed = max(time.time() - fps_measure_start_time, 1e-6)
                    span = frame_idx - fps_measure_start_frame
                    fps = span / elapsed
                    remaining = max(train_deadline - time.time(), 0.0)
                    total = frame_idx + int(fps * remaining)
                    total = max(total, frame_idx + 100)
                    epsilon_tracker.epsilon_frames = max(int(total * cfg["epsilon_ratio"]), frame_idx + 1)
                    beta_frames = max(int(total * cfg["beta_ratio"]), frame_idx + 1)
                    adaptive_applied = True
                    print(f"Adaptive schedule: fps={fps:.2f}  est_total={total:,}  "
                          f"eps_steps={epsilon_tracker.epsilon_frames:,}  beta_frames={beta_frames:,}")

                if frame_idx % cfg["eval_every_step"] == 0:
                    mean_val = common.calc_values_of_states(eval_states, net, device=device)
                    writer.add_scalar("values_mean", mean_val, frame_idx)

                optimizer.zero_grad()
                batch, batch_indices, batch_weights = buffer.sample(cfg["batch_size"], beta)
                loss_v, sample_prios_v = calc_loss(
                    batch, batch_weights, net, tgt_net.target_model, cfg["gamma"], device=device)
                loss_v.backward()
                optimizer.step()
                buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

                if frame_idx % cfg["target_net_sync"] == 0:
                    tgt_net.sync()

                if frame_idx % cfg["checkpoint_every_step"] == 0:
                    torch.save(net.state_dict(),
                               saves_path / f"checkpoint-{frame_idx // cfg['checkpoint_every_step']}.data")

                if frame_idx % cfg["validation_every_step"] == 0:
                    res_val = validation.validation_run(
                        env_val, net, device=device,
                        commission_buy=cfg["commission_buy"],
                        commission_sell=cfg["commission_sell"])
                    for k, v in res_val.items():
                        writer.add_scalar(k + "_val", v, frame_idx)
                    val_reward = float(res_val["episode_reward"])
                    if best_val_reward is None or val_reward > best_val_reward:
                        if best_val_reward is not None:
                            print(f"{frame_idx}: Best val reward {best_val_reward:.3f} -> {val_reward:.3f}")
                        best_val_reward = val_reward
                        best_val_checkpoint = saves_path / f"best_val-{val_reward:.3f}.data"
                        torch.save(net.state_dict(), best_val_checkpoint)

                if train_deadline and time.time() >= train_deadline:
                    print(f"Reached train_hours={cfg['train_hours']}h at frame {frame_idx:,}")
                    break
    finally:
        writer.close()
        if n_envs > 1:
            try:
                train_env.close()
            except Exception:  # noqa: BLE001
                pass

        return {"best_val_reward": best_val_reward,
            "best_val_checkpoint": (str(best_val_checkpoint)
                        if best_val_checkpoint is not None else None),
            "final_frame": frame_idx, "saves_path": str(saves_path)}


def run_walk_forward(symbol: str, cfg: dict, folds: list[int] | None = None) -> list[dict]:
    data_dir = Path(cfg.pop("data_dir", REPO_ROOT / "data"))
    candidate_manifest = cfg.pop("candidate_manifest", None)
    csv_path = data_dir / f"{symbol}_all.csv"
    if not csv_path.is_file():
        raise SystemExit(f"missing data: {csv_path}")

    saves_root = Path(cfg.pop("saves_root", REPO_ROOT / "saves")) / f"{symbol}_all"
    if cfg.get("seed") is not None:
        saves_root /= f"seed_{int(cfg['seed'])}"
    saves_root.mkdir(parents=True, exist_ok=True)
    print(f"walk-forward: symbol={symbol}  saves_root={saves_root}")

    df = load_prefiltered(csv_path)
    all_folds = split_folds(df)
    fold_ids = folds if folds is not None else list(range(len(all_folds)))
    print(f"pre-test rows={len(df)}  ({df['<DATE>'].min().date()} ~ {df['<DATE>'].max().date()})")

    results = []
    for k in fold_ids:
        train_df, val_df = all_folds[k]
        fold_dir = saves_root / f"fold_{k}"
        fp = write_fold(train_df, val_df, fold_dir, k)
        print(f"\n=== fold {k}: val {fp.val_dates[0].date()}~{fp.val_dates[1].date()} "
              f"({len(val_df)} rows), train {len(train_df)} rows ===")
        fold_cfg = dict(cfg)
        fold_cfg.update({
            "train_csv": fp.train_csv,
            "val_csv": fp.val_csv,
            "saves_path": fold_dir,
            "run_name": f"{symbol}_all_fold{k}",
        })
        result = train_one_fold(fold_cfg)
        result["fold"] = k
        results.append(result)

    if candidate_manifest is not None:
        if cfg.get("seed") is None:
            raise ValueError("--candidate-manifest requires --seed")
        valid = [result for result in results if result["best_val_reward"] is not None]
        if not valid:
            raise RuntimeError("no validation checkpoint was produced")
        best = max(valid, key=lambda item: (float(item["best_val_reward"]), -int(item["fold"])))
        checkpoint = Path(best["best_val_checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(f"validation checkpoint missing: {checkpoint}")
        candidate_manifest = Path(candidate_manifest)
        candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        relative_checkpoint = Path(os.path.relpath(checkpoint, candidate_manifest.parent)).as_posix()
        pd.DataFrame([{
            "stock_id": symbol,
            "seed": int(cfg["seed"]),
            "validation_return": float(best["best_val_reward"]),
            "checkpoint_path": relative_checkpoint,
        }]).to_csv(candidate_manifest, index=False, lineterminator="\n")
        print(f"candidate manifest: {candidate_manifest}")
    return results


def _build_argparser() -> argparse.ArgumentParser:
    cfg = default_cfg()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--fold", default="all", help="'all' or comma-separated indices, e.g. 0,2,4")
    p.add_argument("--hours", type=float, default=cfg["train_hours"], help="wall-clock budget per fold")
    p.add_argument("--n-envs", type=int, default=cfg["n_envs"])
    p.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    p.add_argument("--bars", type=int, default=cfg["bars_count"])
    p.add_argument("--lr", type=float, default=cfg["lr"])
    p.add_argument("--gamma", type=float, default=cfg["gamma"])
    p.add_argument("--reward-steps", type=int, default=cfg["reward_steps"])
    p.add_argument("--target-sync", type=int, default=cfg["target_net_sync"])
    p.add_argument("--commission-buy", type=float, default=cfg["commission_buy"])
    p.add_argument("--commission-sell", type=float, default=cfg["commission_sell"])
    p.add_argument("--time-limit", type=int, default=cfg["time_limit"])
    p.add_argument("--validation-every", type=int, default=cfg["validation_every_step"])
    p.add_argument("--checkpoint-every", type=int, default=cfg["checkpoint_every_step"])
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--saves-root", type=Path, default=REPO_ROOT / "saves")
    p.add_argument("--seed", type=int, default=None,
                   help="Reproducible agent seed; seeded runs use an isolated seed_<N> directory")
    p.add_argument("--candidate-manifest", type=Path, default=None,
                   help="Write this seed's best-fold validation checkpoint metadata")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default: use CUDA if available)")
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    cfg = default_cfg()
    cfg.update({
        "train_hours": args.hours,
        "n_envs": args.n_envs,
        "batch_size": args.batch_size,
        "bars_count": args.bars,
        "lr": args.lr,
        "gamma": args.gamma,
        "reward_steps": args.reward_steps,
        "target_net_sync": args.target_sync,
        "commission_buy": args.commission_buy,
        "commission_sell": args.commission_sell,
        "time_limit": args.time_limit,
        "validation_every_step": args.validation_every,
        "checkpoint_every_step": args.checkpoint_every,
        "cuda": not args.cpu,
        "data_dir": args.data_dir,
        "saves_root": args.saves_root,
        "seed": args.seed,
        "candidate_manifest": args.candidate_manifest,
    })
    if args.fold == "all":
        folds = None
    else:
        folds = [int(x) for x in args.fold.split(",")]
    run_walk_forward(args.symbol, cfg, folds=folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
