from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.train_dqn import calc_loss, default_cfg
from src import backtest


class TrainDqnTest(unittest.TestCase):
    def test_backtest_evaluator_hash_covers_sealed_costs(self):
        first = backtest.evaluator_hash(10, 0.1425, 0.4425, 0.0)
        second = backtest.evaluator_hash(10, 0.1425, 0.4425, 0.0)
        changed = backtest.evaluator_hash(10, 0.1425, 0.5, 0.0)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_paper_defaults(self):
        cfg = default_cfg()

        self.assertEqual(cfg["bars_count"], 10)
        self.assertEqual(cfg["gamma"], 0.99)
        self.assertEqual(cfg["target_net_sync"], 5000)
        self.assertEqual(cfg["epsilon_start"], 1.0)
        self.assertEqual(cfg["epsilon_final"], 0.05)
        self.assertEqual(cfg["epsilon_steps"], 100_000)
        self.assertEqual(cfg["commission_buy"], 0.1425)
        self.assertEqual(cfg["commission_sell"], 0.4425)

    def test_loss_uses_online_action_and_target_value(self):
        online = torch.nn.Linear(1, 2, bias=False)
        target = torch.nn.Linear(1, 2, bias=False)
        online.weight.data = torch.tensor([[2.0], [1.0]])
        target.weight.data = torch.tensor([[3.0], [9.0]])
        batch = [SimpleNamespace(
            state=np.array([0.0], dtype=np.float32),
            action=0,
            reward=1.0,
            last_state=np.array([1.0], dtype=np.float32),
        )]

        loss, priorities = calc_loss(
            batch, np.array([1.0], dtype=np.float32), online, target, gamma=0.99)

        expected_loss = (1.0 + 0.99 * 3.0) ** 2
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)
        self.assertAlmostEqual(priorities.item(), expected_loss + 1e-5, places=5)


if __name__ == "__main__":
    unittest.main()