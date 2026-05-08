import unittest

import torch

from spectra_learning.training.schedules import (
    learning_rate_at_step,
    make_cosine_schedule,
)


class WarmupCosineScheduleTests(unittest.TestCase):
    def test_schedule_sets_closed_form_lr_for_tensor_lr(self):
        param = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([param], lr=torch.tensor(3e-4))
        scheduler = make_cosine_schedule(
            optimizer,
            total_steps=17_454_400,
            warmup_steps=50_000,
            min_lr=1e-4,
        )

        scheduler.step(700_000)

        expected = learning_rate_at_step(
            700_000,
            base_lr=3e-4,
            total_steps=17_454_400,
            warmup_steps=50_000,
            min_learning_rate=1e-4,
        )
        self.assertAlmostEqual(float(optimizer.param_groups[0]["lr"]), expected)


if __name__ == "__main__":
    unittest.main()
