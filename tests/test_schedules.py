import unittest

import torch
from ml_collections import config_dict

from spectra_learning.data.gems.mask_schedule import (
    jepa_mask_stage,
    jepa_mask_stage_index,
    jepa_mask_stages,
)
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

    def test_load_state_dict_rejects_legacy_eta_min(self):
        param = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([param], lr=3e-4)
        scheduler = make_cosine_schedule(
            optimizer,
            total_steps=100,
            warmup_steps=10,
            min_lr=1e-4,
        )
        legacy_state = {
            "total_steps": 100,
            "warmup_steps": 10,
            "base_lrs": [3e-4],
            "eta_min": 1e-4,
            "last_epoch": 5,
        }

        with self.assertRaisesRegex(KeyError, "eta_mins"):
            scheduler.load_state_dict(legacy_state)


class JepaMaskScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = config_dict.ConfigDict(
            {
                "jepa_context_fraction": 0.35,
                "jepa_target_fraction": 0.50,
                "jepa_context_fraction_schedule": (0.35, 0.55, 0.75),
                "jepa_target_fraction_schedule": (0.50, 0.30, 0.10),
                "jepa_mask_schedule_step_fractions": (1 / 3, 2 / 3),
                "gradient_accumulation_steps_schedule": (4, 8, 8),
            }
        )

    def test_three_stage_fractions(self):
        stages = jepa_mask_stages(self.config)

        self.assertEqual(
            [(stage.context_fraction, stage.target_fraction) for stage in stages],
            [(0.35, 0.50), (0.55, 0.30), (0.75, 0.10)],
        )
        self.assertEqual(
            [stage.gradient_accumulation_steps for stage in stages],
            [4, 8, 8],
        )

    def test_stage_boundaries_round_up(self):
        self.assertEqual(
            [
                jepa_mask_stage_index(self.config, step, 100)
                for step in (0, 33, 34, 66, 67, 99)
            ],
            [0, 0, 1, 1, 2, 2],
        )
        stage = jepa_mask_stage(self.config, 67, 100)
        self.assertEqual((stage.context_fraction, stage.target_fraction), (0.75, 0.10))

    def test_explicit_step_boundaries_take_precedence(self):
        self.config.jepa_mask_schedule_steps = (10, 80)

        self.assertEqual(
            [
                jepa_mask_stage_index(self.config, step, 100)
                for step in (0, 9, 10, 79, 80, 99)
            ],
            [0, 0, 1, 1, 2, 2],
        )


if __name__ == "__main__":
    unittest.main()
