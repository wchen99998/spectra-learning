import math

import torch
import torch.nn.functional as F
from ml_collections import config_dict
from torch import Tensor, nn

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch


class FakePeakDiscriminator(nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super().__init__()
        self.encoder = build_model_from_config(config).encoder
        self.mz_bin_size = float(config.jepa_mae_mz_bin_size)
        self.num_mz_bins = math.ceil(PEAK_MZ_MAX / self.mz_bin_size)
        self.intensity_bin_size = float(config.jepa_mae_intensity_bin_size)
        self.num_intensity_bins = math.ceil(
            float(config.jepa_mae_intensity_max) / self.intensity_bin_size
        )
        self.detection_weight = float(config.fake_peak_detection_loss_weight)
        self.reconstruction_weight = float(
            config.fake_peak_reconstruction_loss_weight
        )
        self.intensity_weight = float(
            config.fake_peak_intensity_reconstruction_loss_weight
        )
        self.fake_head = nn.Linear(int(config.model_dim), 1)
        self.mz_head = nn.Linear(int(config.model_dim), self.num_mz_bins)
        self.intensity_head = nn.Linear(
            int(config.model_dim),
            self.num_intensity_bins,
        )
        for head in (self.fake_head, self.mz_head, self.intensity_head):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def predict(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        visible = batch.get("student_visible_mask", batch["peak_valid_mask"])
        visible = visible & batch["peak_valid_mask"]
        encoded = self.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
            visible_mask=visible,
            precursor_mz=batch.get("precursor_mz"),
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )[:, : batch["peak_mz"].shape[1]]
        return {
            "fake_logits": self.fake_head(encoded).squeeze(-1).float(),
            "mz_logits": self.mz_head(encoded).float(),
            "intensity_logits": self.intensity_head(encoded).float(),
            "visible_mask": visible,
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        predictions = self.predict(batch)
        fake_logits = predictions["fake_logits"]
        mz_logits = predictions["mz_logits"]
        intensity_logits = predictions["intensity_logits"]
        visible = predictions["visible_mask"].bool()
        fake = batch["fake_peak_mask"] & visible
        real = visible & ~fake
        weights = visible.float()

        detection_per_peak = F.binary_cross_entropy_with_logits(
            fake_logits,
            fake.float(),
            reduction="none",
        )
        detection_loss = (detection_per_peak * weights).sum() / weights.sum()
        true_mz_bins = torch.floor(
            batch["true_peak_mz"].float() * PEAK_MZ_MAX / self.mz_bin_size
        ).long().clamp(0, self.num_mz_bins - 1)
        mz_reconstruction_per_peak = F.cross_entropy(
            mz_logits.transpose(1, 2),
            true_mz_bins,
            reduction="none",
        )
        true_intensity_bins = torch.floor(
            batch["true_peak_intensity"].float() / self.intensity_bin_size
        ).long().clamp(0, self.num_intensity_bins - 1)
        intensity_reconstruction_per_peak = F.cross_entropy(
            intensity_logits.transpose(1, 2),
            true_intensity_bins,
            reduction="none",
        )
        fake_weights = fake.float()
        fake_count = fake_weights.sum().clamp_min(1.0)
        mz_reconstruction_loss = (
            mz_reconstruction_per_peak * fake_weights
        ).sum() / fake_count
        intensity_reconstruction_loss = (
            intensity_reconstruction_per_peak * fake_weights
        ).sum() / fake_count
        reconstruction_loss = (
            mz_reconstruction_loss
            + self.intensity_weight * intensity_reconstruction_loss
        )
        loss = (
            self.detection_weight * detection_loss
            + self.reconstruction_weight * reconstruction_loss
        )

        predicted_fake = fake_logits >= 0
        target = batch["target_masks"][:, 0] & batch["peak_valid_mask"]
        return {
            "loss": loss,
            "detection_loss": detection_loss,
            "reconstruction_loss": reconstruction_loss,
            "mz_reconstruction_loss": mz_reconstruction_loss,
            "intensity_reconstruction_loss": intensity_reconstruction_loss,
            "detection_accuracy": ((predicted_fake == fake) & visible).float().sum()
            / weights.sum(),
            "fake_recall": (predicted_fake & fake).float().sum() / fake_count,
            "real_specificity": ((~predicted_fake) & real).float().sum()
            / real.float().sum().clamp_min(1.0),
            "mz_reconstruction_accuracy": (
                (mz_logits.argmax(dim=-1) == true_mz_bins) & fake
            ).float().sum()
            / fake_count,
            "intensity_reconstruction_accuracy": (
                (intensity_logits.argmax(dim=-1) == true_intensity_bins) & fake
            ).float().sum()
            / fake_count,
            "fake_fraction": fake_weights.sum() / weights.sum(),
            "generator_exact_bin_accuracy": (
                target & batch["generator_exact_bin_mask"]
            ).float().sum()
            / target.float().sum().clamp_min(1.0),
        }
