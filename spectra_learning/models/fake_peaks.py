import math

import torch
import torch.nn.functional as F
from ml_collections import config_dict
from torch import Tensor, nn

from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch

LOGIT_UNIFORM_EPS = 1e-6


def sample_logit_uniform_residual(shift: Tensor) -> Tensor:
    uniform = torch.rand_like(shift).clamp(
        LOGIT_UNIFORM_EPS,
        1.0 - LOGIT_UNIFORM_EPS,
    )
    return torch.sigmoid(torch.logit(uniform) + shift)


def logit_uniform_nll(shift: Tensor, target: Tensor) -> Tensor:
    target_logits = torch.logit(
        target.float().clamp(
            LOGIT_UNIFORM_EPS,
            1.0 - LOGIT_UNIFORM_EPS,
        )
    )
    inverse_logits = target_logits - shift.float()
    return -(
        F.logsigmoid(inverse_logits)
        + F.logsigmoid(-inverse_logits)
        - F.logsigmoid(target_logits)
        - F.logsigmoid(-target_logits)
    )


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
        detection = batch["detection_mask"] & visible
        fake = batch["fake_peak_mask"] & visible
        detection_fake = fake & detection
        detection_real = detection & ~detection_fake
        detection_weights = detection.float()

        detection_per_peak = F.binary_cross_entropy_with_logits(
            fake_logits,
            detection_fake.float(),
            reduction="none",
        )
        detection_count = detection_weights.sum()
        detection_loss = (
            detection_per_peak * detection_weights
        ).sum() / detection_count
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
        detection_fake_count = detection_fake.float().sum().clamp_min(1.0)
        detection_real_count = detection_real.float().sum().clamp_min(1.0)
        fake_recall = (
            predicted_fake & detection_fake
        ).float().sum() / detection_fake_count
        real_specificity = (
            (~predicted_fake) & detection_real
        ).float().sum() / detection_real_count
        fake_probability = fake_logits.sigmoid()
        return {
            "loss": loss,
            "detection_loss": detection_loss,
            "reconstruction_loss": reconstruction_loss,
            "mz_reconstruction_loss": mz_reconstruction_loss,
            "intensity_reconstruction_loss": intensity_reconstruction_loss,
            "detection_accuracy": (
                (predicted_fake == detection_fake) & detection
            ).float().sum()
            / detection_count,
            "detection_balanced_accuracy": 0.5
            * (fake_recall + real_specificity),
            "fake_recall": fake_recall,
            "real_specificity": real_specificity,
            "fake_fake_probability": (
                fake_probability * detection_fake.float()
            ).sum()
            / detection_fake_count,
            "real_fake_probability": (
                fake_probability * detection_real.float()
            ).sum()
            / detection_real_count,
            "mz_reconstruction_accuracy": (
                (mz_logits.argmax(dim=-1) == true_mz_bins) & fake
            ).float().sum()
            / fake_count,
            "intensity_reconstruction_accuracy": (
                (intensity_logits.argmax(dim=-1) == true_intensity_bins) & fake
            ).float().sum()
            / fake_count,
            "fake_fraction": detection_fake.float().sum() / detection_count,
            "generator_exact_bin_accuracy": (
                fake & batch["generator_exact_bin_mask"]
            ).float().sum()
            / fake_count,
        }


class DynamicPeakGenerator(nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super().__init__()
        self.backbone = build_model_from_config(config)
        self.mz_bin_size = float(config.jepa_mae_mz_bin_size)
        self.intensity_bin_size = float(config.jepa_mae_intensity_bin_size)
        self.temperature = float(config.generator_gumbel_temperature)
        self.mz_residual_head = nn.Linear(self.backbone.target_projector_dim, 1)
        self.intensity_residual_head = nn.Linear(
            self.backbone.target_projector_dim,
            1,
        )
        for head in (self.mz_residual_head, self.intensity_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        peak_valid_mask = batch["peak_valid_mask"]
        context_mask = batch["context_mask"] & peak_valid_mask
        target_masks = batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        context_emb = self.backbone.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=peak_valid_mask,
            visible_mask=context_mask,
            precursor_mz=batch.get("precursor_mz"),
            spectrum_metadata=torch_spectrum_metadata_from_batch(batch),
        )
        _, predictor_output = self.backbone._predict_augmented_target_outputs(
            context_emb,
            context_mask,
            target_masks,
        )
        mz_head = self.backbone.jepa_mae_mz_head
        intensity_head = self.backbone.jepa_mae_intensity_head
        assert mz_head is not None
        assert intensity_head is not None
        predictor_features = predictor_output[:, 0]
        mz_logits = mz_head(predictor_features).float()
        intensity_logits = intensity_head(predictor_features).float()
        mz_residual_shift = (
            self.mz_residual_head(predictor_features).squeeze(-1).float()
        )
        intensity_residual_shift = (
            self.intensity_residual_head(predictor_features).squeeze(-1).float()
        )
        mz_residual = sample_logit_uniform_residual(mz_residual_shift)
        intensity_residual = sample_logit_uniform_residual(
            intensity_residual_shift
        )
        sampled_mz_bins = F.gumbel_softmax(
            mz_logits,
            tau=self.temperature,
            hard=True,
        )
        sampled_intensity_bins = F.gumbel_softmax(
            intensity_logits,
            tau=self.temperature,
            hard=True,
        )
        mz_bins = torch.arange(
            mz_logits.shape[-1],
            device=mz_logits.device,
            dtype=mz_logits.dtype,
        )
        intensity_bins = torch.arange(
            intensity_logits.shape[-1],
            device=intensity_logits.device,
            dtype=intensity_logits.dtype,
        )
        predicted_mz = (
            (sampled_mz_bins * mz_bins).sum(dim=-1)
            + mz_residual
        ) * self.mz_bin_size / PEAK_MZ_MAX
        predicted_intensity = (
            (sampled_intensity_bins * intensity_bins).sum(dim=-1)
            + intensity_residual
        ) * self.intensity_bin_size
        target_mask = target_masks[:, 0]
        completed_intensity = torch.where(
            target_mask,
            predicted_intensity,
            batch["peak_intensity"],
        )
        completed_intensity = completed_intensity / torch.clamp(
            completed_intensity.amax(dim=1, keepdim=True),
            min=1e-8,
        )
        return {
            "peak_mz": torch.where(
                target_mask,
                predicted_mz,
                batch["peak_mz"],
            ),
            "peak_intensity": completed_intensity,
            "predicted_mz": predicted_mz,
            "predicted_intensity": completed_intensity,
            "mz_logits": mz_logits,
            "intensity_logits": intensity_logits,
            "mz_residual_shift": mz_residual_shift,
            "intensity_residual_shift": intensity_residual_shift,
            "mz_residual": mz_residual,
            "intensity_residual": intensity_residual,
            "target_mask": target_mask,
        }
