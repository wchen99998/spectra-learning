import torch
import torch.nn.functional as F


class ObjectiveMixin:
    def _embedding_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        return (prediction - target).square().mean(dim=-1)

    def _jepa_mae_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mz_target = torch.floor(
            peak_mz.float() * self.jepa_mae_mz_max / self.jepa_mae_mz_bin_size
        ).long()
        intensity_target = torch.floor(
            peak_intensity.float() / self.jepa_mae_intensity_bin_size
        ).long()
        return (
            mz_target.clamp(0, self.jepa_mae_num_mz_bins - 1),
            intensity_target.clamp(0, self.jepa_mae_num_intensity_bins - 1),
        )

    def _masked_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token = F.cross_entropy(
            logits.flatten(0, -2).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(valid_mask)
        weights = valid_mask.float()
        return (per_token * weights).sum() / weights.sum().clamp_min(1.0)

    def _jepa_mae_value_prediction_loss(
        self,
        predicted_latents: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mz_logits = self.jepa_mae_mz_head(predicted_latents)
        mz_target, intensity_target = self._jepa_mae_targets(peak_mz, peak_intensity)
        view_shape = (mz_logits.shape[0], mz_logits.shape[1], mz_logits.shape[2])
        mz_target = mz_target.unsqueeze(1).expand(view_shape)
        intensity_target = intensity_target.unsqueeze(1).expand(view_shape)
        mz_loss = self._masked_ce_loss(mz_logits, mz_target, target_masks)
        target_weights = target_masks
        mz_accuracy = (
            (mz_logits.argmax(dim=-1) == mz_target).float() * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        if self.masked_token_input_mode == "mz_sentinel":
            zero = mz_loss.new_zeros(())
            return mz_loss, mz_loss, zero, mz_accuracy, zero

        intensity_logits = self.jepa_mae_intensity_head(predicted_latents)
        intensity_loss = self._masked_ce_loss(
            intensity_logits,
            intensity_target,
            target_masks,
        )
        value_loss = mz_loss + intensity_loss
        intensity_accuracy = (
            (intensity_logits.argmax(dim=-1) == intensity_target).float()
            * target_weights.float()
        ).sum() / target_weights.float().sum().clamp_min(1.0)
        return value_loss, mz_loss, intensity_loss, mz_accuracy, intensity_accuracy

    def _predict_augmented_targets(
        self,
        context_emb: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_target_blocks, num_peaks = target_masks.shape
        context_mask_by_view = context_mask.unsqueeze(1)
        predictor_input = (
            context_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1)
            * context_mask_by_view.unsqueeze(-1)
        )
        if self.masked_token_input_mode == "mz_sentinel":
            predictor_input = torch.where(
                target_masks.unsqueeze(-1),
                context_emb.unsqueeze(1).expand(-1, num_target_blocks, -1, -1),
                predictor_input,
            )
        else:
            predictor_input = torch.where(
                target_masks.unsqueeze(-1),
                self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
                predictor_input,
            )
        predictor_visible_mask = (context_mask_by_view | target_masks).reshape(
            batch_size * num_target_blocks,
            num_peaks,
        )
        flat_predictor_input = predictor_input.reshape(
            batch_size * num_target_blocks,
            num_peaks,
            -1,
        )
        predictor_features = self.predict_masked_target_features(
            flat_predictor_input,
            predictor_visible_mask,
        )
        predictor_features = predictor_features.reshape(
            batch_size,
            num_target_blocks,
            num_peaks,
            -1,
        )
        predictor_output = self.project_targets(predictor_features)
        return predictor_features, predictor_output

    def _masked_prediction_loss(
        self,
        predictor_output: torch.Tensor,
        teacher_targets: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        per_token = self._embedding_loss(predictor_output, teacher_targets.unsqueeze(1))
        target_weights = target_masks.float()
        return (per_token * target_weights).sum() / target_weights.sum().clamp_min(1.0)

    def _jepa_mae_metrics(
        self,
        predictor_output: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        target_masks: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.jepa_mae_loss_weight <= 0:
            return reference.new_tensor(0.0), {}
        (
            value_loss,
            _mz_loss,
            _intensity_loss,
            _mz_accuracy,
            _intensity_accuracy,
        ) = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
        )
        loss_weight = reference.new_tensor(self.jepa_mae_loss_weight)
        term = loss_weight * value_loss.to(dtype=reference.dtype)
        return term, {
            "jepa_mae_loss": value_loss.to(dtype=reference.dtype),
            "jepa_mae_term": term,
        }

    def _mae_metrics(
        self,
        predictor_output: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        target_masks: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            value_loss,
            mz_loss,
            intensity_loss,
            mz_accuracy,
            intensity_accuracy,
        ) = self._jepa_mae_value_prediction_loss(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
        )
        loss_weight = reference.new_tensor(self.mae_loss_weight)
        term = loss_weight * value_loss.to(dtype=reference.dtype)
        return term, {
            "mae_loss": value_loss.to(dtype=reference.dtype),
            "mae_term": term,
            "mae_mz_loss": mz_loss.to(dtype=reference.dtype),
            "mae_intensity_loss": intensity_loss.to(dtype=reference.dtype),
            "mae_mz_accuracy": mz_accuracy.to(dtype=reference.dtype),
            "mae_intensity_accuracy": intensity_accuracy.to(dtype=reference.dtype),
        }

    def _sigreg_weights(self, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        if self.use_precursor_token:
            weights = weights.clone()
            weights[..., 0] *= self.sigreg_precursor_scale
        return weights

    def _regularizer_metrics(
        self,
        context_emb: torch.Tensor,
        context_mask: torch.Tensor,
        predictor_output_features: torch.Tensor,
        predictor_output: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.sigreg_lambda <= 0:
            return context_emb.new_tensor(0.0), {}

        target_weights = target_masks

        if self.representation_regularizer in (
            "sigreg-enc-pred",
            "slot-sigreg-enc-pred",
        ):
            encoder_loss = self.sigreg(
                context_emb.float(),
                valid_mask=self._sigreg_weights(context_mask),
            ).to(dtype=context_emb.dtype)
            predictor_loss = self.sigreg(
                predictor_output_features.float(),
                valid_mask=self._sigreg_weights(target_weights),
            ).to(dtype=context_emb.dtype)
            sigreg_loss = encoder_loss + predictor_loss
            sigreg_term = context_emb.new_tensor(self.sigreg_lambda) * sigreg_loss
            return sigreg_term, {
                "sigreg_loss": sigreg_loss,
                "sigreg_term": sigreg_term,
                "sigreg_encoder_loss": encoder_loss,
                "sigreg_predictor_loss": predictor_loss,
            }

        if self.representation_regularizer in ("sigreg-enc", "slot-sigreg-enc"):
            embeddings = context_emb.float()
            weights = self._sigreg_weights(context_mask)
        elif self.representation_regularizer in ("sigreg-pred", "slot-sigreg-pred"):
            embeddings = predictor_output_features.float()
            weights = self._sigreg_weights(target_weights)
        elif self.representation_regularizer in ("sigreg-proj", "slot-sigreg-proj"):
            embeddings = predictor_output.float()
            weights = self._sigreg_weights(target_weights)
        else:
            return context_emb.new_tensor(0.0), {}

        sigreg_loss = self.sigreg(embeddings, valid_mask=weights).to(
            dtype=context_emb.dtype
        )
        sigreg_term = context_emb.new_tensor(self.sigreg_lambda) * sigreg_loss
        return sigreg_term, {
            "sigreg_loss": sigreg_loss,
            "sigreg_term": sigreg_term,
        }

    def _covariance_pooling_metrics(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            not self.train_covariance_pooling
            or self.covariance_pooling_loss_weight <= 0
            or not hasattr(self, "covariance_pooler")
        ):
            return embeddings.new_tensor(0.0), {}

        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            x = embeddings.detach().float()
            mask = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask
            denom = mask.sum(dim=1).clamp_min(1.0)

            left = self.covariance_pooler.left_proj(x)
            right = self.covariance_pooler.right_proj(x)
            covariance = left.transpose(1, 2) @ right / denom.unsqueeze(-1)

            left_gram = (
                self.covariance_pooler.left_proj.weight.float()
                @ self.covariance_pooler.left_proj.weight.float().T
            )
            right_gram = (
                self.covariance_pooler.right_proj.weight.float()
                @ self.covariance_pooler.right_proj.weight.float().T
            )
            projected_reconstruction = (
                torch.einsum("ij,bjk->bik", left_gram, covariance)
            )
            projected_reconstruction = torch.einsum(
                "bij,jk->bik",
                projected_reconstruction,
                right_gram,
            )
            reconstruction_norm_sq = (
                covariance * projected_reconstruction
            ).sum(dim=(1, 2))
            cross_term = covariance.square().sum(dim=(1, 2))

            token_gram = x @ x.transpose(1, 2) / denom.unsqueeze(-1)
            target_norm_sq = token_gram.square().sum(dim=(1, 2))
            loss_sq = (
                reconstruction_norm_sq
                - 2.0 * cross_term
                + target_norm_sq
            ).clamp_min(0.0)
            covariance_loss = torch.sqrt(loss_sq + 1e-12).mean()

        term = embeddings.new_tensor(self.covariance_pooling_loss_weight) * (
            covariance_loss.to(dtype=embeddings.dtype)
        )
        return term, {
            "covariance_pooling_loss": covariance_loss.to(dtype=embeddings.dtype),
            "covariance_pooling_term": term,
        }

    def pool(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.shape[1] == valid_mask.shape[1] + 1:
            embeddings = embeddings[:, :-1]
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
