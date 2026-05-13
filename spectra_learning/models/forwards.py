import torch

from spectra_learning.data.spectra import PRECURSOR_TOKEN_INTENSITY


class ForwardMixin:
    @staticmethod
    def prepend_precursor_token(
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B = peak_mz.shape[0]
        device = peak_mz.device
        pre_int = torch.full(
            (B, 1),
            PRECURSOR_TOKEN_INTENSITY,
            device=device,
            dtype=peak_mz.dtype,
        )
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([precursor_mz.unsqueeze(1), peak_mz], dim=1),
            "peak_intensity": torch.cat([pre_int, peak_intensity], dim=1),
            "peak_valid_mask": torch.cat([pre_valid, peak_valid_mask], dim=1),
        }
        if context_mask is not None:
            pre_ctx = torch.ones(B, 1, device=device, dtype=torch.bool)
            result["context_mask"] = torch.cat([pre_ctx, context_mask], dim=1)
        if target_masks is not None:
            K = target_masks.shape[1]
            pre_tgt = torch.zeros(B, K, 1, device=device, dtype=torch.bool)
            result["target_masks"] = torch.cat([pre_tgt, target_masks], dim=2)
        return result

    def _condition_precursor_masks(
        self,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
        peak_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_precursor_token:
            context_mask[:, 0] = peak_valid_mask[:, 0]
            target_masks[:, :, 0] = False
        return context_mask, target_masks

    @staticmethod
    def _target_mask_metrics(
        target_masks: torch.Tensor,
        valid_peak_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_entries = target_masks.float().sum()
        target_union = target_masks.any(dim=1).float().sum()
        target_overlap_entries = target_entries - target_union
        per_view_denominator = valid_peak_count * max(int(target_masks.shape[1]), 1)
        return {
            "target_fraction": target_entries / per_view_denominator,
            "target_fraction_per_view": target_entries / per_view_denominator,
            "target_union_fraction": target_union / valid_peak_count,
            "target_entry_fraction": target_entries / valid_peak_count,
            "target_overlap_entries": target_overlap_entries,
        }

    def _get_temporal_frame_inputs(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_mz = batch[f"{prefix}_peak_mz"]
        peak_intensity = batch[f"{prefix}_peak_intensity"]
        peak_valid_mask = batch[f"{prefix}_peak_valid_mask"]
        if not self.use_precursor_token:
            return peak_mz, peak_intensity, peak_valid_mask
        with_precursor = self.prepend_precursor_token(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            batch[f"{prefix}_precursor_mz"],
        )
        return (
            with_precursor["peak_mz"],
            with_precursor["peak_intensity"],
            with_precursor["peak_valid_mask"],
        )

    def forward_augmented(
        self,
        augmented_batch: dict[str, torch.Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.training_mode == "mae":
            return self.forward_mae(
                augmented_batch,
                return_collapse_data=return_collapse_data,
            )

        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        context_mask, target_masks = self._condition_precursor_masks(
            context_mask,
            target_masks,
            peak_valid_mask,
        )
        (
            teacher_target_features,
            teacher_peak_emb,
            teacher_cls_emb,
            context_emb,
        ) = self._encode_augmented_teacher_and_context(
            peak_mz,
            peak_intensity,
            peak_valid_mask,
            context_mask,
            target_masks,
            precursor_mz=precursor_mz,
        )
        predictor_output_features, predictor_output = self._predict_augmented_targets(
            context_emb,
            context_mask,
            target_masks,
        )
        teacher_target_features_normalized = self._apply_jepa_target_normalization(
            teacher_target_features.detach()
        )
        with torch.no_grad():
            teacher_targets = self.project_teacher_targets(
                teacher_target_features_normalized
            )

        masked_prediction_loss = self._masked_prediction_loss(
            predictor_output,
            teacher_targets,
            target_masks,
        )
        masked_prediction_term = self.masked_token_loss_weight * masked_prediction_loss
        jepa_mae_term, jepa_mae_metrics = self._jepa_mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_emb,
        )
        sigreg_term, sigreg_metrics = self._regularizer_metrics(
            context_emb,
            context_mask,
            predictor_output_features,
            predictor_output,
            target_masks,
        )
        covariance_term, covariance_metrics = self._covariance_pooling_metrics(
            context_emb,
            context_mask,
        )
        loss = (
            masked_prediction_term
            + jepa_mae_term
            + sigreg_term
            + covariance_term
        )
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        collapse_data: dict[str, torch.Tensor] = {}
        if return_collapse_data:
            pooled_mean = self.pool(teacher_peak_emb, peak_valid_mask)
            collapse_data = {
                "teacher_peak_emb": teacher_peak_emb.detach(),
                "teacher_cls_emb": teacher_cls_emb.detach(),
                "context_emb": context_emb.detach(),
                "context_mask": context_mask.detach(),
                "peak_valid_mask": peak_valid_mask.detach(),
                "target_masks": target_masks.detach(),
                "teacher_target_features": teacher_target_features.detach(),
                "teacher_target_features_normalized": (
                    teacher_target_features_normalized.detach()
                ),
                "teacher_targets": teacher_targets.detach(),
                "predictor_output_features": predictor_output_features.detach(),
                "predictor_output": predictor_output.detach(),
                "pooled_mean": pooled_mean.detach(),
            }
        metrics = {
            "loss": loss,
            "masked_prediction_loss": masked_prediction_loss,
            "masked_prediction_term": masked_prediction_term,
            "context_fraction": context_mask.float().sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(jepa_mae_metrics)
        metrics.update(sigreg_metrics)
        metrics.update(covariance_metrics)
        if return_collapse_data:
            return metrics, collapse_data
        return metrics

    def forward_mae(
        self,
        augmented_batch: dict[str, torch.Tensor],
        return_collapse_data: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        precursor_mz = augmented_batch.get("precursor_mz", None)
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        context_mask, target_masks = self._condition_precursor_masks(
            context_mask,
            target_masks,
            peak_valid_mask,
        )

        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )

        context_encoded = self.encoder(
            context_mz,
            context_intensity,
            valid_mask=peak_valid_mask,
            visible_mask=context_visible_mask,
            precursor_mz=precursor_mz,
        )
        context_emb, _ = self._split_encoder_output(
            self.encoder,
            context_encoded,
            peak_valid_mask,
        )
        predictor_output_features, predictor_output = self._predict_augmented_targets(
            context_emb,
            context_mask,
            target_masks,
        )
        mae_term, mae_metrics = self._mae_metrics(
            predictor_output,
            peak_mz,
            peak_intensity,
            target_masks,
            context_emb,
        )
        sigreg_term, sigreg_metrics = self._regularizer_metrics(
            context_emb,
            context_mask,
            predictor_output_features,
            predictor_output,
            target_masks,
        )
        covariance_term, covariance_metrics = self._covariance_pooling_metrics(
            context_emb,
            context_visible_mask,
        )
        loss = mae_term + sigreg_term + covariance_term
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        metrics = {
            "loss": loss,
            "context_fraction": context_mask.float().sum() / valid_peak_count,
        }
        metrics.update(self._target_mask_metrics(target_masks, valid_peak_count))
        metrics.update(mae_metrics)
        metrics.update(sigreg_metrics)
        metrics.update(covariance_metrics)
        if return_collapse_data:
            return metrics, {}
        return metrics

    def compute_next_frame_teacher_embeddings(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute teacher embeddings for the next frame."""
        next_frame_mz, next_frame_int, next_frame_valid = self._get_temporal_frame_inputs(
            batch,
            "next_frame",
        )
        teacher_encoder = (
            self.teacher_encoder
            if self.teacher_encoder is not None
            else self.encoder
        )
        if self.teacher_encoder is None:
            teacher_embeddings = teacher_encoder(
                next_frame_mz,
                next_frame_int,
                valid_mask=next_frame_valid,
                visible_mask=next_frame_valid,
                precursor_mz=batch.get("next_frame_precursor_mz", None),
            )
        else:
            with torch.no_grad():
                teacher_embeddings = teacher_encoder(
                    next_frame_mz,
                    next_frame_int,
                    valid_mask=next_frame_valid,
                    visible_mask=next_frame_valid,
                    precursor_mz=batch.get("next_frame_precursor_mz", None),
                )
        teacher_embeddings, _ = self._split_encoder_output(
            teacher_encoder,
            teacher_embeddings,
            next_frame_valid,
        )
        return teacher_embeddings

    def forward_temporal(
        self,
        batch: dict[str, torch.Tensor],
        teacher_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next-frame token embeddings from the full current frame."""
        if self.temporal_predictor_num_layers <= 0:
            raise ValueError(
                "forward_temporal requires temporal_predictor_num_layers > 0"
            )
        frame_mz, frame_int, frame_valid = self._get_temporal_frame_inputs(
            batch,
            "frame",
        )
        next_frame_mz, next_frame_int, next_frame_valid = self._get_temporal_frame_inputs(
            batch,
            "next_frame",
        )
        frame_rt = batch["frame_rt"]
        next_frame_rt = batch["next_frame_rt"]
        B = frame_mz.shape[0]

        frame_encoded = self.encoder(
            frame_mz,
            frame_int,
            valid_mask=frame_valid,
            visible_mask=frame_valid,
            precursor_mz=batch.get("frame_precursor_mz", None),
        )  # [B, N, D]
        frame_emb, _ = self._split_encoder_output(
            self.encoder,
            frame_encoded,
            frame_valid,
        )

        delta_rt = (next_frame_rt - frame_rt).unsqueeze(-1)  # [B, 1] in minutes
        rt_emb = self.temporal_rt_proj(delta_rt)  # [B, D]

        queries = self.temporal_query_token.view(1, 1, -1).expand(
            B, frame_emb.shape[1], -1
        )
        queries = queries + rt_emb.unsqueeze(1)
        queries = self._add_predictor_positions(queries)
        queries, _ = self._append_predictor_register_tokens(queries, None)

        for block in self.temporal_predictor:
            queries = block(queries, frame_emb, memory_mask=frame_valid)
        if self.predictor_num_register_tokens > 0:
            queries = queries[:, :-self.predictor_num_register_tokens]
        predicted_next_frame = queries  # [B, N, D]

        if teacher_embeddings is not None:
            next_frame_emb = teacher_embeddings
        else:
            next_frame_emb = self.compute_next_frame_teacher_embeddings(batch)

        per_token = self._embedding_loss(predicted_next_frame, next_frame_emb)

        next_frame_mask = next_frame_valid.float()
        loss = (per_token * next_frame_mask).sum() / next_frame_mask.sum().clamp_min(1.0)

        return {
            "loss": loss,
            "next_frame_pred_loss": loss.detach(),
        }

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
        )
        encoded = self.encoder(
            mz,
            intensity,
            valid_mask=valid,
            visible_mask=valid,
            precursor_mz=batch.get("precursor_mz", None),
        )
        peak_x, cls_x = self._split_encoder_output(self.encoder, encoded, valid)
        return cls_x
