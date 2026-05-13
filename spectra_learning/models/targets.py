import torch

from spectra_learning.models.common import _active_autocast_context
from spectra_learning.models.encoder import PeakSetEncoder


class TargetProjectionMixin:
    def _apply_group_target_normalization(
        self,
        x: torch.Tensor,
        group_dim: int,
    ) -> torch.Tensor:
        if self.jepa_target_normalization == "none":
            return x
        orig_dtype = x.dtype
        x = x.float().reshape(*x.shape[:-1], -1, group_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = ((x - mean) / std).reshape(*x.shape[:-2], -1)
        return normalized.to(dtype=orig_dtype)

    def _apply_jepa_target_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_group_target_normalization(x, self.model_dim)

    def _predictor_slot_queries(self, memory: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(memory.shape[1], device=memory.device)
        queries = self.predictor_slot_embedding(positions)
        return queries.unsqueeze(0).expand(memory.shape[0], -1, -1).to(
            dtype=memory.dtype
        )

    def _append_predictor_register_queries(self, x: torch.Tensor) -> torch.Tensor:
        if self.predictor_register_tokens is None:
            return x
        registers = self.predictor_register_tokens.unsqueeze(0).expand(
            x.shape[0],
            -1,
            -1,
        )
        return torch.cat([x, registers.to(dtype=x.dtype)], dim=1)

    def predict_masked_latents(
        self,
        context_emb: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder_to_predictor_proj(context_emb)
        x = self._predictor_slot_queries(memory)
        x = self._append_predictor_register_queries(x)
        for block in self.masked_latent_predictor:
            x = block(x, memory, memory_mask=context_mask)
        x = self.predictor_final_norm(x)
        if self.predictor_num_register_tokens > 0:
            x = x[:, :-self.predictor_num_register_tokens]
        return x

    def project_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_projector(x)

    def project_teacher_targets(self, x: torch.Tensor) -> torch.Tensor:
        projector = (
            self.teacher_target_projector
            if self.teacher_target_projector is not None
            else self.target_projector
        )
        return projector(x)

    def predict_masked_target_features(
        self,
        context_emb: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.masked_latent_readout(
            self.predict_masked_latents(
                context_emb,
                context_mask,
            )
        )

    def predict_masked_targets(
        self,
        context_emb: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.project_targets(
            self.predict_masked_target_features(
                context_emb,
                context_mask,
            )
        )

    def _split_encoder_output(
        self,
        encoder: PeakSetEncoder,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak_embeddings, cls_embedding = encoder.split_peak_and_cls(embeddings)
        if not encoder.use_cls_token:
            cls_embedding = self.pool(peak_embeddings, valid_mask)
        return peak_embeddings, cls_embedding

    def _compute_jepa_teacher_target_features(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_peak_outputs = teacher_encoder.forward_peak_block_outputs(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=peak_valid_mask,
                block_indices=self.jepa_target_layers,
                precursor_mz=precursor_mz,
            )
            return torch.cat(teacher_peak_outputs, dim=-1)

    def _compute_jepa_teacher_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_target_features = self._compute_jepa_teacher_target_features(
                peak_mz,
                peak_intensity,
                peak_valid_mask,
                precursor_mz=precursor_mz,
            )
            return self.project_teacher_targets(
                self._apply_jepa_target_normalization(teacher_target_features)
            )

    def _context_encoder_inputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.masked_token_input_mode != "mz_sentinel":
            return peak_mz, peak_intensity, context_mask
        target_union = target_masks.any(dim=1)
        masked_mz = torch.where(
            target_union,
            torch.full_like(peak_mz, self.masked_mz_sentinel),
            peak_mz,
        )
        return masked_mz, peak_intensity, context_mask | target_union

    def compute_teacher_targets(
        self,
        augmented_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._compute_jepa_teacher_targets(
            augmented_batch["peak_mz"],
            augmented_batch["peak_intensity"],
            augmented_batch["peak_valid_mask"],
            precursor_mz=augmented_batch.get("precursor_mz", None),
        )

    def _encode_augmented_teacher_and_context(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = peak_mz.shape[0]
        context_mz, context_intensity, context_visible_mask = self._context_encoder_inputs(
            peak_mz,
            peak_intensity,
            context_mask,
            target_masks,
        )
        if self.teacher_encoder is not None:
            with torch.no_grad(), _active_autocast_context(peak_mz.device.type):
                teacher_encoded, teacher_peak_outputs = (
                    self.teacher_encoder.forward_with_block_outputs(
                        peak_mz,
                        peak_intensity,
                        valid_mask=peak_valid_mask,
                        visible_mask=peak_valid_mask,
                        block_indices=self.jepa_target_layers,
                        precursor_mz=precursor_mz,
                    )
                )
            context_encoded = self.encoder(
                context_mz,
                context_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=context_visible_mask,
                precursor_mz=precursor_mz,
            )
            teacher_target_features = torch.cat(teacher_peak_outputs, dim=-1)
            teacher_peak_emb, teacher_cls_emb = self._split_encoder_output(
                self.teacher_encoder,
                teacher_encoded,
                peak_valid_mask,
            )
            context_emb, _ = self._split_encoder_output(
                self.encoder,
                context_encoded,
                peak_valid_mask,
            )
            return (
                teacher_target_features,
                teacher_peak_emb,
                teacher_cls_emb,
                context_emb,
            )
        encoded, teacher_peak_outputs = self.encoder.forward_with_block_outputs(
            torch.cat([peak_mz, context_mz], dim=0),
            torch.cat([peak_intensity, context_intensity], dim=0),
            valid_mask=torch.cat([peak_valid_mask, peak_valid_mask], dim=0),
            visible_mask=torch.cat([peak_valid_mask, context_visible_mask], dim=0),
            block_indices=self.jepa_target_layers,
            precursor_mz=(
                None
                if precursor_mz is None
                else torch.cat([precursor_mz, precursor_mz], dim=0)
            ),
        )
        teacher_target_features = torch.cat(
            [peak_output[:batch_size] for peak_output in teacher_peak_outputs],
            dim=-1,
        )
        teacher_peak_emb, teacher_cls_emb = self._split_encoder_output(
            self.encoder,
            encoded[:batch_size],
            peak_valid_mask,
        )
        context_emb, _ = self._split_encoder_output(
            self.encoder,
            encoded[batch_size:],
            peak_valid_mask,
        )
        return teacher_target_features, teacher_peak_emb, teacher_cls_emb, context_emb

    def _compute_pooled_teacher_peak_targets(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visible_mask is None:
            visible_mask = peak_valid_mask
        with _active_autocast_context(peak_mz.device.type):
            teacher_encoder = (
                self.teacher_encoder
                if self.teacher_encoder is not None
                else self.encoder
            )
            teacher_encoded = teacher_encoder(
                peak_mz,
                peak_intensity,
                valid_mask=peak_valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
            )
            teacher_peak_emb, _ = self._split_encoder_output(
                teacher_encoder,
                teacher_encoded,
                peak_valid_mask,
            )
        return self.pool(teacher_peak_emb, visible_mask)
