import math

import torch
from torch import nn


class EMATeacherMixin:
    def ema_teacher_momentum_at(
        self,
        step: int,
        total_steps: int,
    ) -> float:
        if self.ema_teacher_schedule == "constant":
            return self.ema_teacher_momentum_start
        progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
        if self.ema_teacher_schedule == "slow-fast-slow":
            peak = min(1.0, max(1e-6, self.ema_teacher_schedule_peak_fraction))
            if progress <= peak:
                phase = progress / peak
                eased = 0.5 - 0.5 * math.cos(math.pi * phase)
                return self.ema_teacher_momentum_start + eased * (
                    self.ema_teacher_momentum_mid - self.ema_teacher_momentum_start
                )
            phase = (progress - peak) / max(1e-6, 1.0 - peak)
            eased = 0.5 - 0.5 * math.cos(math.pi * phase)
            return self.ema_teacher_momentum_mid + eased * (
                self.ema_teacher_momentum_final - self.ema_teacher_momentum_mid
            )
        if self.ema_teacher_schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return self.ema_teacher_momentum_start + progress * (
            self.ema_teacher_momentum_final - self.ema_teacher_momentum_start
        )

    @torch.no_grad()
    def sync_ema_teacher(self) -> None:
        if self.teacher_encoder is not None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        if self.teacher_target_projector is not None:
            self.teacher_target_projector.load_state_dict(
                self.target_projector.state_dict()
            )

    @staticmethod
    @torch.no_grad()
    def _update_ema_module(
        teacher: nn.Module,
        student: nn.Module,
        momentum: float,
    ) -> None:
        teacher_params = list(teacher.parameters())
        if teacher_params:
            torch._foreach_lerp_(
                teacher_params,
                list(student.parameters()),
                1.0 - momentum,
            )
        teacher_float_buffers = []
        student_float_buffers = []
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(),
            student.buffers(),
        ):
            if torch.is_floating_point(teacher_buffer):
                teacher_float_buffers.append(teacher_buffer)
                student_float_buffers.append(student_buffer)
            else:
                teacher_buffer.copy_(student_buffer)
        if teacher_float_buffers:
            torch._foreach_lerp_(
                teacher_float_buffers,
                student_float_buffers,
                1.0 - momentum,
            )

    @torch.no_grad()
    def update_ema_teacher(
        self,
        step: int,
        total_steps: int,
    ) -> float | None:
        if not self.use_ema_teacher or self.teacher_encoder is None:
            return None
        momentum = self.ema_teacher_momentum_at(step, total_steps)
        self._update_ema_module(self.teacher_encoder, self.encoder, momentum)
        if self.teacher_target_projector is not None:
            self._update_ema_module(
                self.teacher_target_projector,
                self.target_projector,
                momentum,
            )
        return momentum
