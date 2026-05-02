from typing import Any

from torch import nn

from spectra_learning.models.ema import EMATeacherMixin
from spectra_learning.models.forwards import ForwardMixin
from spectra_learning.models.objectives import ObjectiveMixin
from spectra_learning.models.settings import PeakSetSIGRegSettings
from spectra_learning.models.setup import configure_peak_set_sigreg
from spectra_learning.models.targets import TargetProjectionMixin


class PeakSetSIGReg(
    EMATeacherMixin,
    TargetProjectionMixin,
    ObjectiveMixin,
    ForwardMixin,
    nn.Module,
):
    def __init__(
        self,
        settings: PeakSetSIGRegSettings | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        configure_peak_set_sigreg(
            self,
            PeakSetSIGRegSettings.create(settings, **overrides),
        )

    def forward(self, augmented_batch, return_collapse_data: bool = False):
        if return_collapse_data:
            return self.forward_augmented(
                augmented_batch,
                return_collapse_data=True,
            )
        return self.forward_augmented(augmented_batch)
