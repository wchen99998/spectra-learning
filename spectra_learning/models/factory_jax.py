from __future__ import annotations

from ml_collections import config_dict

from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.settings import PeakSetJEPASettings


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetJEPAJax:
    return PeakSetJEPAJax(PeakSetJEPASettings.from_config(config))
