from ml_collections import config_dict

from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.models.settings import PeakSetSIGRegSettings


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    return PeakSetSIGReg(PeakSetSIGRegSettings.from_config(config))
