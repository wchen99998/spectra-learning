from ml_collections import config_dict

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.models.settings import PeakSetJEPASettings


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetJEPA:
    return PeakSetJEPA(PeakSetJEPASettings.from_config(config))
