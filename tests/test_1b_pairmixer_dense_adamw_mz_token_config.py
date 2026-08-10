import math

from spectra_learning.config import load_config
from spectra_learning.data.spectra import PEAK_MZ_MAX
from spectra_learning.models.pairmixer import PairFeatureEmbedder


BASE_CONFIG = "configs/1b_pairmixer_dense_adamw.py"
TOKEN_CONFIG = "configs/1b_pairmixer_dense_adamw_mz_token.py"


def test_mz_token_config_changes_only_input_representation_and_name() -> None:
    base = load_config(BASE_CONFIG).to_dict()
    token = load_config(TOKEN_CONFIG).to_dict()
    changed = {
        "config_path",
        "encoder_mz_embedding",
        "encoder_mz_token_bin_size",
        "pairmixer_mz_embedding",
        "run_name_suffix",
    }

    assert {key for key in base if base[key] != token[key]} == changed
    assert token["encoder_mz_embedding"] == "token"
    assert token["pairmixer_mz_embedding"] == "token"
    assert math.ceil(PEAK_MZ_MAX / token["encoder_mz_token_bin_size"]) == 10_000
    assert math.ceil(PEAK_MZ_MAX / token["pairmixer_mz_token_bin_size"]) == 10_000
    assert token["jepa_mae_mz_bin_size"] == base["jepa_mae_mz_bin_size"] == 0.5


def test_pair_token_input_preserves_baseline_raw_projection_width() -> None:
    fourier = PairFeatureEmbedder(
        single_dim=4,
        pair_dim=8,
        hidden_dim=16,
        fourier_num_freqs=16,
    )
    token = PairFeatureEmbedder(
        single_dim=4,
        pair_dim=8,
        hidden_dim=16,
        mz_embedding="token",
        token_bin_size=0.1,
        token_embedding_dim=128,
    )

    assert fourier.raw_proj[0].in_features == token.raw_proj[0].in_features == 142
    assert token.mz_features.num_tokens == 10_000
