from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from spectra_learning.data import nist_mgf_embeddings
from spectra_learning.probes.massspec import msg_probe, msg_probe_jax
from spectra_learning.training import contrastive


def _torch_batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "peak_intensity": torch.tensor([[1.0, 0.5]], dtype=torch.float32),
        "peak_valid_mask": torch.tensor([[True, True]]),
        "precursor_mz": torch.tensor([0.5], dtype=torch.float32),
        "collision_energy": torch.tensor([0.35], dtype=torch.float32),
        "charge": torch.tensor([2.0], dtype=torch.float32),
    }


class _CapturingTorchEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrum_metadata = None

    def forward(
        self,
        peak_mz,
        peak_intensity,
        *,
        valid_mask,
        precursor_mz,
        spectrum_metadata,
    ):
        self.spectrum_metadata = spectrum_metadata
        return peak_mz.unsqueeze(-1)

    def forward_with_pair(
        self,
        peak_mz,
        peak_intensity,
        *,
        valid_mask,
        precursor_mz,
        spectrum_metadata,
    ):
        self.spectrum_metadata = spectrum_metadata
        return peak_mz.unsqueeze(-1), peak_mz[:, :, None, None]


@pytest.mark.parametrize("use_pair_features", [False, True])
def test_torch_msg_probe_passes_spectrum_metadata(use_pair_features: bool) -> None:
    encoder = _CapturingTorchEncoder()
    extractor = msg_probe._make_msg_probe_feature_extractor(
        SimpleNamespace(encoder=encoder),
        use_pair_features=use_pair_features,
    )

    extractor(_torch_batch())

    torch.testing.assert_close(
        encoder.spectrum_metadata,
        torch.tensor([[0.35, 2.0 / 21.0]], dtype=torch.float32),
    )


@pytest.mark.parametrize("use_pair_features", [False, True])
def test_jax_msg_probe_passes_spectrum_metadata(
    monkeypatch,
    use_pair_features: bool,
) -> None:
    captured = {}

    def extract(*args):
        captured["spectrum_metadata"] = args[-1]
        features = args[1][..., None]
        return (features, features[:, :, None]) if use_pair_features else features

    monkeypatch.setattr(
        msg_probe_jax,
        (
            "_extract_pair_features_jitted"
            if use_pair_features
            else "_extract_single_features_jitted"
        ),
        extract,
    )
    batch = {
        key: jnp.asarray(value.numpy())
        for key, value in _torch_batch().items()
    }

    msg_probe_jax._extract_features(
        object(),
        batch,
        use_pair_features=use_pair_features,
    )

    np.testing.assert_allclose(
        captured["spectrum_metadata"],
        np.asarray([[0.35, 2.0 / 21.0]], dtype=np.float32),
    )


def test_contrastive_encoder_receives_spectrum_metadata() -> None:
    encoder = _CapturingTorchEncoder()

    class Pooler:
        def __call__(self, peak_embeddings, valid_mask, pair_embeddings):
            return peak_embeddings[:, 0]

    module = SimpleNamespace(
        model=SimpleNamespace(encoder=encoder),
        pooler=Pooler(),
    )

    contrastive.ContrastiveTrainingModule._pooled_features(module, _torch_batch())

    torch.testing.assert_close(
        encoder.spectrum_metadata,
        torch.tensor([[0.35, 2.0 / 21.0]], dtype=torch.float32),
    )


def test_nist_mgf_embedding_parses_and_passes_spectrum_metadata() -> None:
    records = [
        {
            "peak_mz": np.asarray([100.0, 200.0], dtype=np.float32),
            "peak_intensity": np.asarray([1.0, 0.5], dtype=np.float32),
            "pepmass": "500.0",
            "collisionenergy": "35 eV",
            "charge": "2+",
        }
    ]
    batch = nist_mgf_embeddings._preprocess_mgf_batch(
        records,
        {
            "num_peaks": 2,
            "min_peak_intensity": 0.0,
            "peak_drop_min_intensity": 0.0,
            "peak_ordering": "mz",
            "max_precursor_mz": 1000.0,
        },
        torch.device("cpu"),
    )
    encoder = _CapturingTorchEncoder()

    nist_mgf_embeddings._encode_peak_tokens(
        SimpleNamespace(encoder=encoder),
        batch,
        torch.device("cpu"),
    )

    torch.testing.assert_close(batch["collision_energy"], torch.tensor([0.35]))
    torch.testing.assert_close(batch["charge"], torch.tensor([2.0]))
    torch.testing.assert_close(
        encoder.spectrum_metadata,
        torch.tensor([[0.35, 2.0 / 21.0]], dtype=torch.float32),
    )


def test_nist_mgf_embedding_uses_canonical_peak_count_default() -> None:
    records = [
        {
            "peak_mz": np.asarray([100.0, 200.0], dtype=np.float32),
            "peak_intensity": np.asarray([1.0, 0.5], dtype=np.float32),
            "pepmass": "500.0",
        }
    ]

    batch = nist_mgf_embeddings._preprocess_mgf_batch(
        records,
        {},
        torch.device("cpu"),
    )

    assert batch["peak_mz"].shape == (1, 60)


@pytest.mark.parametrize("pepmass", ("", "0.5", "1000.1"))
def test_nist_mgf_embedding_rejects_out_of_contract_precursor(
    pepmass: str,
) -> None:
    records = [
        {
            "peak_mz": np.asarray([100.0], dtype=np.float32),
            "peak_intensity": np.asarray([1.0], dtype=np.float32),
            "pepmass": pepmass,
        }
    ]

    with pytest.raises(ValueError, match="precursor m/z"):
        nist_mgf_embeddings._preprocess_mgf_batch(
            records,
            {},
            torch.device("cpu"),
        )
