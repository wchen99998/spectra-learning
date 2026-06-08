import math

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data.distributed import DistributedSampler

from spectra_learning.probes.massspec.targets import MACCS_FINGERPRINT_BITS
from spectra_learning.training.contrastive import (
    ContrastiveSplit,
    WeightedOnlineSampler,
    build_contrastive_loader,
)
from spectra_learning.training.distributed import DistributedContext


def _contrastive_split(num_compounds: int = 8) -> ContrastiveSplit:
    num_rows = num_compounds * 2
    smiles = np.asarray(
        [f"C{compound}H" for compound in range(num_compounds) for _ in range(2)]
    )
    collision_energy = np.asarray(
        [energy for _ in range(num_compounds) for energy in (10.0, 20.0)],
        dtype=np.float32,
    )
    return ContrastiveSplit(
        spectra=np.zeros((num_rows, 2, 8), dtype=np.float32),
        precursor_mz=np.arange(num_rows, dtype=np.float32) + 100.0,
        smiles=smiles,
        collision_energy=collision_energy,
        collision_energy_present=np.ones(num_rows, dtype=np.int32),
        probe_maccs=np.zeros((num_rows, MACCS_FINGERPRINT_BITS), dtype=np.int8),
    )


def _config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.num_peaks = 4
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0
    cfg.peak_drop_min_intensity = 0.0
    cfg.peak_ordering = "mz"
    cfg.precursor_peak_exclusion_window_da = 0.0
    cfg.dataloader_num_workers = 0
    cfg.dataloader_pin_memory = False
    return cfg


def test_weighted_online_sampler_uses_shared_epoch_stream_across_ranks():
    weights = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    world_size = 3
    seed = 123
    epoch = 2
    samplers = [
        WeightedOnlineSampler(
            weights,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            drop_last=False,
        )
        for rank in range(world_size)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)

    rank_indices = [list(sampler) for sampler in samplers]

    num_samples = math.ceil(len(weights) / world_size)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    expected_global = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples * world_size,
        replacement=True,
        generator=generator,
    ).tolist()
    assert rank_indices == [
        expected_global[rank::world_size] for rank in range(world_size)
    ]
    assert [len(indices) for indices in rank_indices] == [num_samples] * world_size


def test_contrastive_loader_distributed_sampler_shards_pair_indices_by_rank():
    split = _contrastive_split()
    world_size = 4
    global_pairs_per_batch = 8
    pairs_per_epoch = 16
    loaders = [
        build_contrastive_loader(
            _config(),
            split,
            split_name="train",
            pairs_per_epoch=pairs_per_epoch,
            global_pairs_per_batch=global_pairs_per_batch,
            seed=7,
            distributed=DistributedContext(
                rank=rank,
                local_rank=rank,
                world_size=world_size,
                device=torch.device("cpu"),
            ),
        )
        for rank in range(world_size)
    ]

    sampler_indices = []
    for loader in loaders:
        assert isinstance(loader.sampler, DistributedSampler)
        sampler_indices.append(list(loader.sampler))

    local_pairs_per_batch = global_pairs_per_batch // world_size
    assert [len(loader) for loader in loaders] == [2] * world_size
    for step in range(len(loaders[0])):
        step_indices = [
            idx
            for indices in sampler_indices
            for idx in indices[
                step * local_pairs_per_batch : (step + 1) * local_pairs_per_batch
            ]
        ]
        assert len(step_indices) == global_pairs_per_batch
        assert len(set(step_indices)) == global_pairs_per_batch
    assert sorted(idx for indices in sampler_indices for idx in indices) == list(
        range(pairs_per_epoch)
    )
