"""Direct pair tokens for FLARE, with packed storage and exact candidate chunks."""

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


class RaggedPairTokens:
    def __init__(self, root: Path, values_path: Path | None = None):
        metadata = json.loads((root / "embedding_metadata.json").read_text())
        self.values = np.load(values_path if values_path is not None else root / "pair_values.float16.npy", mmap_mode="r")
        self.offsets = np.load(root / "pair_offsets.npy")
        self.shape = (metadata["samples"], metadata["maximum_pair_tokens"], metadata["pair_dim"])
        self.dtype = self.values.dtype
        assert self.values.shape == (int(self.offsets[-1]), self.shape[-1])
        assert len(self.offsets) == self.shape[0] + 1

    def batch_values(self, rows, cls_tokens):
        values = np.concatenate([self.values[self.offsets[i]:self.offsets[i + 1]] for i in rows])
        return values, np.asarray(cls_tokens[rows])


@dataclass
class PackedPairBatch:
    values: torch.Tensor
    owners: torch.Tensor
    positions: torch.Tensor
    batch_size: int
    slots: int


def packed_encoder(base_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
    class FlatPairEncoder(base_class):
        def forward(self, tokens: PackedPairBatch, mz, intensity, cls):
            local, global_embedding = super().forward(tokens.values, mz, intensity, cls)
            padded = local.new_zeros(tokens.batch_size, tokens.slots, local.shape[-1])
            padded[tokens.owners, tokens.positions] = local
            return padded, global_embedding

    return FlatPairEncoder


def load_pair_batch(tokens, cls_tokens, mz, intensity, idx, peak_slots, device):
    counts = tokens.offsets[idx + 1] - tokens.offsets[idx]
    values, cls = tokens.batch_values(idx, cls_tokens)
    owners = np.repeat(np.arange(len(idx)), counts)
    positions = np.concatenate([np.arange(count) for count in counts])
    packed = PackedPairBatch(
        torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.from_numpy(owners).to(device), torch.from_numpy(positions).to(device), len(idx), peak_slots,
    )
    source_mz = np.concatenate([mz[i, :count] for i, count in zip(idx, counts, strict=True)])
    source_intensity = np.concatenate([intensity[i, :count] for i, count in zip(idx, counts, strict=True)])
    return (packed, torch.from_numpy(source_mz.astype(np.float32)).to(device),
            torch.from_numpy(source_intensity.astype(np.float32)).to(device),
            torch.as_tensor(cls, dtype=torch.float32, device=device))


@dataclass
class PairAgreement:
    module: torch.nn.Module
    peak_fractions: torch.Tensor
    atom_fractions: torch.Tensor
    atom_mask: torch.Tensor

    def candidate_slice(self, selection: slice):
        return self.module.bonus_batched(self.peak_fractions, self.atom_fractions[:, selection],
                                         self.atom_mask[:, selection])


def lazy_agreement_bonus(agreement, peak_fractions, afrac, rows, unique, mapping, atom_mask, device):
    if agreement is None:
        return None
    fractions = torch.nn.utils.rnn.pad_sequence([afrac[s] for s in unique], batch_first=True, padding_value=-1)
    fractions = F.pad(fractions, (0, 0, 0, atom_mask.shape[-1] - fractions.shape[1]), value=-1)
    fp = torch.from_numpy(np.asarray(peak_fractions[rows], dtype=np.float32)).to(device)
    return PairAgreement(agreement, fp, fractions.to(device)[mapping], atom_mask)


def chunked_alignment(reference, chunk_size: int = 128):
    def alignment(peaks, peak_mask, atoms, atom_mask, bonus: PairAgreement | None = None):
        matches, means = [], []
        for start in range(0, atoms.shape[1], chunk_size):
            selection = slice(start, start + chunk_size)
            if bonus is None:
                current = reference(peaks, peak_mask, atoms[:, selection], atom_mask[:, selection])
            else:
                current = reference(peaks, peak_mask, atoms[:, selection], atom_mask[:, selection],
                                    bonus.candidate_slice(selection))
            matches.append(current[0])
            means.append(current[1])
        # Candidate-conditioned evidence is computed by the original scorer only
        # after all candidates have been reunited; its softmax is never chunked.
        return torch.cat(matches, dim=1), torch.cat(means, dim=1)

    return alignment


def chunked_query(reference, chunk_size: int = 128):
    def query(peaks, peak_mask, atoms, atom_mask):
        return torch.cat([reference(peaks, peak_mask, atoms[start:start + chunk_size],
                                    atom_mask[start:start + chunk_size])
                          for start in range(0, len(atoms), chunk_size)])

    return query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scripts", required=True, type=Path)
    parser.add_argument("--stage", choices=["baseline", "candidate"], required=True)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--local-pair-values", type=Path)
    storage.add_argument("--online-prepared", type=Path)
    parser.add_argument("--online-checkpoint", type=Path)
    parser.add_argument("--online-checkpoint-metadata", type=Path,
                        default=Path("/results/pair1b_step80000_flare/checkpoint_metadata.json"))
    args, remaining = parser.parse_known_args()
    if args.online_checkpoint is not None and args.online_prepared is None:
        parser.error("--online-checkpoint requires --online-prepared")
    out = Path(remaining[remaining.index("--out-dir") + 1])
    root = (Path(remaining[remaining.index("--tokens") + 1]).parent if args.stage == "baseline"
            else Path(remaining[remaining.index("--root") + 1]))
    metadata = json.loads((root / "embedding_metadata.json").read_text())
    assert metadata["maximum_pair_tokens"] == 2209 and metadata["pair_dim"] == 1536
    if "--init-checkpoint" in remaining:
        checkpoint = Path(remaining[remaining.index("--init-checkpoint") + 1])
        assert json.loads((checkpoint.parent / "pair_method.json").read_text())["readout"] == "flat_pairs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair_method.json").write_text(json.dumps({
        "readout": "flat_pairs", "input_dim": 1536, "maximum_pair_tokens": 2209,
        "local_tokens": "All ordered valid physical pairs, including diagonal",
        "mass_features_and_agreement": "Source peak i for each pair (i,j)",
        "cls": "Separate Z_CLS_CLS", "candidate_chunk": 128,
        "reference_scripts": str(args.reference_scripts),
        "local_pair_values": str(args.local_pair_values) if args.local_pair_values is not None else None,
        "online_prepared": str(args.online_prepared) if args.online_prepared is not None else None,
        "frozen_encoder_checkpoint": str(args.online_checkpoint) if args.online_checkpoint is not None else metadata["checkpoint"],
        "frozen_encoder_batch_cache": 1 if args.online_prepared is not None else None,
    }, indent=2) + "\n")
    sys.path.insert(0, str(args.reference_scripts))
    filename = "09_jepa_flare.py" if args.stage == "baseline" else "10_candidate_flare.py"
    spec = importlib.util.spec_from_file_location("flat_pair_flare_stage", args.reference_scripts / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_load = np.load
    token_source = None

    def load(path, *load_args, **load_kwargs):
        nonlocal token_source
        if Path(path).parent == root and Path(path).name in ["pair_values.float16.npy", "covariance_embeddings.float16.npy"]:
            if token_source is None:
                if args.online_prepared is not None:
                    from scripts.pair_flare_online import OnlinePairTokens
                    token_source = OnlinePairTokens(root, args.online_prepared, args.online_checkpoint_metadata,
                                                    checkpoint_path=args.online_checkpoint)
                else:
                    token_source = RaggedPairTokens(root, args.local_pair_values)
            return token_source
        if args.online_prepared is not None and Path(path) == root / "cls_embeddings.float16.npy":
            return None
        return original_load(path, *load_args, **load_kwargs)

    # Only this script's NumPy reference is adapted; libraries and other workers
    # retain the real module, and ordinary arrays still use the original loader.
    module.np = SimpleNamespace(**{**vars(np), "load": load})
    hybrid = module if args.stage == "baseline" else module.hybrid
    hybrid.HybridSpectrumEncoder = packed_encoder(hybrid.HybridSpectrumEncoder)
    hybrid.load_spectrum_batch = load_pair_batch
    hybrid.base.maxsim_query = chunked_query(hybrid.base.maxsim_query)
    ranks = []
    if args.stage == "candidate":
        module.alignment_components = chunked_alignment(module.alignment_components)
        module.agreement_bonus = lazy_agreement_bonus
        if "--eval-only" in remaining:
            reference_rank = module.ranks_with_ties

            def rank(scores, true_index):
                result = reference_rank(scores, true_index)
                ranks.append(result)
                return result

            module.ranks_with_ties = rank
    sys.argv = [filename, *remaining]
    module.main()
    if args.online_prepared is not None:
        name = "online_encoding_full_test.json" if "--eval-only" in remaining else "online_encoding.json"
        (out / name).write_text(json.dumps(token_source.statistics(), indent=2) + "\n")
    if ranks:
        (out / "test_ranks.json").write_text(json.dumps(ranks) + "\n")


if __name__ == "__main__":
    main()
