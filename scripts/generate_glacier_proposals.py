"""Export frozen GLACIER proposals in the original graph-store atom order.

This is offline preprocessing in GLACIER's own PyTorch environment. Retrieval
training consumes only the resulting NumPy shards and remains entirely in JAX.
"""

import argparse
from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem


SOURCE_COMMIT = "67ba0f202305d4bed3134991dd4562fe1edb1763"
CHECKPOINT_SHA256 = "5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11"
SCHEMA = "glacier_learned_proposals_original_order_v1"
ENERGIES = (20.0, 30.0, 40.0)


def canonical_atom_mapping(
    smiles: str, *, annotation_changes: dict[str, int] | None = None,
) -> tuple[str, np.ndarray]:
    """Map the canonical writer's atom order directly onto original graph atoms.

    A stereochemical substructure search can fail after RDKit normalizes the
    written stereochemistry. The writer records the actual permutation, so no
    matching/search or choice between symmetric graph automorphisms is needed.
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or original.GetNumAtoms() == 0:
        raise ValueError(f"Cannot parse graph-store molecule: {smiles}")
    heavy = Chem.RemoveHs(original)
    if heavy.GetNumAtoms() != original.GetNumAtoms():
        raise ValueError("GLACIER heavy-atom indexing differs from graph-store indexing")
    canonical = Chem.MolToSmiles(heavy)
    canonical_mol = Chem.MolFromSmiles(canonical)
    mapping = json.loads(heavy.GetProp("_smilesAtomOutputOrder"))
    n = original.GetNumAtoms()
    if canonical_mol is None or canonical_mol.GetNumAtoms() != n or sorted(mapping) != list(range(n)):
        raise ValueError(f"Cannot map canonical atoms bijectively: {smiles}")
    for canonical_index, original_index in enumerate(mapping):
        a = canonical_mol.GetAtomWithIdx(canonical_index)
        b = original.GetAtomWithIdx(original_index)
        signature = lambda atom: (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                                  atom.GetTotalNumHs())
        if signature(a) != signature(b):
            raise ValueError(f"Canonical writer changed atom identity: {smiles}")
        if annotation_changes is not None and a.GetIsAromatic() != b.GetIsAromatic():
            annotation_changes["aromatic_atoms"] = annotation_changes.get("aromatic_atoms", 0) + 1
    if canonical_mol.GetNumBonds() != original.GetNumBonds():
        raise ValueError(f"Canonical writer changed bond count: {smiles}")
    for bond in canonical_mol.GetBonds():
        original_bond = original.GetBondBetweenAtoms(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])
        if original_bond is None:
            raise ValueError(f"Canonical writer changed bond connectivity: {smiles}")
        # Aromatic/Kekule annotations can normalize on parsing the written
        # SMILES. Membership mapping requires the same atoms and connectivity.
        if annotation_changes is not None and original_bond.GetBondType() != bond.GetBondType():
            annotation_changes["bond_types"] = annotation_changes.get("bond_types", 0) + 1
    return canonical, np.asarray(mapping, np.int16)


def _preflight_chunk(rows: list[tuple[int, str]]) -> dict:
    failures, unsupported, input_issues, annotation_changes = [], [], [], []
    minimum_atoms = 32767
    allowed_elements = {"C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B",
                        "Se", "Fe", "Co", "As", "Na", "K"}
    allowed_bonds = {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC}
    for graph_id, smiles in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            minimum_atoms = min(minimum_atoms, mol.GetNumAtoms())
            elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - allowed_elements)
            max_degree = max((atom.GetDegree() for atom in mol.GetAtoms()), default=0)
            bond_types = sorted({str(bond.GetBondType()) for bond in mol.GetBonds()
                                 if bond.GetBondType() not in allowed_bonds})
            if elements or max_degree > 6 or bond_types:
                input_issues.append({"graph_id": graph_id, "elements": elements,
                                     "max_degree": max_degree, "bond_types": bond_types})
            if mol.GetNumAtoms() > 100:
                unsupported.append(graph_id)
                continue
        try:
            annotations = {}
            canonical_atom_mapping(smiles, annotation_changes=annotations)
            if annotations:
                annotation_changes.append({"graph_id": graph_id, **annotations})
        except ValueError as error:
            failures.append({"graph_id": graph_id, "smiles": smiles, "error": str(error)})
    return {"checked": len(rows), "failures": failures, "unsupported_graph_ids": unsupported,
            "input_issues": input_issues, "minimum_atoms": minimum_atoms,
            "annotation_changes": annotation_changes}


def preflight_mapping(graphs: str, output: str, workers: int = 16) -> dict:
    """Check every supported graph on CPU before allocating an inference GPU."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from rdkit import rdBase

    started = time.monotonic()
    graph_directory = Path(graphs)
    smiles = json.loads((graph_directory / "smiles.json").read_text())
    graph_metadata = json.loads((graph_directory / "metadata.json").read_text())
    digest = sha256()
    for value in smiles:
        digest.update(value.encode() + b"\0")
    if digest.hexdigest() != graph_metadata["smiles_sha256"]:
        raise ValueError("Graph-store SMILES fingerprint is invalid")
    report = {
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "smiles_sha256": digest.hexdigest(), "molecules": len(smiles),
        "rdkit_version": rdBase.rdkitVersion, "checked": 0, "failures": [],
        "unsupported_graph_ids": [], "input_issues": [], "annotation_changes": [],
        "minimum_atoms": 32767, "complete": False,
    }
    chunk_size = 4096
    chunks = (list(enumerate(smiles[start:start + chunk_size], start))
              for start in range(0, len(smiles), chunk_size))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for index, result in enumerate(pool.map(_preflight_chunk, chunks)):
            report["checked"] += result["checked"]
            report["failures"].extend(result["failures"])
            report["unsupported_graph_ids"].extend(result["unsupported_graph_ids"])
            report["input_issues"].extend(result["input_issues"])
            report["annotation_changes"].extend(result["annotation_changes"])
            report["minimum_atoms"] = min(report["minimum_atoms"], result["minimum_atoms"])
            if index % 16 == 0:
                print(json.dumps({"checked": report["checked"], "molecules": len(smiles),
                                  "failures": len(report["failures"]),
                                  "seconds": time.monotonic() - started}), flush=True)
                _write_json(path, report)
    report.update(complete=True, seconds=time.monotonic() - started)
    _write_json(path, report)
    return report


def merge_predictions(
    merged: dict[tuple[int, ...], float], membership: np.ndarray,
    intensity: np.ndarray, mapping: np.ndarray,
) -> None:
    """Union atom sets over spectral shifts/energies, taking maximum intensity."""
    membership = np.asarray(membership, bool)
    intensity = np.asarray(intensity, np.float32)
    if membership.ndim != 2 or membership.shape[0] != len(intensity):
        raise ValueError("Fragment masks and intensities have inconsistent shapes")
    if membership.shape[1] < len(mapping) or membership[:, len(mapping):].any():
        raise ValueError("Predicted fragment includes an atom outside its molecule")
    if not np.isfinite(intensity).all() or np.any(intensity < 0) or np.any(intensity > 1):
        raise ValueError("GLACIER intensity must be finite and in [0,1]")
    # Shift rows repeat memberships; merge them before converting to Python keys.
    unique, inverse = np.unique(membership[:, :len(mapping)], axis=0, return_inverse=True)
    maximum = np.zeros(len(unique), np.float32)
    np.maximum.at(maximum, inverse, intensity)
    for row, value in zip(unique, maximum, strict=True):
        atoms = tuple(sorted(mapping[row].tolist()))
        if atoms:
            merged[atoms] = max(merged.get(atoms, 0.0), float(value))


def pack_proposals(records: list[tuple[int, dict[tuple[int, ...], float]]]) -> dict[str, np.ndarray]:
    graph_ids, frag_off, mem_off, atoms, intensity = [], [0], [0], [], []
    for graph_id, proposals in records:
        graph_ids.append(graph_id)
        for member, value in sorted(proposals.items(), key=lambda item: (-item[1], item[0])):
            atoms.extend(member)
            mem_off.append(len(atoms))
            intensity.append(value)
        frag_off.append(len(intensity))
    return {
        "graph_ids": np.asarray(graph_ids, np.int32),
        "frag_off": np.asarray(frag_off, np.int64),
        "mem_off": np.asarray(mem_off, np.int64),
        "atoms": np.asarray(atoms, np.int16),
        "inten": np.asarray(intensity, np.float32),
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generate(args: argparse.Namespace, commit: Callable[[], None] | None = None) -> dict:
    import importlib.metadata
    import torch
    from torch.utils.data import DataLoader
    from ms_pred.glacier.joint_model import JointModel
    from ms_pred.glacier.predict_smis_joint import (
        GraphormerFeatDataset, collate_predict, _batch_to_device,
    )

    torch.set_num_threads(1)
    torch.manual_seed(0)
    start = time.monotonic()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_hash = sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the pinned official GLACIER release")
    graphs = Path(args.graphs)
    smiles = json.loads((graphs / "smiles.json").read_text())
    graph_meta = json.loads((graphs / "metadata.json").read_text())
    digest = sha256()
    for value in smiles:
        digest.update(value.encode() + b"\0")
    fingerprint = digest.hexdigest()
    if graph_meta["smiles_sha256"] != fingerprint:
        raise ValueError("Graph-store SMILES do not match their metadata fingerprint")
    count = len(smiles)
    if args.ids:
        selected = np.unique(np.load(args.ids)).astype(np.int32)
        if selected.min() < 0 or selected.max() >= count:
            raise ValueError("Selected graph IDs lie outside the graph store")
    elif args.limit:
        selected = np.sort(np.random.default_rng(args.seed).choice(
            count, min(args.limit, count), replace=False,
        )).astype(np.int32)
    else:
        selected = np.arange(count, dtype=np.int32)
    identity = {
        "schema": SCHEMA, "source_commit": SOURCE_COMMIT,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_hash, "smiles_sha256": fingerprint,
        "molecules": count, "selected_molecules": len(selected),
        "selection_seed": args.seed, "selection_limit": args.limit,
        "collision_energies": list(ENERGIES), "adduct": args.adduct, "instrument": args.instrument,
        **({"selection_ids_sha256": sha256(Path(args.ids).read_bytes()).hexdigest()} if args.ids else {}),
        "precision": "float32",
        "atom_order": "Chem.MolFromSmiles(original_graph_store_smiles)",
        "aggregation": "union atom sets; maximum predicted intensity over shifts and energies",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing proposal directory belongs to a different configuration")
    else:
        metadata = {**identity, "completed_shards": [], "unsupported_graph_ids": [], "complete": False}
        _write_json(metadata_path, metadata)
    completed = np.zeros(count, bool)
    for name in metadata["completed_shards"]:
        with np.load(output / name, allow_pickle=False) as shard:
            ids = shard["graph_ids"]
            if completed[ids].any():
                raise ValueError("Existing shards contain duplicate graph IDs")
            completed[ids] = True
    remaining = selected[~completed[selected]]
    if not len(remaining):
        metadata["complete"] = True
        _write_json(metadata_path, metadata)
        if commit:
            commit()
        return metadata
    print(json.dumps({"phase": "preparing", "remaining_molecules": len(remaining)}), flush=True)

    mapping: dict[int, np.ndarray] = {}
    canonical: dict[int, str] = {}
    unsupported = []
    for prepared_index, graph_id in enumerate(remaining.tolist(), 1):
        mol = Chem.MolFromSmiles(smiles[graph_id])
        if mol.GetNumAtoms() > 100:
            unsupported.append(graph_id)
            continue
        text, permutation = canonical_atom_mapping(smiles[graph_id])
        canonical[graph_id] = text
        mapping[graph_id] = permutation
        if prepared_index % 50_000 == 0:
            print(json.dumps({"phase": "canonicalizing", "prepared": prepared_index,
                              "total": len(remaining), "seconds": time.monotonic() - start}), flush=True)
    # Keep all three energies adjacent, while grouping similarly sized graphs.
    ordered = sorted(canonical, key=lambda i: (len(mapping[i]), canonical[i], i))
    entries = [(canonical[i], str(i), energy, args.adduct, args.instrument, "")
               for i in ordered for energy in ENERGIES]
    metadata["unsupported_graph_ids"] = sorted(set(metadata["unsupported_graph_ids"]) | set(unsupported))
    _write_json(metadata_path, metadata)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = JointModel(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    del ckpt
    model.eval().to("cuda")
    dataset = GraphormerFeatDataset(entries, model.tree_processor,
                                   sort_for_batching=False, feat_cache_size=256)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": True}
    if args.workers:
        loader_kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_predict, **loader_kwargs)
    print(json.dumps({"phase": "inference", "supported": len(ordered), "unsupported": len(unsupported),
                      "batch_size": args.batch_size, "workers": args.workers,
                      "gpu": torch.cuda.get_device_name()}), flush=True)
    records = [(i, {}) for i in unsupported]
    pending: dict[int, dict[tuple[int, ...], float]] = {}
    seen_energy: dict[int, int] = {}
    batches, processed, forward_seconds, cpu_seconds, merge_seconds = 0, 0, 0.0, 0.0, 0.0
    warmup_seconds = None
    warmup_molecules = 0
    stopped_for_budget = False
    iteration_start = time.monotonic()

    def flush() -> None:
        nonlocal records
        if not records:
            return
        name = f"shard-{len(metadata['completed_shards']):06d}.npz"
        temporary = output / f"{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **pack_proposals(records))
        temporary.replace(output / name)
        metadata["completed_shards"].append(name)
        _write_json(metadata_path, metadata)
        records = []
        if commit:
            commit()

    for batch in loader:
        previous = time.monotonic()
        cpu_seconds += previous - iteration_start
        batch = _batch_to_device(batch, "cuda")
        with torch.inference_mode():
            prediction = model.predict_inten_frag_batch(batch)
        torch.cuda.synchronize()
        forward_seconds += time.monotonic() - previous
        merge_start = time.monotonic()
        # _assemble_spec_frag repeats each membership for two adduct states and
        # model.output_size hydrogen shifts. Reduce these on GPU before transfer.
        shifts = batch["adduct_mass_shifts"].shape[1] * model.output_size
        for name, spec, member in zip(batch["pred_names"], prediction["spec"], prediction["frag"], strict=True):
            graph_id = int(name)
            proposals = pending.setdefault(graph_id, {})
            merge_predictions(proposals, member[::shifts].cpu().numpy(),
                              spec[:, 1].reshape(-1, shifts).amax(1).cpu().numpy(), mapping[graph_id])
            seen_energy[graph_id] = seen_energy.get(graph_id, 0) + 1
            if seen_energy[graph_id] == len(ENERGIES):
                records.append((graph_id, pending.pop(graph_id)))
                del seen_energy[graph_id]
                processed += 1
        merge_seconds += time.monotonic() - merge_start
        batches += 1
        if batches == 1:
            warmup_seconds = time.monotonic() - start
            warmup_molecules = processed
        if len(records) >= args.shard_size:
            flush()
        if batches % 10 == 0:
            elapsed = time.monotonic() - start
            print(json.dumps({"batches": batches, "molecules": processed, "elapsed_seconds": elapsed,
                              "molecules_per_second": processed / elapsed}), flush=True)
        if args.max_runtime_seconds and time.monotonic() - start >= args.max_runtime_seconds and not pending:
            stopped_for_budget = True
            break
        iteration_start = time.monotonic()
    if pending or seen_energy:
        raise ValueError("Incomplete collision-energy predictions remain at export end")
    flush()
    elapsed = time.monotonic() - start
    metadata["complete"] = not stopped_for_budget
    metadata["benchmark"] = {
        "seconds": elapsed, "new_molecules": processed + len(unsupported), "supported_molecules": processed,
        "molecules_per_second": (processed + len(unsupported)) / elapsed, "forward_seconds": forward_seconds,
        "dataloader_wait_seconds": cpu_seconds, "first_batch_seconds": warmup_seconds,
        "postprocess_seconds": merge_seconds,
        "steady_molecules_per_second": ((processed - warmup_molecules) / (elapsed - warmup_seconds)
                                         if warmup_seconds is not None and processed > warmup_molecules else None),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "batch_size": args.batch_size, "workers": args.workers,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "dgl", "rdkit", "numpy", "pytorch-lightning", "LinSATNet", "pygmtools")},
    }
    _write_json(metadata_path, metadata)
    if commit:
        commit()
    print(json.dumps(metadata["benchmark"], indent=2), flush=True)
    return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--graphs", default="/results/atom_flare_jax_55dfbdf_latest_val5000/prepared/graphs")
    result.add_argument("--checkpoint", default="/results/glacier_release_20260918/best.ckpt")
    result.add_argument("--output", required=True)
    result.add_argument("--batch-size", type=int, default=48)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--shard-size", type=int, default=1024)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--ids", default="", help="npy of graph IDs to generate (overrides --limit)")
    result.add_argument("--adduct", default="[M+H]+")
    result.add_argument("--instrument", default="Orbitrap")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-runtime-seconds", type=float, default=0)
    return result


if __name__ == "__main__":
    generate(parser().parse_args())
