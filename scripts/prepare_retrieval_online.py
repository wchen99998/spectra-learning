"""Bind raw retrieval inputs to a pinned student without exporting frozen tokens."""

import json
from pathlib import Path

from scripts.export_retrieval_tokens import validate_encoder_contract
from scripts.export_salt_glacier_tokens import atomic_json, file_sha256, link_original_inputs


def prepare_online_inputs(paths: dict, reference: dict, selection_path: Path):
    paths = {k: Path(v) for k, v in paths.items()}
    reference = {k: Path(v) for k, v in reference.items()}
    for name in ("prepared", "fragments", "peak_fractions", "jepa_data"):
        if paths[name] != reference[name]:
            raise ValueError(f"Shared retrieval input changed: {name}")
    selection = json.loads(selection_path.read_text())
    metadata = json.loads(paths["jepa_metadata"].read_text())
    original_config = json.loads(reference["jepa_metadata"].read_text())["task_contract"]["config"]
    contract = validate_encoder_contract(original_config, metadata["task_contract"]["config"])
    copy_record = json.loads((paths["jepa_checkpoint"] / "checkpoint_copy.json").read_text())
    if (not selection["all_checks_passed"] or not copy_record["all_checks_passed"]
            or copy_record["objects"] != selection["objects"]
            or file_sha256(paths["jepa_metadata"]) != selection["checkpoint_metadata_sha256"]):
        raise ValueError("Staged checkpoint differs from pinned inventory")
    root = paths["input_root"]
    root.mkdir(parents=True, exist_ok=True)
    link_original_inputs(root, reference["input_root"])
    original = json.loads((reference["input_root"] / "embedding_metadata.json").read_text())
    # Retain only the representation/input contract, not an old cache's completion or fingerprint.
    fields = ("embedding", "embedding_shape", "embedding_dim", "embedding_dtype", "samples",
              "metadata_input_dim", "tsv_sha256")
    embedding = {key: original[key] for key in fields}
    embedding.update(
        input_mode="online_trainable_jepa", embedding_complete=False,
        run_id=selection["run_id"], checkpoint=str(paths["jepa_checkpoint"]),
        checkpoint_role="student", checkpoint_step=selection["selected_checkpoint_step"],
        config_metadata_path=str(paths["jepa_metadata"]), data_root=str(paths["jepa_data"]),
        source_embedding_root=str(reference["input_root"]),
        checkpoint_selection_sha256=file_sha256(selection_path), encoder_contract=contract)
    atomic_json(root / "embedding_metadata.json", embedding)
    return embedding
