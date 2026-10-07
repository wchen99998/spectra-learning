"""Morgan fingerprints (radius 2, 2048 bits, RDKit defaults) for every graph-store molecule.

Row i is the packed bit vector of ``graphs/smiles.json[i]``; the structure-aware
retrieval loss uses 1 - Tanimoto between a candidate and the true molecule.
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RADIUS, BITS = 2, 2048


def chunk_fingerprints(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=BITS)
    out = np.zeros((len(smiles), BITS // 8), np.uint8)
    for i, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit cannot parse graph-store SMILES {value!r}")
        out[i] = np.packbits(generator.GetFingerprintAsNumPy(molecule).astype(bool))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True, help="Prepared graph store directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args(argv)
    smiles = json.loads((args.graphs / "smiles.json").read_text())
    chunks = [smiles[i:i + 20000] for i in range(0, len(smiles), 20000)]
    with Pool(args.workers) as pool:
        packed = np.concatenate(pool.map(chunk_fingerprints, chunks))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "morgan2_2048.u8.npy", packed)
    meta = {"radius": RADIUS, "bits": BITS, "rows": len(smiles), "generator": "rdFingerprintGenerator.Morgan",
            "graph_smiles_sha256": hashlib.sha256((args.graphs / "smiles.json").read_bytes()).hexdigest()}
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
