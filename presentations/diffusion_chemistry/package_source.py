"""Archive the editable supplement and its compact evidence dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> None:
    paths = [path for path in ROOT.rglob("*") if path.is_file()
             and "build" not in path.relative_to(ROOT).parts and "__pycache__" not in path.parts
             and path.suffix in {".tex", ".py", ".sh", ".md", ".json", ".pdf", ".svg"}]
    shared = ROOT.parent / "shared"
    paths += [shared / name for name in ("presentation-theme.sty", "build_deck.sh", "figure_standards.md")]
    paths += [path for path in (shared / "assets").iterdir() if path.suffix in {".otf", ".ttf", ".txt"}]
    paths = sorted(set(paths))
    archive = ROOT / "diffusion_chemistry_source.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as stream:
        for path in paths:
            stream.write(path, path.relative_to(REPO))
    record = {"archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "scope": "Editable report, compact measured inputs, self-contained plotting code and shared theme/fonts. Existing repository .venv needed; no full artifact directory, checkpoint state or GPU inference is required.",
              "files": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (ROOT / "build/source_bundle.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Packaged {len(paths)} source/evidence files.")


if __name__ == "__main__":
    main()
