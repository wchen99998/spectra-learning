"""Package the editable report with compact inputs and its shared dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> None:
    paths = [p for p in ROOT.rglob('*') if p.is_file() and 'build' not in p.relative_to(ROOT).parts
             and '__pycache__' not in p.parts and p.suffix in {'.tex','.sty','.py','.sh','.md','.json','.csv','.pdf','.svg'}]
    shared=ROOT.parent/'shared'
    paths.extend([shared/'presentation-theme.sty',shared/'build_deck.sh',shared/'figure_standards.md'])
    paths.extend(p for p in (shared/'assets').iterdir() if p.suffix in {'.otf','.ttf','.txt'})
    paths=sorted(set(paths))
    archive=ROOT/'encoder_chemistry_source.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as stream:
        for path in paths:
            stream.write(path,path.relative_to(REPO))
    record={'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'scope':'Editable report, compact measured plot inputs, vector figures and shared theme/font dependencies. Existing repository .venv required; no checkpoint files or GPU jobs included.',
        'files':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'build/source_bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Packaged {len(paths)} files: {archive}')


if __name__=='__main__':
    main()
