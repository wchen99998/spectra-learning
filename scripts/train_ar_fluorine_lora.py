from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.probes.massspec.ar_fluorine import main


if __name__ == "__main__":
    main()
