import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.training.tpu_compile import main


if __name__ == "__main__":
    main()
