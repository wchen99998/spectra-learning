"""CLI entry point — delegates to train.py __main__ block."""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("train", run_name="__main__")
