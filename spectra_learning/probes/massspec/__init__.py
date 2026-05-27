from typing import Any

from spectra_learning.probes.massspec.data import MassSpecProbeData


def __getattr__(name: str) -> Any:
    if name == "run_msg_probe":
        from spectra_learning.probes.massspec.msg_probe import run_msg_probe

        return run_msg_probe
    raise AttributeError(name)

__all__ = ["MassSpecProbeData", "run_msg_probe"]
