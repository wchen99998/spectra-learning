from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spectra_learning.data.massspec_probe import MassSpecProbeData


def __getattr__(name: str) -> Any:
    if name == "MassSpecProbeData":
        from spectra_learning.data.massspec_probe import MassSpecProbeData

        return MassSpecProbeData
    if name == "run_msg_probe":
        from spectra_learning.probes.massspec.msg_probe import run_msg_probe

        return run_msg_probe
    if name == "run_msg_probe_jax":
        from spectra_learning.probes.massspec.msg_probe_jax import run_msg_probe_jax

        return run_msg_probe_jax
    raise AttributeError(name)

__all__ = ["MassSpecProbeData", "run_msg_probe", "run_msg_probe_jax"]
