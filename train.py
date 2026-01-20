import jax
from absl import logging
from jax.sharding import NamedSharding, Mesh
from jax.sharding import PartitionSpec as P


def create_device_mesh(config):
    """Create device mesh based on configuration."""
    mesh_config = config.mesh_config
    mesh = jax.make_mesh(mesh_config.mesh_shape, mesh_config.mesh_axis_names)
    logging.info(f"Created mesh: {mesh}")
    return mesh


def build_mesh_and_sharding(
    config,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    device_mesh = create_device_mesh(config)
    replicated_sharding = NamedSharding(device_mesh, P())
    data_sharding = NamedSharding(device_mesh, P("data", None))

    return device_mesh, replicated_sharding, data_sharding


def get_data_sharding_for_rank(
    data_sharding: NamedSharding | None,
    rank: int,
) -> NamedSharding | None:
    """Return a sharding that partitions the batch axis regardless of tensor rank."""
    if data_sharding is None:
        return None

    if rank == len(data_sharding.spec):
        return data_sharding

    if rank == 0:
        pspec = P()
    else:
        pspec = P("data", *([None] * (rank - 1)))

    return NamedSharding(data_sharding.mesh, pspec)
