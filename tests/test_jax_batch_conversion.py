import json
import numpy as np
import os
import socket
import subprocess
import sys
import textwrap
import time
import torch


def test_numpy_batch_converts_to_jax() -> None:
    from spectra_learning.training.pretrain_jax import numpy_batch_to_jax

    batch = {
        "peak_mz": np.zeros((2, 4), dtype=np.float32),
        "peak_intensity": np.ones((2, 4), dtype=np.float32),
        "peak_valid_mask": np.ones((2, 4), dtype=bool),
    }

    jax_batch = numpy_batch_to_jax(batch)

    assert jax_batch["peak_mz"].shape == (2, 4)
    assert jax_batch["peak_intensity"].dtype.name == "float32"
    assert jax_batch["peak_valid_mask"].dtype.name == "bool"


def test_stacked_micro_batches_stay_on_host_until_jax_conversion() -> None:
    from spectra_learning.training.pretrain_jax import _stack_micro_batches

    batches = [
        {
            "peak_mz": np.full((2, 4), step, dtype=np.float32),
            "peak_valid_mask": np.ones((2, 4), dtype=bool),
        }
        for step in range(3)
    ]

    stacked = _stack_micro_batches(batches)

    assert isinstance(stacked["peak_mz"], np.ndarray)
    assert stacked["peak_mz"].shape == (3, 2, 4)
    assert stacked["peak_mz"][2, 0, 0] == 2.0


def test_process_local_mesh_conversion_passes_host_data_to_jax(monkeypatch) -> None:
    from spectra_learning.training import pretrain_jax

    pretrain_jax._jax_data_mesh_for_device_count.cache_clear()
    calls = []

    def fake_make_array_from_process_local_data(sharding, local_data):
        calls.append((sharding, local_data))
        return local_data

    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        pretrain_jax.jax,
        "make_array_from_process_local_data",
        fake_make_array_from_process_local_data,
    )
    data_mesh = pretrain_jax._jax_data_mesh_for_device_count(1)
    batch = {
        "peak_mz": np.zeros((2, 4), dtype=np.float32),
        "peak_valid_mask": np.ones((2, 4), dtype=bool),
    }

    converted = pretrain_jax.numpy_batch_to_jax(
        batch,
        data_mesh=data_mesh,
        batch_axis=0,
    )

    assert converted["peak_mz"] is batch["peak_mz"]
    assert len(calls) == 2
    assert all(isinstance(local_data, np.ndarray) for _, local_data in calls)


def test_process_local_mesh_conversion_accepts_torch_tensors(monkeypatch) -> None:
    from spectra_learning.training import pretrain_jax

    pretrain_jax._jax_data_mesh_for_device_count.cache_clear()
    calls = []

    def fake_make_array_from_process_local_data(sharding, local_data):
        calls.append((sharding, local_data))
        return local_data

    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        pretrain_jax.jax,
        "make_array_from_process_local_data",
        fake_make_array_from_process_local_data,
    )
    data_mesh = pretrain_jax._jax_data_mesh_for_device_count(1)
    batch = {
        "input_token_ids": torch.zeros((256, 392), dtype=torch.long),
        "target_loss_mask": torch.ones((256, 392), dtype=torch.bool),
    }

    converted = pretrain_jax.numpy_batch_to_jax(
        batch,
        data_mesh=data_mesh,
        batch_axis=0,
    )

    assert converted["input_token_ids"].dtype == np.int64
    assert converted["target_loss_mask"].dtype == np.bool_
    assert len(calls) == 2
    assert all(isinstance(local_data, np.ndarray) for _, local_data in calls)


def test_multihost_data_mesh_groups_devices_by_process(monkeypatch) -> None:
    from spectra_learning.training import pretrain_jax

    class FakeDevice:
        def __init__(self, process_index: int, device_id: int):
            self.process_index = process_index
            self.id = device_id

    devices = [
        FakeDevice(0, 0),
        FakeDevice(1, 1),
        FakeDevice(2, 2),
        FakeDevice(3, 3),
        FakeDevice(0, 4),
        FakeDevice(1, 5),
        FakeDevice(2, 6),
        FakeDevice(3, 7),
    ]

    monkeypatch.setattr(pretrain_jax.jax, "device_count", lambda: len(devices))
    monkeypatch.setattr(pretrain_jax.jax, "process_count", lambda: 4)
    monkeypatch.setattr(pretrain_jax.jax, "devices", lambda: devices)
    pretrain_jax._jax_data_mesh_for_device_count.cache_clear()

    mesh = pretrain_jax._jax_data_mesh_for_device_count(len(devices))

    assert [device.process_index for device in mesh.devices] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_process_local_data_sharding_with_simulated_multi_process() -> None:
    port = _free_port()
    script = textwrap.dedent(
        """
        import json
        import sys

        import jax
        import numpy as np

        process_id = int(sys.argv[1])
        num_processes = int(sys.argv[2])
        port = sys.argv[3]
        jax.distributed.initialize(
            f"127.0.0.1:{port}",
            num_processes=num_processes,
            process_id=process_id,
            initialization_timeout=60,
        )

        from spectra_learning.training import pretrain_jax

        local_batch = (
            process_id * 1000 + np.arange(8, dtype=np.float32).reshape(2, 4, 1)
        )
        batch = {
            "peak_mz": local_batch,
            "peak_valid_mask": np.ones(local_batch.shape, dtype=bool),
        }
        data_mesh = pretrain_jax._jax_data_mesh_for_device_count(jax.device_count())
        jax_batch = pretrain_jax.numpy_batch_to_jax(
            batch,
            data_mesh=data_mesh,
            batch_axis=1,
        )
        peak_mz = jax_batch["peak_mz"]
        local_values = []
        shard_shapes = []
        for shard in peak_mz.addressable_shards:
            shard_values = np.asarray(shard.data)
            local_values.extend(shard_values.reshape(-1).astype(int).tolist())
            shard_shapes.append(shard_values.shape)

        print(
            json.dumps(
                {
                    "process_id": jax.process_index(),
                    "process_count": jax.process_count(),
                    "device_count": jax.device_count(),
                    "local_device_count": jax.local_device_count(),
                    "global_shape": peak_mz.shape,
                    "committed": peak_mz.committed,
                    "fully_addressable": peak_mz.is_fully_addressable,
                    "local_values": local_values,
                    "shard_shapes": shard_shapes,
                }
            ),
            flush=True,
        )
        jax.effects_barrier()
        """
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(process_id), "2", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for process_id in range(2)
    ]

    outputs = _communicate_processes(processes, timeout_seconds=90)
    records = []
    for returncode, stdout, stderr in outputs:
        assert returncode == 0, stderr
        records.extend(
            json.loads(line) for line in stdout.splitlines() if line.startswith("{")
        )

    records.sort(key=lambda record: record["process_id"])

    assert [record["process_id"] for record in records] == [0, 1]
    assert all(record["process_count"] == 2 for record in records)
    assert all(record["device_count"] == 4 for record in records)
    assert all(record["local_device_count"] == 2 for record in records)
    assert all(record["global_shape"] == [2, 8, 1] for record in records)
    assert all(record["committed"] for record in records)
    assert all(not record["fully_addressable"] for record in records)
    assert sorted(records[0]["local_values"]) == list(range(8))
    assert sorted(records[1]["local_values"]) == list(range(1000, 1008))
    assert records[0]["shard_shapes"] == [[2, 2, 1], [2, 2, 1]]
    assert records[1]["shard_shapes"] == [[2, 2, 1], [2, 2, 1]]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _communicate_processes(
    processes: list[subprocess.Popen[str]],
    *,
    timeout_seconds: float,
) -> list[tuple[int, str, str]]:
    deadline = time.monotonic() + timeout_seconds
    outputs: list[tuple[int, str, str]] = []
    try:
        for process in processes:
            remaining = max(1.0, deadline - time.monotonic())
            stdout, stderr = process.communicate(timeout=remaining)
            outputs.append((int(process.returncode), stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    return outputs
