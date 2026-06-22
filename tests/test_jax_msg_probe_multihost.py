import json
import os
import socket
import subprocess
import sys
import textwrap
import time


def test_msg_probe_jax_runs_with_simulated_multihost_sharded_batches() -> None:
    port = _free_port()
    script = textwrap.dedent(
        """
        import json
        import sys

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh
        from ml_collections import config_dict

        process_id = int(sys.argv[1])
        num_processes = int(sys.argv[2])
        port = sys.argv[3]
        jax.distributed.initialize(
            f"127.0.0.1:{port}",
            num_processes=num_processes,
            process_id=process_id,
            initialization_timeout=60,
        )

        from spectra_learning.probes.massspec import msg_probe_jax

        regression_names = tuple(msg_probe_jax.REGRESSION_PROBE_TASKS)
        binary_names = tuple(msg_probe_jax.BINARY_PROBE_TASKS)

        class FakeProbeData:
            batch_size = 4
            info = {
                "massspec_train_size": 3,
                "massspec_val_size": 3,
                "massspec_test_size": 3,
                "massspec_mcebio_test_size": 1,
                "probe_maccs_bits": 2,
            }

            def build_dataset(self, split, **kwargs):
                world_size = kwargs["distributed_world_size"]
                rank = kwargs["distributed_rank"]
                size = int(self.info[f"{split}_size"])
                local_batch_size = self.batch_size // world_size
                local_size = (size + world_size - 1) // world_size
                valid_local_size = (
                    0 if rank >= size else (size - 1 - rank) // world_size + 1
                )
                batches = []
                seen = 0
                while seen < local_size:
                    take = min(local_batch_size, local_size - seen)
                    valid_take = min(take, max(valid_local_size - seen, 0))
                    row_ids = rank + world_size * np.arange(seen, seen + take)
                    batch = {
                        "peak_mz": (
                            np.zeros((take, 3), dtype=np.float32) + row_ids[:, None]
                        ),
                        "peak_intensity": np.ones((take, 3), dtype=np.float32),
                        "peak_valid_mask": np.ones((take, 3), dtype=bool),
                        "probe_valid_mol": np.arange(take) < valid_take,
                        "probe_maccs": np.stack(
                            [row_ids % 2, (row_ids + 1) % 2],
                            axis=1,
                        ).astype(np.int32),
                    }
                    for name in regression_names:
                        batch[f"probe_{name}"] = row_ids.astype(np.float32) + 1.0
                    for name in binary_names:
                        batch[f"probe_{name}"] = (row_ids % 2).astype(np.float32)
                    batches.append(batch)
                    seen += take
                return batches

        class FakeMassSpecProbeData:
            @staticmethod
            def from_config(config, **kwargs):
                return FakeProbeData()

        def fake_extract_features(model, batch, *, use_pair_features):
            del model, use_pair_features
            peak_mz = batch["peak_mz"].astype(jnp.float32)
            features = jnp.stack(
                [
                    peak_mz[:, 0],
                    peak_mz[:, 1] + 1.0,
                    peak_mz[:, 2] + 2.0,
                    peak_mz[:, 0] + 3.0,
                ],
                axis=-1,
            )
            return jnp.repeat(features[:, None, :], repeats=3, axis=1)

        msg_probe_jax.MassSpecProbeData = FakeMassSpecProbeData
        msg_probe_jax._extract_features = fake_extract_features

        cfg = config_dict.ConfigDict()
        cfg.seed = 1
        cfg.model_dim = 4
        cfg.msg_probe_num_epochs = 1
        cfg.msg_probe_learning_rate = 0.01
        cfg.msg_probe_weight_decay = 0.0
        cfg.msg_probe_warmup_steps = 0
        cfg.msg_probe_mlp_hidden_dim = 4
        cfg.msg_probe_variants = ["mean"]
        cfg.msg_probe_early_stopping = False
        cfg.msg_probe_num_repeats = 1
        cfg.peak_ordering = "mz"
        cfg.msg_probe_batch_size = 4

        data_mesh = Mesh(np.asarray(jax.devices()), ("data",))
        with jax.set_mesh(data_mesh):
            metrics = msg_probe_jax.run_msg_probe_jax(
                config=cfg,
                model=object(),
                data_mesh=data_mesh,
            )
        print(
            json.dumps(
                {
                    "process_id": jax.process_index(),
                    "process_count": jax.process_count(),
                    "metric_count": len(metrics),
                    "has_mcebio": (
                        "msg_probe/mean/mcebio_sulfur_test/auc_sulfur" in metrics
                    ),
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

    outputs = _communicate_processes(processes, timeout_seconds=120)
    records = []
    for returncode, stdout, stderr in outputs:
        assert returncode == 0, stderr
        records.extend(
            json.loads(line) for line in stdout.splitlines() if line.startswith("{")
        )
    records.sort(key=lambda record: record["process_id"])

    assert [record["process_id"] for record in records] == [0, 1]
    assert all(record["process_count"] == 2 for record in records)
    assert all(record["metric_count"] > 0 for record in records)
    assert all(record["has_mcebio"] for record in records)


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
