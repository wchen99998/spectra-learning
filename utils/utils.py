# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils."""

import jax
import numpy as np
from clu import platform

def get_rng(seed: None | int | tuple[int, int]) -> np.ndarray:
    """Returns a JAX RNGKey."""
    if seed is None:
        # Case 1: No random seed given, use XManager ID.
        # All processes (and restarts) get exactly the same seed but every work unit
        # and experiment is different.
        work_unit = platform.work_unit()
        rng = (work_unit.experiment_id, work_unit.id)
    elif isinstance(seed, int):
        # Case 2: Single integer given.
        rng = (0, seed)
    else:
        # Case 3: tuple[int, int] given.
        if not isinstance(seed, (tuple, list)) or len(seed) != 2:
            raise ValueError(
                "Random seed must be an integer or tuple of 2 integers "
                f"but got {seed!r}"
            )
        rng = seed
    # JAX RNGKeys are arrays of np.uint32 and shape [2].
    return np.asarray(rng, dtype=np.uint32)


class StepTraceContextHelper:
    """Helper class to use jax.profiler.StepTraceAnnotation."""

    def __init__(self, name: str, init_step_num: int):
        self.name = name
        self.step_num = init_step_num
        self.context = None

    def __enter__(self):
        self.context = jax.profiler.StepTraceAnnotation(
            self.name, step_num=self.step_num
        )
        self.step_num += 1
        self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        assert self.context is not None, "Exited context without entering."
        self.context.__exit__(exc_type, exc_value, tb)
        self.context = None

    def next_step(self):
        if self.context is None:
            raise ValueError("Must call next_step() within a context.")
        self.__exit__(None, None, None)
        self.__enter__()
