import unittest

import torch

from gram_newton_schulz.gram_newton_schulz import GramNewtonSchulz
from gram_newton_schulz.muon import muon as muon_mod
from gram_newton_schulz.muon.muon_utils import muon_opt_utils
from gram_newton_schulz.standard_newton_schulz import StandardNewtonSchulz
from utils.training import (
    _dynamic_muon_update_pre_orthogonalize,
    _sorted_create_param_batches,
    patch_gns_muon_compile_for_dynamic_shapes,
)

_REFERENCE_MUON_UPDATE_PRE = getattr(
    muon_opt_utils.muon_update_pre_orthogonalize,
    "_torchdynamo_orig_callable",
    muon_opt_utils.muon_update_pre_orthogonalize,
)


class GNSDynamicPatchTests(unittest.TestCase):
    def test_dynamic_muon_update_pre_orthogonalize_matches_reference(self):
        gradients = [torch.randn(4, 4), torch.randn(4, 4)]
        momentums = [torch.randn(4, 4), torch.randn(4, 4)]
        gradients_ref = [tensor.clone() for tensor in gradients]
        momentums_ref = [tensor.clone() for tensor in momentums]
        momentum = torch.tensor(0.95)

        expected = _REFERENCE_MUON_UPDATE_PRE(
            gradients_ref,
            momentums_ref,
            momentum,
            True,
        )
        actual = _dynamic_muon_update_pre_orthogonalize(
            gradients,
            momentums,
            momentum,
            True,
        )

        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            self.assertTrue(torch.allclose(actual_tensor, expected_tensor))
        for actual_tensor, expected_tensor in zip(
            momentums,
            momentums_ref,
            strict=True,
        ):
            self.assertTrue(torch.allclose(actual_tensor, expected_tensor))

    def test_patch_gns_muon_compile_for_dynamic_shapes_rebinds_dependency(self):
        patch_gns_muon_compile_for_dynamic_shapes()

        self.assertIs(
            muon_mod.muon_update_pre_orthogonalize,
            _dynamic_muon_update_pre_orthogonalize,
        )
        self.assertIs(
            muon_opt_utils.muon_update_pre_orthogonalize,
            _dynamic_muon_update_pre_orthogonalize,
        )
        self.assertTrue(hasattr(GramNewtonSchulz.__call__, "_torchdynamo_orig_callable"))
        self.assertTrue(
            hasattr(StandardNewtonSchulz.__call__, "_torchdynamo_orig_callable")
        )

    def test_sorted_create_param_batches_prefers_large_shape_buckets_first(self):
        batch_large = [torch.randn(8, 8) for _ in range(3)]
        batch_small = [torch.randn(4, 4)]
        batches = _sorted_create_param_batches(
            [batch_small[0], batch_large[0], batch_large[1], batch_large[2]]
        )
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(tuple(batches[0][0].shape), (8, 8))
