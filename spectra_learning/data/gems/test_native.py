import unittest

import torch

import spectra_learning.data.massspec_probe as massspec_probe_data


class MassSpecPreprocessTests(unittest.TestCase):
    def test_probe_collator_applies_peak_and_precursor_window_filters(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=8,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=0.01,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=5.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :5] = torch.tensor([90.0, 95.0, 97.0, 98.5, 150.0])
        spectra[1, :4] = torch.tensor([50.0, 60.0, 194.0, 210.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[0, :5] = torch.tensor([1.0, 0.009, 0.8, 0.7, 0.5])
        intensity[1, :4] = torch.tensor([0.02, 0.5, 0.8, 0.9])
        batch = collator(
            [
                {
                    "spectra": torch.stack([spectra[0], intensity[0]], dim=0),
                    "precursor_mz_raw": 100.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
                {
                    "spectra": torch.stack([spectra[1], intensity[1]], dim=0),
                    "precursor_mz_raw": 200.0,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCN",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                },
            ]
        )

        peak_mz = batch["peak_mz"] * 1000.0
        valid = batch["peak_valid_mask"]
        precursor = batch["precursor_mz"] * 1000.0

        self.assertEqual(int(valid[0].sum().item()), 1)
        self.assertTrue(torch.allclose(peak_mz[0, :1], torch.tensor([90.0])))
        self.assertEqual(int(valid[1].sum().item()), 3)
        self.assertTrue(
            torch.allclose(peak_mz[1, :3], torch.tensor([50.0, 60.0, 194.0]))
        )
        self.assertEqual(
            int((valid & (peak_mz > (precursor.unsqueeze(1) - 5.0))).sum().item()),
            0,
        )
        self.assertGreaterEqual(
            float(batch["peak_intensity"][valid].min().item()),
            0.01 - 1e-6,
        )

    def test_probe_collator_keeps_num_peaks(self):
        collator = massspec_probe_data._ProbeBatchCollator(
            num_peaks=4,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0,
        )
        spectra = torch.zeros((2, 128), dtype=torch.float32)
        spectra[0, :6] = torch.tensor([90.0, 100.0, 110.0, 120.0, 130.0, 140.0])
        spectra[1, :6] = torch.tensor([190.0, 200.0, 210.0, 220.0, 230.0, 240.0])
        intensity = torch.zeros((2, 128), dtype=torch.float32)
        intensity[:, :6] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        samples = []
        for idx, precursor_mz in enumerate((150.0, 250.0)):
            samples.append(
                {
                    "spectra": torch.stack([spectra[idx], intensity[idx]], dim=0),
                    "precursor_mz_raw": precursor_mz,
                    "fingerprint": torch.zeros(1024, dtype=torch.int32),
                    "smiles": "CCO",
                    "adduct_id": 0,
                    "instrument_type_id": 0,
                    "collision_energy": 0.0,
                    "collision_energy_present": 0,
                    "charge": 1.0,
                    "probe_valid_mol": True,
                    "probe_maccs": torch.zeros(166, dtype=torch.int32),
                    "probe_morgan": torch.zeros(4096, dtype=torch.int32),
                    "probe_mol_weight": 0.0,
                    "probe_logp": 0.0,
                    "probe_num_heavy_atoms": 0.0,
                    "probe_num_rings": 0.0,
                }
            )

        batch = collator(samples)

        self.assertEqual(tuple(batch["peak_mz"].shape), (2, 4))
        self.assertEqual(batch["peak_valid_mask"].sum().item(), 8)


if __name__ == "__main__":
    unittest.main()
