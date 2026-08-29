import unittest

import awkward as ak
import math

from services.calculations.im_calculator import IMCalculator
from services.parsing.file_parser import FileParser


def _particles(counts, *, charge=1):
    return ak.Array([
        [
            {
                "pt": 100_000.0,
                "eta": 0.0,
                "phi": 0.0,
                "mass": 0.511,
                "charge": charge,
            }
            for _ in range(count)
        ]
        for count in counts
    ])


class FinalStateAlignmentTests(unittest.TestCase):
    def test_invalid_event_does_not_shift_final_state_mask(self):
        events = ak.zip({"Electrons": _particles([0, 1])}, depth_limit=1)
        calculator = IMCalculator(events, 1, 1, 4, 1, 4)

        selected = calculator.get_events_for_final_state(
            "1e_0m_0j_0g_0t_0b"
        )

        self.assertEqual(ak.to_list(ak.num(selected.Electrons)), [1])


class CombinationSelectionTests(unittest.TestCase):
    def test_two_lepton_combination_is_kept_in_three_lepton_state(self):
        events = ak.zip({"Electrons": _particles([3])}, depth_limit=1)
        calculator = IMCalculator(events, 1, 1, 4, 1, 4)
        combination = {"Electrons": (2, 0)}

        selected = calculator.filter_by_particle_counts(
            events, combination, is_exact_count=True
        )
        sliced = calculator.slice_by_field(selected, combination)

        self.assertEqual(len(selected), 1)
        self.assertEqual(ak.to_list(ak.num(sliced.Electrons)), [2])


class InvariantMassUnitTests(unittest.TestCase):
    def test_parsed_mass_field_is_used_and_atlas_mev_is_converted(self):
        jets = ak.Array([[
            {"pt": 100_000.0, "eta": 0.0, "phi": 0.0, "mass": 10_000.0},
            {"pt": 100_000.0, "eta": 0.0, "phi": math.pi, "mass": 10_000.0},
        ]])
        events = ak.zip({"Jets": jets}, depth_limit=1)
        calculator = IMCalculator(events, 1, 1, 4, 1, 4)

        mass = ak.to_list(calculator.calculate_invariant_mass(events))[0]

        self.assertAlmostEqual(mass, 200.997512, places=5)

    def test_cms_gev_inputs_are_not_divided_by_one_thousand(self):
        muons = ak.Array([[
            {"pt": 50.0, "eta": 0.0, "phi": 0.0, "mass": 0.105},
            {"pt": 50.0, "eta": 0.0, "phi": math.pi, "mass": 0.105},
        ]])
        events = ak.zip({"Muons": muons}, depth_limit=1)
        calculator = IMCalculator(
            events, 1, 1, 4, 1, 4, momentum_scale_to_gev=1.0
        )

        mass = ak.to_list(calculator.calculate_invariant_mass(events))[0]

        self.assertAlmostEqual(mass, 100.0002205, places=5)


class RootBatchIntegrityTests(unittest.TestCase):
    def test_one_failed_basket_rejects_the_whole_file(self):
        class FakeTree:
            def arrays(self, branches, entry_start, entry_stop, library):
                if entry_start == 2:
                    raise OSError("corrupt basket")
                return ak.Array({
                    "E.pt": [[1.0], [2.0]],
                    "E.eta": [[0.0], [0.0]],
                    "E.phi": [[0.0], [0.0]],
                })

        mapping = {
            "Electrons": {
                "E.pt": "pt",
                "E.eta": "eta",
                "E.phi": "phi",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "Incomplete ROOT read"):
            FileParser._read_file_in_batches(
                FakeTree(), set(mapping["Electrons"]), mapping, 4, 2
            )


if __name__ == "__main__":
    unittest.main()
