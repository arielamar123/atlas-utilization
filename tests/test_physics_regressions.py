import unittest

import awkward as ak

from services.calculations.im_calculator import IMCalculator


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


if __name__ == "__main__":
    unittest.main()
