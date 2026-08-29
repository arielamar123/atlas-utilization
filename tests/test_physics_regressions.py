import unittest

import awkward as ak
import math
import os
import tempfile

import numpy as np

from services.calculations.im_calculator import IMCalculator
from services.calculations.physics_calcs import (
    filter_events_by_kinematics,
    filter_events_by_particle_counts,
    find_actual_field_name,
)
from services.parsing.file_parser import FileParser
from services.parsing.threaded_processor import ThreadedFileProcessor
from services.storage.sqlite_shards import (
    SqliteArrayShardWriter,
    list_signatures,
    prune_final_states_below_min_events,
)


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


class ParsingFailureAccountingTests(unittest.TestCase):
    def test_parser_none_result_invokes_error_callback(self):
        class NullParser:
            def parse_file(self, *args, **kwargs):
                return None

        failures = []
        processor = ThreadedFileProcessor(NullParser(), 1, show_progress=False)

        batches = list(processor.process_files(
            ["bad.root"],
            ["events"],
            "2024r-pp",
            on_error=lambda url, error: failures.append((url, error)),
        ))

        self.assertEqual(batches, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "bad.root")


class GlobalFinalStateThresholdTests(unittest.TestCase):
    def test_counts_are_aggregated_across_parsing_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "im.sqlite")
            writer = SqliteArrayShardWriter(path)
            writer.append_array(
                "parsed_chunk1_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                np.arange(6),
            )
            writer.append_array(
                "parsed_chunk2_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                np.arange(6),
            )
            writer.append_array(
                "parsed_chunk1_FS_0e_2m_0j_0g_0t_0b_IM_m0m1",
                np.arange(4),
            )
            writer.close()

            removed = prune_final_states_below_min_events(path, 10)
            signatures = list_signatures(path)

            self.assertEqual(removed, ["_FS_0e_2m_0j_0g_0t_0b"])
            self.assertEqual(len(signatures), 2)
            self.assertTrue(all("_FS_2e_" in sig for sig in signatures))


class MissingCollectionSelectionTests(unittest.TestCase):
    def test_missing_required_collection_rejects_event(self):
        events = ak.zip({"BJets": _particles([1])}, depth_limit=1)

        selected = filter_events_by_particle_counts(
            events,
            {"Electrons": {"min": 1, "max": 4}},
            is_particle_counts_range=True,
        )

        self.assertEqual(len(selected), 0)

    def test_jets_does_not_alias_bjets(self):
        self.assertIsNone(find_actual_field_name(["BJets"], "Jets"))

    def test_requested_isolation_must_exist(self):
        events = ak.zip({"Electrons": _particles([1])}, depth_limit=1)
        with self.assertRaisesRegex(ValueError, "requires missing field"):
            filter_events_by_kinematics(
                events,
                {"Electrons": {"rel_isolation_max": 0.06}},
            )


class BTaggingValidationTests(unittest.TestCase):
    def test_cms_threshold_key_matches_configuration(self):
        jets = _particles([2])
        direct = ak.Array([{
            "Jet_btagDeepFlavB": [0.8, 0.2],
        }])
        objects = {"Jets": jets, "DirectObjects": direct}

        split = FileParser._calculate_btagging_and_split(
            objects, {"btagDeepFlavB": 0.5}
        )

        self.assertEqual(ak.to_list(ak.num(split["BJets"])), [1])
        self.assertEqual(ak.to_list(ak.num(split["Jets"])), [1])

    def test_enabled_tagging_rejects_missing_tagger_data(self):
        with self.assertRaisesRegex(ValueError, "tagger branches are missing"):
            FileParser._calculate_btagging_and_split(
                {"Jets": _particles([1])}, {"DL1d": 2.51}
            )


if __name__ == "__main__":
    unittest.main()
