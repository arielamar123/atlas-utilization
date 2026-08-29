import unittest

import awkward as ak
import math
import os
import tempfile
import threading
import time
import warnings
from unittest import mock

import numpy as np

from services.calculations.im_calculator import IMCalculator
from services.calculations.physics_calcs import (
    calc_inv_mass,
    filter_events_by_kinematics,
    filter_events_by_particle_counts,
    find_actual_field_name,
)
from services.parsing.file_parser import FileParser
from services.parsing.event_selection import normalize_particle_collections
from services.parsing.threaded_processor import ThreadedFileProcessor
from services.storage.sqlite_shards import (
    SqliteArrayShardWriter,
    list_signatures,
    prune_final_states_below_min_events,
)
from services.pipelines.im_pipeline import _apply_ossf_dilepton_cut
from orchestration.handlers.parsing_handler import select_metadata_for_parsing
from orchestration.handlers.mass_calculation_handler import MassCalculationHandler
from services.metadata.fetcher import MetadataFetcher
from domain.events import EventBatch
from domain.events import EventChunk
from domain.config import PipelineConfig
from services.parsing.event_accumulator import EventAccumulator


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


class ConfigDrivenParticleCollectionTests(unittest.TestCase):
    def test_atlas_muons_do_not_require_an_optional_mass_branch(self):
        muons = ak.Array([[
            {"pt": 50_000.0, "eta": 0.0, "phi": 0.0, "charge": -1},
            {"pt": 50_000.0, "eta": 0.0, "phi": math.pi, "charge": 1},
        ]])

        normalized = normalize_particle_collections(
            ak.zip({"Muons": muons}, depth_limit=1),
            ("Muons",),
            "2024r-pp",
        )
        calculator = IMCalculator(normalized, 1, 1, 4, 1, 4)

        self.assertEqual(normalized.Muons.fields, ["pt", "eta", "phi", "charge"])
        self.assertAlmostEqual(
            ak.to_list(calculator.calculate_invariant_mass(normalized))[0],
            100.0002205,
            places=5,
        )

    def test_mixed_file_schemas_keep_every_requested_collection(self):
        jets = ak.Array([[{
            "pt": 100_000.0, "eta": 0.0, "phi": 0.0, "mass": 10_000.0
        }]])
        leptons = _particles([1], charge=-1)
        requested = ("Electrons", "Muons", "Jets", "BJets", "Photons")

        first = normalize_particle_collections(
            ak.zip({"Jets": jets, "Taus": jets}, depth_limit=1),
            requested,
            "2024r-pp",
        )
        second = normalize_particle_collections(
            ak.zip(
                {"Electrons": leptons, "Muons": leptons, "Jets": jets},
                depth_limit=1,
            ),
            requested,
            "2024r-pp",
        )
        chunk = EventChunk.from_batches(
            [
                EventBatch(first, 0, "2024r-pp", first.layout.nbytes, 1, 0.1),
                EventBatch(second, 1, "2024r-pp", second.layout.nbytes, 1, 0.1),
            ],
            chunk_index=0,
            release_year="2024r-pp",
        )

        self.assertEqual(chunk.events.fields, list(requested))
        self.assertNotIn("Taus", chunk.events.fields)
        self.assertEqual(ak.to_list(ak.num(chunk.events.Electrons)), [0, 1])
        self.assertEqual(ak.to_list(ak.num(chunk.events.Muons)), [0, 1])

    def test_missing_requested_collection_is_typed_for_root_output(self):
        jets = ak.Array([[{
            "pt": 100_000.0, "eta": 0.0, "phi": 0.0, "mass": 10_000.0
        }]])
        normalized = normalize_particle_collections(
            ak.zip({"Jets": jets}, depth_limit=1),
            ("Electrons", "Jets"),
            "2024r-pp",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "normalized.root")
            import uproot
            with uproot.recreate(path) as root_file:
                root_file["events"] = {
                    field: normalized[field] for field in normalized.fields
                }
            with uproot.open(path) as root_file:
                branches = set(root_file["events"].keys())

        self.assertIn("Electrons_pt", branches)
        self.assertIn("Electrons_charge", branches)
        self.assertIn("Jets_pt", branches)

    def test_config_parser_loads_objects_for_parsing_only_runs(self):
        config = {
            "tasks": {"do_parsing": True},
            "parsing_task_config": {
                "output_path": "/tmp/parsed",
                "file_urls_path": "/tmp/files.json",
                "jobs_logs_path": "/tmp/logs",
            },
            "mass_calculation_task_config": {
                "input_dir": "/tmp/parsed",
                "output_dir": "/tmp/masses",
                "objects_to_calculate": ["Muons", "Jets"],
            },
        }

        parsed = PipelineConfig.from_dict(config)

        self.assertEqual(
            parsed.mass_calculation_config.objects_to_calculate,
            ("Muons", "Jets"),
        )


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

    def test_legacy_calculator_uses_one_consistent_unit(self):
        jets = ak.Array([[
            {"pt": 100_000.0, "eta": 0.0, "phi": 0.0, "mass": 10_000.0},
            {"pt": 100_000.0, "eta": 0.0, "phi": math.pi, "mass": 10_000.0},
        ]])
        events = ak.zip({"Jets": jets}, depth_limit=1)

        mass = ak.to_list(calc_inv_mass(events))[0]

        self.assertAlmostEqual(mass, 200.997512, places=5)


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
    def test_long_file_emits_active_worker_heartbeat(self):
        class SlowParser:
            def parse_file(self, *args, **kwargs):
                time.sleep(0.04)
                return ak.zip({"Jets": _particles([1])}, depth_limit=1)

        processor = ThreadedFileProcessor(
            SlowParser(),
            1,
            show_progress=False,
            file_read_timeout_sec=1,
            status_interval_sec=0.01,
        )

        with self.assertLogs(level="INFO") as captured:
            batches = list(processor.process_files(
                ["large.root"], ["events"], "2024r-pp"
            ))

        self.assertEqual(len(batches), 1)
        self.assertTrue(any(
            "Still parsing 1 active file(s): large.root" in line
            for line in captured.output
        ))

    def test_file_read_timeout_is_forwarded_to_uproot(self):
        parsed = ak.zip({"Jets": _particles([1])}, depth_limit=1)
        fake_file = mock.MagicMock()
        fake_context = mock.MagicMock()
        fake_context.__enter__.return_value = fake_file

        with mock.patch("services.parsing.file_parser.uproot.open", return_value=fake_context) as opened:
            with mock.patch.object(FileParser, "_parse_opened_file", return_value=parsed):
                result = FileParser.parse_file(
                    "remote.root",
                    ["events"],
                    "2024r-pp",
                    read_timeout_sec=7.5,
                )

        self.assertIs(result, parsed)
        opened.assert_called_once_with("remote.root", timeout=7.5)

    def test_closing_generator_does_not_submit_the_whole_file_list(self):
        release_worker = threading.Event()
        started = []

        class ControlledParser:
            def parse_file(self, file_path, *args, **kwargs):
                started.append(file_path)
                if file_path != "first.root":
                    release_worker.wait(timeout=2)
                return ak.zip({"Jets": _particles([1])}, depth_limit=1)

        processor = ThreadedFileProcessor(
            ControlledParser(), 2, show_progress=False, file_read_timeout_sec=1
        )
        batches = processor.process_files(
            ["first.root", "second.root", "third.root", "fourth.root"],
            ["events"],
            "2024r-pp",
        )

        next(batches)
        batches.close()
        release_worker.set()

        self.assertLessEqual(len(started), 2)

    def test_configured_read_timeout_reaches_threaded_parser(self):
        class RecordingParser:
            def __init__(self):
                self.kwargs = None

            def parse_file(self, *args, **kwargs):
                self.kwargs = kwargs
                return ak.zip({"Jets": _particles([1])}, depth_limit=1)

        parser = RecordingParser()
        processor = ThreadedFileProcessor(
            parser, 1, show_progress=False, file_read_timeout_sec=12.0
        )

        list(processor.process_files(["one.root"], ["events"], "2024r-pp"))

        self.assertEqual(parser.kwargs["read_timeout_sec"], 12.0)

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

    def test_counts_are_aggregated_across_batch_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, f"im_batch_{i}.sqlite") for i in (1, 2)]
            for index, path in enumerate(paths, start=1):
                writer = SqliteArrayShardWriter(path)
                writer.record_final_state_count(
                    "2e_0m_0j_0g_0t_0b", 6
                )
                writer.append_array(
                    f"parsed_batch{index}_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                    # Combination-specific cuts may retain fewer entries than
                    # the population threshold must count.
                    np.arange(1),
                )
                writer.close()

            removed = prune_final_states_below_min_events(paths, 10)

            self.assertEqual(removed, [])
            self.assertTrue(all(len(list_signatures(path)) == 1 for path in paths))


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
    def test_zero_dl1d_denominator_has_defined_tagging_without_warning(self):
        jets = _particles([3])
        direct = ak.Array([{
            "AnalysisJetsAuxDyn.btaggingLink/AnalysisJetsAuxDyn.btaggingLink.m_persIndex": [0, 1, 2],
            "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb": [0.8, 0.0, 0.2],
            "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc": [0.0, 0.0, 0.0],
            "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu": [0.0, 0.0, 0.0],
        }])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            split = FileParser._calculate_btagging_and_split(
                {"Jets": jets, "DirectObjects": direct},
                {"DL1d": 2.51},
            )

        self.assertEqual(ak.to_list(ak.num(split["BJets"])), [2])
        self.assertEqual(ak.to_list(ak.num(split["Jets"])), [1])

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


class OSSFZCutTests(unittest.TestCase):
    def test_cut_uses_charge_and_preserves_same_sign_pair(self):
        electrons = ak.Array([
            [
                {"pt": 60.0, "eta": 0.0, "phi": 0.0, "charge": 1},
                {"pt": 60.0, "eta": 0.0, "phi": 3.14, "charge": -1},
            ],
            [
                {"pt": 60.0, "eta": 0.0, "phi": 0.0, "charge": 1},
                {"pt": 60.0, "eta": 0.0, "phi": 3.14, "charge": 1},
            ],
            [
                {"pt": 70.0, "eta": 0.0, "phi": 0.0, "charge": 1},
                {"pt": 70.0, "eta": 0.0, "phi": 3.14, "charge": -1},
            ],
        ])
        events = ak.zip({"Electrons": electrons}, depth_limit=1)

        kept = _apply_ossf_dilepton_cut(
            events,
            {"Electrons": (2, 0)},
            ak.Array([100.0, 100.0, 120.0]),
            115.0,
        )

        self.assertEqual(ak.to_list(kept), [100.0, 120.0])

    def test_multibody_mass_is_not_treated_as_dilepton_mass(self):
        events = ak.zip({
            "Electrons": ak.Array([[
                {"charge": 1}, {"charge": -1},
            ]]),
            "Jets": ak.Array([[{}]]),
        }, depth_limit=1)

        kept = _apply_ossf_dilepton_cut(
            events,
            {"Electrons": (2, 0), "Jets": (1, 0)},
            ak.Array([100.0]),
            115.0,
        )

        self.assertEqual(ak.to_list(kept), [100.0])


class MetadataModeSelectionTests(unittest.TestCase):
    def test_parse_mc_selects_mc_partner_of_base_release(self):
        metadata = {
            "2024r-pp": ["data.root"],
            "2024r-pp_mc": ["mc.root"],
        }

        selected = select_metadata_for_parsing(metadata, ["2024r-pp"], True)

        self.assertEqual(selected, {"2024r-pp_mc": ["mc.root"]})

    def test_data_mode_excludes_mc_partner(self):
        metadata = {
            "2024r-pp": ["data.root"],
            "2024r-pp_mc": ["mc.root"],
        }

        selected = select_metadata_for_parsing(metadata, ["2024r-pp"], False)

        self.assertEqual(selected, {"2024r-pp": ["data.root"]})


class MetadataCompletenessTests(unittest.TestCase):
    def test_one_failed_dataset_rejects_the_release(self):
        fetcher = MetadataFetcher(timeout=1)
        with (
            mock.patch("services.metadata.fetcher.atom.set_release"),
            mock.patch(
                "services.metadata.fetcher.atom.available_datasets",
                return_value=[1, 2],
            ),
            mock.patch(
                "services.metadata.fetcher.atom.get_urls",
                side_effect=[["first.root"], OSError("API failure")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "before release .* was complete"):
                fetcher._fetch_urls_for_releases(["2024r-pp"])


class ReleaseAccumulationTests(unittest.TestCase):
    def test_release_change_flushes_existing_batches(self):
        accumulator = EventAccumulator(1_000_000)
        events = ak.zip({"Electrons": _particles([1])}, depth_limit=1)
        first = EventBatch(events, 1, "release_a", 100, 1, 0.1)
        second = EventBatch(events, 2, "release_b", 100, 1, 0.1)

        self.assertIsNone(accumulator.add_batch(first))
        flushed = accumulator.add_batch(second)
        remaining = accumulator.flush()

        self.assertEqual(flushed.release_year, "release_a")
        self.assertEqual(flushed.file_ids, (1,))
        self.assertEqual(remaining.release_year, "release_b")
        self.assertEqual(remaining.file_ids, (2,))


class CombinedLeptonSchemaTests(unittest.TestCase):
    def test_legacy_leptons_are_split_by_pdg_type(self):
        leptons = ak.Array([[
            {"pt": 30.0, "eta": 0.0, "phi": 0.0, "type": 11},
            {"pt": 40.0, "eta": 0.0, "phi": 1.0, "type": -13},
        ]])

        split = FileParser._split_combined_leptons(
            {"Electrons": leptons, "Muons": leptons}, "2016e-8tev_mc"
        )

        self.assertEqual(ak.to_list(ak.num(split["Electrons"])), [1])
        self.assertEqual(ak.to_list(ak.num(split["Muons"])), [1])
        self.assertEqual(ak.to_list(split["Electrons"]["type"]), [[11]])
        self.assertEqual(ak.to_list(split["Muons"]["type"]), [[-13]])


class MultiDigitFinalStateTests(unittest.TestCase):
    def test_limit_parses_the_complete_multiplicity(self):
        self.assertEqual(
            IMCalculator._limit_particles_in_fs(
                "0e_0m_10j_0g_0t_0b", threshold=4
            ),
            "0e_0m_4j_0g_0t_0b",
        )

    def test_containment_parses_the_complete_multiplicity(self):
        final_state = "0e_0m_10j_0g_0t_0b"
        self.assertTrue(IMCalculator.does_final_state_contain_combination(
            final_state, {"Jets": (4, 6)}
        ))
        self.assertFalse(IMCalculator.does_final_state_contain_combination(
            final_state, {"Jets": (5, 6)}
        ))


class MassFailurePropagationTests(unittest.TestCase):
    def test_any_failed_input_fails_the_mass_stage(self):
        with self.assertRaisesRegex(
            RuntimeError, "Invariant-mass calculation failed for 1 file"
        ):
            MassCalculationHandler._raise_mass_failures([
                ("parsed_bad.root", ValueError("broken event layout"))
            ])


class MassOnlyZConfigurationTests(unittest.TestCase):
    def test_mass_only_batch_retains_z_cutoff_configuration(self):
        config = PipelineConfig.from_dict({
            "tasks": {"do_mass_calculating": True},
            "mass_calculation_task_config": {
                "input_dir": "parsed",
                "output_dir": "im",
            },
            "post_processing_task_config": {
                "input_dir": "im",
                "output_dir": "processed",
                "z_peak_cutoff": 115.0,
            },
        })

        self.assertIsNotNone(config.post_processing_config)
        self.assertEqual(config.post_processing_config.z_peak_cutoff, 115.0)


class BatchedTreeReconstructionTests(unittest.TestCase):
    def test_parsed_particle_branches_are_read_in_one_call(self):
        class FakeTree:
            def __init__(self):
                self.calls = 0

            def keys(self):
                return [
                    "nElectrons",
                    "Electrons_pt",
                    "Electrons_eta",
                    "Electrons_phi",
                    "Electrons_charge",
                ]

            def arrays(self, branches, library):
                self.calls += 1
                return ak.Array({
                    "Electrons_pt": [[30.0]],
                    "Electrons_eta": [[0.1]],
                    "Electrons_phi": [[0.2]],
                    "Electrons_charge": [[-1]],
                })

            def __getitem__(self, branch):
                raise AssertionError("per-branch reads must not be used")

        tree = FakeTree()
        events = MassCalculationHandler._reconstruct_particle_arrays(tree)

        self.assertEqual(tree.calls, 1)
        self.assertEqual(ak.to_list(events.Electrons.charge), [[-1]])


if __name__ == "__main__":
    unittest.main()
