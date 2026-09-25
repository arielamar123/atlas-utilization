import unittest

import awkward as ak

from services.parsing import schemas
from services.parsing.event_selection import apply_trigger_selection
from services.parsing.file_parser import FileParser


class TriggerParsingTests(unittest.TestCase):
    class _MetadataBranch:
        def __init__(self, values):
            self.values = values

        def array(self, library):
            if library != "ak":
                raise ValueError(f"expected awkward read, got {library}")
            return self.values

    class _RootFile:
        def __init__(self, payload, key=1):
            self.metadata = {
                schemas.DATA_TRIGGER_MENU_KEY_BRANCH:
                    TriggerParsingTests._MetadataBranch(ak.Array([[key]])),
                schemas.DATA_TRIGGER_MENU_PAYLOAD_BRANCH:
                    TriggerParsingTests._MetadataBranch(payload)
            }

        def __getitem__(self, key):
            if key != "MetaData":
                raise KeyError(key)
            return self.metadata

    def test_trigger_metadata_is_not_requested_when_disabled(self):
        trigger_branch = schemas.get_all_trigger_branches()[0]
        tree_branches = {
            trigger_branch,
            schemas.RANDOM_RUN_NUMBER_BRANCH,
        }

        result = FileParser._extract_branches_by_schema(
            tree_branches,
            "2024r-pp",
            enable_trigger_matching=False,
        )

        self.assertNotIn("_triggerMatch", result)
        self.assertNotIn("_runNumber", result)

    def test_trigger_metadata_is_requested_when_enabled(self):
        trigger_branch = schemas.get_all_trigger_branches()[0]
        tree_branches = {
            trigger_branch,
            schemas.RANDOM_RUN_NUMBER_BRANCH,
        }

        result = FileParser._extract_branches_by_schema(
            tree_branches,
            "2024r-pp",
            enable_trigger_matching=True,
        )

        self.assertEqual(result["_triggerMatch"], {trigger_branch: trigger_branch})
        self.assertIn("_triggerRunNumber", result)

    def test_data_trigger_decision_is_requested_when_no_mc_match_branch_exists(self):
        result = FileParser._extract_branches_by_schema(
            {schemas.DATA_TRIGGER_DECISION_BRANCH, schemas.DATA_TRIGGER_SMK_BRANCH},
            "2024r-pp",
            enable_trigger_matching=True,
        )

        self.assertEqual(
            result["_triggerDecision"],
            {schemas.DATA_TRIGGER_DECISION_BRANCH: "_triggerDecision"},
        )

    def test_trigger_matches_support_one_jagged_level(self):
        raw = ak.Array([[1, 2], [], [3]])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_trigger_matches_support_multiple_jagged_levels(self):
        raw = ak.Array([[[1], []], [[]], [[2], [3]]])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_open_data_element_link_vectors_are_collapsed_per_event(self):
        """PHYSLITE serializes matches as vector<vector<ElementLink>>."""
        raw = ak.Array([
            [[{"m_persKey": 1, "m_persIndex": 7}], []],
            [[]],
            [[{"m_persKey": 1, "m_persIndex": 11}]],
        ])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_event_level_trigger_flags_remain_supported(self):
        raw = ak.Array([True, False, True])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_open_data_uses_the_year_in_the_data_file_path(self):
        electron_2015 = "HLT_e24_lhmedium_L1EM20VH"
        electron_2016 = "HLT_e26_lhtight_nod0_ivarloose"
        events = ak.zip({
            "event": [0, 1],
            "_triggerPass": ak.zip({
                electron_2015: [True, False],
                electron_2016: [False, True],
            }),
        }, depth_limit=1)

        selected = apply_trigger_selection(
            events,
            release_year="2024r-pp",
            file_path="root://example.org/data15_13TeV/DAOD_PHYSLITE.root",
        )

        self.assertEqual(ak.to_list(selected["event"]), [0])
        self.assertNotIn("_triggerPass", selected.fields)

    def test_open_data_hlt_bits_use_the_embedded_menu_and_data_year(self):
        # Counter 47 lives in word 1, bit 15; counters are already zero-based.
        payload = ak.Array([["""{
            \"chains\": {
                \"HLT_e24_lhmedium_L1EM20VH\": {\"counter\": \"47\"},
                \"HLT_e60_lhmedium\": {\"counter\": \"40\"},
                \"HLT_e120_lhloose\": {\"counter\": \"58\"},
                \"HLT_mu20_iloose_L1MU15\": {\"counter\": \"243\"},
                \"HLT_mu50\": {\"counter\": \"239\"}
            }
        }"""]])
        ef_passed_physics = ak.Array([
            [0, 1 << 15, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        trigger_match = FileParser._decode_data_trigger_decisions(
            self._RootFile(payload),
            ef_passed_physics,
            ak.Array([1, 1]),
            "2024r-pp",
            "root://example.org/data15_13TeV/DAOD_PHYSLITE.root",
        )
        branch = "HLT_e24_lhmedium_L1EM20VH"

        self.assertEqual(ak.to_list(trigger_match[branch]), [True, False])

    def test_mc_uses_each_event_random_run_number(self):
        electron_2015 = "HLT_e24_lhmedium_L1EM20VH"
        electron_2016 = "HLT_e26_lhtight_nod0_ivarloose"
        events = ak.zip({
            "event": [0, 1, 2],
            "_triggerRunNumber": [280000, 300000, 300000],
            "_triggerPass": ak.zip({
                electron_2015: [True, True, False],
                electron_2016: [False, False, True],
            }),
        }, depth_limit=1)

        selected = apply_trigger_selection(events, release_year="2024r-pp_mc")

        self.assertEqual(ak.to_list(selected["event"]), [0, 2])
        self.assertNotIn("_triggerRunNumber", selected.fields)


if __name__ == "__main__":
    unittest.main()
