import unittest

import awkward as ak

from services.parsing import schemas
from services.parsing.file_parser import FileParser


class TriggerParsingTests(unittest.TestCase):
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
        self.assertIn("_runNumber", result)

    def test_trigger_matches_support_one_jagged_level(self):
        raw = ak.Array([[1, 2], [], [3]])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_trigger_matches_support_multiple_jagged_levels(self):
        raw = ak.Array([[[1], []], [[]], [[2], [3]]])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])

    def test_event_level_trigger_flags_remain_supported(self):
        raw = ak.Array([True, False, True])

        result = FileParser._collapse_trigger_matches(raw)

        self.assertEqual(ak.to_list(result), [True, False, True])


if __name__ == "__main__":
    unittest.main()
