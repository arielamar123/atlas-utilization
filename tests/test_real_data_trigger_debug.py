"""Regression coverage for the removable collision-trigger debug module."""

import awkward as ak

from services.parsing import schemas
from plots.trigger_debug.real_data_trigger_debug import RealDataTriggerDebugger


def test_trigger_debug_text_and_histogram_share_counts(tmp_path):
    events = ak.Array({
        "_dataRunNumber": [276300, 276300, 300000],
        "_triggerDecision": {
            "HLT_e24_lhmedium_L1EM20VH": [True, True, False],
            "HLT_mu26_ivarmedium": [False, False, True],
        },
        # The second event fired e24 but has no matched electron: not retained.
        "_triggerMatch": {
            schemas.data_trigger_match_branch("HLT_e24_lhmedium_L1EM20VH"): [True, False, False],
            schemas.data_trigger_match_branch("HLT_mu26_ivarmedium"): [False, False, True],
        },
        "Electrons": [
            [{"pt": 30_000.0}], [{"pt": 18_000.0}], [],
        ],
        "Muons": [
            [], [], [{"pt": 32_000.0}],
        ],
    })
    debugger = RealDataTriggerDebugger(tmp_path, bins=5, pt_max_gev=50)
    debugger.add_events(events)
    rows = {row["chain"]: row for row in debugger.write()}

    electron = rows["HLT_e24_lhmedium_L1EM20VH"]
    assert (electron["relevant_events"], electron["passed_chain"], electron["did_not_pass_chain"]) == (2, 1, 1)
    muon = rows["HLT_mu26_ivarmedium"]
    assert (muon["relevant_events"], muon["passed_chain"], muon["did_not_pass_chain"]) == (1, 1, 0)
    assert (tmp_path / "trigger_counts.txt").read_text(encoding="utf-8").count("fired & matched / retained: 1") >= 2
    assert (tmp_path / "HLT_e24_lhmedium_L1EM20VH_leading_pt.png").exists()
    assert (tmp_path / "HLT_mu26_ivarmedium_response.png").exists()
