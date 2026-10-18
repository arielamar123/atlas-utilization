"""Focused regression coverage for data/MC trigger-menu selection."""

import pytest

ak = pytest.importorskip("awkward")

from services.parsing import schemas
from services.parsing.event_selection import (
    TriggerInformationUnavailableError,
    apply_trigger_selection,
    evaluate_collision_trigger_selection,
)


E15 = "AnalysisTrigMatch_HLT_e24_lhmedium_L1EM20VH" + schemas.TRIGGER_BRANCH_SUFFIX
M15 = "AnalysisTrigMatch_HLT_mu20_iloose_L1MU15" + schemas.TRIGGER_BRANCH_SUFFIX
E16 = "AnalysisTrigMatch_HLT_e26_lhtight_nod0_ivarloose" + schemas.TRIGGER_BRANCH_SUFFIX
M16 = "AnalysisTrigMatch_HLT_mu26_ivarmedium" + schemas.TRIGGER_BRANCH_SUFFIX
D15 = "HLT_e24_lhmedium_L1EM20VH"
DM15 = "HLT_mu20_iloose_L1MU15"
D16 = "HLT_e26_lhtight_nod0_ivarloose"
DM16 = "HLT_mu26_ivarmedium"


def _events(runs, decisions, matches=None, years=None):
    """Collision events; ``matches`` defaults to a matched lepton wherever a chain fired."""
    matches = decisions if matches is None else matches
    if years is None:
        years = ["2015" if run < 290000 else "2016" for run in runs]
    # The real HLT decoder emits a field for every configured chain; sparse
    # hand-written decisions would instead mean corrupt metadata.
    decoded = {
        chain: [False] * len(runs)
        for menu in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS.values()
        for chains in menu.values()
        for chain in chains
    }
    decoded.update(decisions)
    return ak.Array({
        "event_id": list(range(len(runs))),
        "_dataRunNumber": runs,
        "_dataYear": years,
        "_triggerDecision": decoded,
        "_triggerMatch": {schemas.data_trigger_match_branch(c): v for c, v in matches.items()},
        "Electrons": [[{}] for _ in runs],
        "Muons": [[{}] for _ in runs],
    })


def _selected(events, **kwargs):
    return ak.to_list(apply_trigger_selection(events, parse_mc=False, **kwargs).event_id)


def test_data_selects_the_menu_from_each_event_run_number():
    events = _events(
        [276300, 300000, 276300, 300000],
        {D15: [True, False, False, False], DM16: [False, True, False, False]},
    )

    selected = apply_trigger_selection(events, parse_mc=False)

    assert ak.to_list(selected.event_id) == [0, 1]
    assert "_triggerMatch" not in selected.fields
    assert "_triggerDecision" not in selected.fields
    assert "_dataRunNumber" not in selected.fields


def test_data_requires_both_the_hlt_decision_and_a_matched_lepton():
    decisions = {D15: [True, True, False, False]}
    matches = {D15: [True, False, True, False]}

    # Fired without a matched lepton, or matched while prescaled away: rejected.
    assert _selected(_events([276300] * 4, decisions, matches)) == [0]


def test_scalar_legacy_match_is_treated_as_selected_match_when_no_object_mask_is_available():
    result = evaluate_collision_trigger_selection(_events([276300], {D15: [True]}))
    assert ak.to_list(result.event_trigger_pass) == [True]
    # Legacy callers may supply only scalar event matches. Parsed collision
    # files instead provide the per-object ElementLink masks tested below.
    assert ak.to_list(result.selected_lepton_trigger_pass) == [True]


def test_collision_match_must_point_to_a_lepton_surviving_the_offline_cut():
    events = _events([276300], {D15: [True]}, {D15: [[True, False]]})
    events = ak.with_field(
        events,
        ak.Array([[{"pt": 10_000.0, "eta": 0.1}, {"pt": 30_000.0, "eta": 0.2}]]),
        "Electrons",
    )
    result = evaluate_collision_trigger_selection(
        events, collision_kinematic_cuts={"electrons": {"pt_min": 25_000.0, "eta_max": 2.47}}
    )
    assert ak.to_list(result.event_trigger_pass) == [False]
    assert ak.to_list(result.selected_lepton_trigger_pass) == [False]


def test_fired_data_chain_without_match_branch_is_undetermined():
    events = _events([276300, 276300], {D15: [True, True], DM15: [True, False]}, {DM15: [True, False]})
    with pytest.raises(TriggerInformationUnavailableError, match="insufficient"):
        _selected(events)

    no_matches = ak.Array({
        "event_id": [0], "_dataRunNumber": [276300], "_dataYear": ["2015"], "_triggerDecision": {D15: [True]},
    })
    with pytest.raises(TriggerInformationUnavailableError, match="insufficient"):
        _selected(no_matches)


def test_provenanced_early_run_is_not_rejected_by_an_mc_range():
    events = _events([266904], {D15: [True]}, years=["2015"])
    assert _selected(events, file_path="data15_13TeV.root") == [0]


def test_data_rejects_cross_year_matches_and_empty_lists():
    events = _events(
        [276300, 300000, 276300],
        {D15: [False, True, False], DM16: [True, False, False]},
    )

    assert _selected(events) == []


def test_data_uses_run_number_even_when_random_run_number_is_present():
    events = _events([276300], {DM16: [True]})
    events = ak.with_field(events, ak.Array([300000]), "_runNumber")

    # Collision selection follows data provenance, not RandomRunNumber.
    assert _selected(events) == []


def test_unknown_data_provenance_is_a_coverage_error():
    events = _events([1], {D15: [True]}, years=["2099"])
    with pytest.raises(TriggerInformationUnavailableError, match="insufficient"):
        _selected(events, file_path="data2099.root")


def test_missing_data_trigger_metadata_is_an_explicit_error():
    events = ak.Array({"event_id": [1], "_dataRunNumber": [276300]})
    with pytest.raises(TriggerInformationUnavailableError, match="No readable"):
        apply_trigger_selection(events, parse_mc=False, file_path="data.root")


def test_missing_data_run_number_is_an_error():

    with pytest.raises(TriggerInformationUnavailableError, match="runNumber"):
        apply_trigger_selection(ak.Array({"_triggerDecision": {D15: [True]}}))

    events = _events([276300], {DM16: [True]})
    assert _selected(events) == []


def test_mc_keeps_random_run_number_menu_selection():
    events = ak.Array({
        "event_id": [10, 11],
        "_runNumber": [276300, 300000],
        "_triggerMatch": {E15: [True, False], M16: [False, True]},
    })

    selected = apply_trigger_selection(events, parse_mc=True)

    assert ak.to_list(selected.event_id) == [10, 11]


def test_mc_year_without_chain_branches_fails_events_as_on_master():
    events = ak.Array({
        "event_id": [10, 11],
        "_runNumber": [276300, 300000],
        "_triggerMatch": {E15: [True, True]},
    })

    selected = apply_trigger_selection(events, parse_mc=True)

    assert ak.to_list(selected.event_id) == [10]
