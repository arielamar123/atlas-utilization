"""Focused regression coverage for data/MC trigger-menu selection."""

import pytest

ak = pytest.importorskip("awkward")

from services.parsing import schemas
from services.parsing.event_selection import (
    TriggerInformationUnavailableError,
    apply_trigger_selection,
)


E15 = "AnalysisTrigMatch_HLT_e24_lhmedium_L1EM20VH" + schemas.TRIGGER_BRANCH_SUFFIX
M15 = "AnalysisTrigMatch_HLT_mu20_iloose_L1MU15" + schemas.TRIGGER_BRANCH_SUFFIX
E16 = "AnalysisTrigMatch_HLT_e26_lhtight_nod0_ivarloose" + schemas.TRIGGER_BRANCH_SUFFIX
M16 = "AnalysisTrigMatch_HLT_mu26_ivarmedium" + schemas.TRIGGER_BRANCH_SUFFIX
D15 = "HLT_e24_lhmedium_L1EM20VH"
DM15 = "HLT_mu20_iloose_L1MU15"
D16 = "HLT_e26_lhtight_nod0_ivarloose"
DM16 = "HLT_mu26_ivarmedium"


def _events(runs, decisions, matches=None):
    """Collision events; ``matches`` defaults to a matched lepton wherever a chain fired."""
    matches = decisions if matches is None else matches
    return ak.Array({
        "event_id": list(range(len(runs))),
        "_dataRunNumber": runs,
        "_triggerDecision": decisions,
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


def test_data_chain_without_match_branch_is_unmatched():
    events = _events([276300, 276300], {D15: [True, True], DM15: [True, False]}, {DM15: [True, False]})
    assert _selected(events) == [0]

    no_matches = ak.Array({
        "event_id": [0], "_dataRunNumber": [276300], "_triggerDecision": {D15: [True]},
    })
    assert _selected(no_matches) == []


def test_data_2015_range_starts_where_mc_starts(caplog):
    events = _events([266904, 276261, 276262], {D15: [True, True, True]})

    assert schemas.DATA_YEAR_RUN_RANGES["2015"][0] == schemas.YEAR_RUN_RANGES["2015"][0]
    assert _selected(events, file_path="data.root") == [2]
    assert "Rejecting 2 / 3 collision-data events from runs outside" in caplog.text


def test_data_rejects_cross_year_matches_and_empty_lists():
    events = _events(
        [276300, 300000, 276300],
        {D15: [False, True, False], DM16: [True, False, False]},
    )

    assert _selected(events) == []


def test_data_uses_run_number_even_when_random_run_number_is_present():
    events = _events([276300], {DM16: [True]})
    events = ak.with_field(events, ak.Array([300000]), "_runNumber")

    with pytest.raises(TriggerInformationUnavailableError, match="2015"):
        apply_trigger_selection(events, parse_mc=False)


def test_unsupported_data_runs_are_rejected_with_diagnostics(caplog):
    events = _events([1], {D15: [True]})

    assert _selected(events, file_path="data.root") == []
    assert "runs outside the MC-modelled ranges" in caplog.text


def test_missing_data_trigger_metadata_is_an_explicit_error():
    events = ak.Array({"event_id": [1], "_dataRunNumber": [276300]})
    with pytest.raises(TriggerInformationUnavailableError, match="No readable"):
        apply_trigger_selection(events, parse_mc=False, file_path="data.root")


def test_missing_data_run_number_is_an_error():

    with pytest.raises(TriggerInformationUnavailableError, match="runNumber"):
        apply_trigger_selection(ak.Array({"_triggerDecision": {D15: [True]}}))

    events = _events([276300], {DM16: [True]})
    with pytest.raises(TriggerInformationUnavailableError, match="No applicable trigger decisions"):
        apply_trigger_selection(events, parse_mc=False)


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
