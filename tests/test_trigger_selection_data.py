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
D15 = "HLT_e24_lhmedium_iloose_L1EM20VH"
DM15 = "HLT_mu20_iloose_L1MU15"
D16 = "HLT_e26_lhtight_nod0_ivarloose"
DM16 = "HLT_mu26_ivarmedium"


def _events(runs, **matches):
    return ak.Array({
        "event_id": list(range(len(runs))),
        "_dataRunNumber": runs,
        "_triggerDecision": matches,
        "Electrons": [[{}] for _ in runs],
        "Muons": [[{}] for _ in runs],
    })


def test_data_selects_the_menu_from_each_event_run_number():
    events = _events(
        [276300, 300000, 276300, 300000],
        **{D15: [True, False, False, False], DM16: [False, True, False, False]},
    )

    selected = apply_trigger_selection(events, parse_mc=False)

    assert ak.to_list(selected.event_id) == [0, 1]
    assert "_triggerMatch" not in selected.fields
    assert "_dataRunNumber" not in selected.fields


def test_early_2015_collision_runs_use_the_2015_menu():
    events = _events([266904], **{D15: [True]})

    selected = apply_trigger_selection(events, parse_mc=False)

    assert ak.to_list(selected.event_id) == [0]


def test_data_rejects_cross_year_matches_and_empty_lists():
    events = _events(
        [276300, 300000, 276300],
        **{D15: [False, True, False], DM16: [True, False, False]},
    )

    selected = apply_trigger_selection(events, parse_mc=False)

    assert len(selected) == 0


def test_data_uses_run_number_even_when_random_run_number_is_present():
    events = _events([276300], **{DM16: [True]})
    events = ak.with_field(events, ak.Array([300000]), "_runNumber")

    with pytest.raises(ValueError, match="2015"):
        apply_trigger_selection(events, parse_mc=False)


def test_unsupported_data_runs_are_rejected_with_diagnostics(caplog):
    events = _events([1], **{D15: [True]})

    assert len(apply_trigger_selection(events, parse_mc=False, file_path="data.root")) == 0
    assert "unsupported run numbers" in caplog.text


def test_missing_data_trigger_metadata_is_an_explicit_error():
    events = ak.Array({"event_id": [1], "_dataRunNumber": [276300]})
    with pytest.raises(TriggerInformationUnavailableError, match="No readable"):
        apply_trigger_selection(events, parse_mc=False, file_path="data.root")


def test_missing_data_run_number_is_an_error():

    with pytest.raises(ValueError, match="runNumber"):
        apply_trigger_selection(ak.Array({"_triggerDecision": {D15: [True]}}))

    events = _events([276300], **{DM16: [True]})
    with pytest.raises(ValueError, match="No applicable trigger-match branches"):
        apply_trigger_selection(events, parse_mc=False)


def test_mc_keeps_random_run_number_menu_selection():
    events = ak.Array({
        "event_id": [10, 11],
        "_runNumber": [276300, 300000],
        "_triggerMatch": {E15: [True, False], M16: [False, True]},
    })

    selected = apply_trigger_selection(events, parse_mc=True)

    assert ak.to_list(selected.event_id) == [10, 11]
