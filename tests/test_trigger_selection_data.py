"""Focused regression coverage for data/MC trigger-menu selection."""

import pytest

ak = pytest.importorskip("awkward")

from services.parsing import schemas
from services.parsing.event_selection import apply_trigger_selection


E15 = "AnalysisTrigMatch_HLT_e24_lhmedium_L1EM20VH" + schemas.TRIGGER_BRANCH_SUFFIX
M15 = "AnalysisTrigMatch_HLT_mu20_iloose_L1MU15" + schemas.TRIGGER_BRANCH_SUFFIX
E16 = "AnalysisTrigMatch_HLT_e26_lhtight_nod0_ivarloose" + schemas.TRIGGER_BRANCH_SUFFIX
M16 = "AnalysisTrigMatch_HLT_mu26_ivarmedium" + schemas.TRIGGER_BRANCH_SUFFIX


def _events(runs, **matches):
    return ak.Array({
        "event_id": list(range(len(runs))),
        "_dataRunNumber": runs,
        "_triggerMatch": matches,
    })


def test_data_selects_the_menu_from_each_event_run_number():
    events = _events(
        [276300, 300000, 276300, 300000],
        **{E15: [True, False, False, False], M16: [False, True, False, False]},
    )

    selected = apply_trigger_selection(events, parse_mc=False)

    assert ak.to_list(selected.event_id) == [0, 1]
    assert "_triggerMatch" not in selected.fields
    assert "_dataRunNumber" not in selected.fields


def test_data_rejects_cross_year_matches_and_empty_lists():
    events = _events(
        [276300, 300000, 276300],
        **{E15: [False, True, False], M16: [True, False, False]},
    )

    selected = apply_trigger_selection(events, parse_mc=False)

    assert len(selected) == 0


def test_data_uses_run_number_even_when_random_run_number_is_present():
    events = _events([276300], **{M16: [True]})
    events = ak.with_field(events, ak.Array([300000]), "_runNumber")

    assert len(apply_trigger_selection(events, parse_mc=False)) == 0


def test_unsupported_data_runs_are_rejected_with_diagnostics(caplog):
    events = _events([1], **{E15: [True]})

    assert len(apply_trigger_selection(events, parse_mc=False, file_path="data.root")) == 0
    assert "unsupported run numbers" in caplog.text


def test_missing_data_trigger_metadata_is_explicit_and_non_fatal(caplog):
    events = ak.Array({"event_id": [1], "_dataRunNumber": [276300]})
    selected = apply_trigger_selection(events, parse_mc=False, file_path="data.root")
    assert ak.to_list(selected.event_id) == [1]
    assert "continuing without trigger filtering" in caplog.text


def test_missing_data_run_number_is_an_error():

    with pytest.raises(ValueError, match="runNumber"):
        apply_trigger_selection(ak.Array({"_triggerMatch": {E15: [True]}}))

    events = _events([276300], **{M16: [True]})
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
