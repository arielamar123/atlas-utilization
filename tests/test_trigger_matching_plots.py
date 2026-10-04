"""Tests for the enabled/disabled collision-data comparison plotter."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("matplotlib")

from plots.trigger_matching_plots.compare import build_comparison


def _write_parsed_file(path: Path, scale: float) -> None:
    with uproot.recreate(path) as root_file:
        root_file["events"] = {
            "Muons_pt": ak.Array([[30_000.0 * scale], [45_000.0 * scale], []]),
            "Jets_pt": ak.Array([[80_000.0], [65_000.0, 40_000.0], [30_000.0]]),
            "nJets": [4, 2, 1],
            "nBJets": [1, 0, 0],
        }


def test_build_comparison_writes_collision_data_validation_plots(tmp_path: Path) -> None:
    enabled = tmp_path / "enabled" / "parsed_data"
    disabled = tmp_path / "disabled" / "parsed_data"
    enabled.mkdir(parents=True)
    disabled.mkdir(parents=True)
    _write_parsed_file(enabled / "parsed_enabled.root", 1.0)
    _write_parsed_file(disabled / "parsed_disabled.root", 0.8)

    output = tmp_path / "plots"
    summary = build_comparison(enabled.parent, disabled.parent, output)

    assert summary["trigger_enabled"]["event_count"] == 3
    assert summary["trigger_disabled"]["event_count"] == 3
    assert summary["trigger_enabled"]["top_control_event_count"] == 1
    assert (output / "summary.json").is_file()
    for name in (
        "01_event_yields.png",
        "02_leading_muon_pt.png",
        "03_all_muon_pt.png",
        "04_jet_multiplicity.png",
        "05_leading_jet_pt.png",
        "06_bjet_multiplicity.png",
        "07_muon_4jet_1b_control_muon_pt.png",
    ):
        assert (output / name).is_file()
