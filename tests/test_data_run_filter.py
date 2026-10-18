"""Collision-data files are pre-filtered by their dataset's run."""

import pytest

from services.parsing.data_run_filter import collision_year_for_file, dataset_runs, filter_collision_files_by_run

BASE = "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/data15_13TeV/"


def _url(dataset, n=1):
    return f"{BASE}DAOD_PHYSLITE.{dataset}._{n:06d}.pool.root.1"


def test_files_outside_mc_ranges_are_not_prefiltered(caplog):
    runs = {"1": 266904, "2": 276262, "3": 300000, "4": 290000}
    metadata = {"2024r-pp": [_url(1), _url(1, 2), _url(2), _url(3), _url(4), _url(5)]}
    caplog.set_level("INFO")

    kept = filter_collision_files_by_run(metadata, runs)["2024r-pp"]

    # Dataset 5 has no known run too: every file remains eligible for the GRL
    # and trigger-metadata checks that happen later.
    assert kept == metadata["2024r-pp"]
    assert "outside the MC-modelled" not in caplog.text
    assert "1 collision-data file(s) from datasets without a known run" in caplog.text


def test_dataset_run_table_covers_early_2015_without_becoming_an_eligibility_cut():
    runs = dataset_runs()
    assert len(runs) >= 300
    # The first datasets of the release are early-2015 commissioning runs.
    assert runs["37001626"] == 266904 and runs["37001628"] == 266919
    assert any(run < 276262 for run in runs.values())
    assert collision_year_for_file(_url(1)) == "2015"
