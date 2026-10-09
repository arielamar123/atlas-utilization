"""Collision-data files are pre-filtered by their dataset's run."""

import pytest

from services.parsing import schemas
from services.parsing.data_run_filter import dataset_runs, filter_collision_files_by_run

BASE = "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/data15_13TeV/"


def _url(dataset, n=1):
    return f"{BASE}DAOD_PHYSLITE.{dataset}._{n:06d}.pool.root.1"


def test_files_of_runs_outside_mc_ranges_are_dropped_before_the_file_budget(caplog):
    runs = {"1": 266904, "2": 276262, "3": 300000, "4": 290000}
    metadata = {"2024r-pp": [_url(1), _url(1, 2), _url(2), _url(3), _url(4), _url(5)]}
    caplog.set_level("INFO")

    kept = filter_collision_files_by_run(metadata, runs)["2024r-pp"]

    # Dataset 5 has no known run: kept and checked per event later.
    assert kept == [_url(2), _url(3), _url(5)]
    assert "dropped 3 / 6 collision-data files from 2 runs" in caplog.text
    assert "1 collision-data file(s) from datasets without a known run" in caplog.text


def test_dataset_run_table_covers_the_release_and_its_ranges():
    runs = dataset_runs()
    assert len(runs) >= 300
    # The first datasets of the release are early-2015 commissioning runs.
    assert runs["37001626"] == 266904 and runs["37001628"] == 266919
    lo15 = schemas.DATA_YEAR_RUN_RANGES["2015"][0]
    assert any(run < lo15 for run in runs.values())
    assert any(run >= lo15 for run in runs.values())
