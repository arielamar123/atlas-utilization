"""Drop collision-data files whose run lies outside the MC-modelled ranges.

Every 2024r-pp collision-data rucio dataset (``DAOD_PHYSLITE.<dataset>._<n>``)
holds a single run.  ``data_runs_2024r_pp.json`` maps each dataset to that
run; it was built from EventInfo runNumber of the first and last file of every
dataset.  Collision data is restricted to the MC-modelled runs whether or not
trigger selection is enabled, so both modes read the same files.  Filtering
before batch splitting and ``max_files_to_process`` keeps runs outside the
ranges (for example the early-2015 commissioning runs that start the file
list) from using the file budget.  Datasets missing from the table are kept
with a warning; with trigger selection enabled the per-event run check still
applies to them.  With the Good Runs List enabled, datasets of runs without
any good lumiblock are dropped as well; the per-event GRL check still applies
to the rest.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Collection

from services.parsing import schemas

_DATASET_RUNS_FILE = Path(__file__).with_name("data_runs_2024r_pp.json")
_DATASET_PATTERN = re.compile(r"DAOD_PHYSLITE\.(\d+)\._\d+")


@lru_cache(maxsize=1)
def dataset_runs() -> dict[str, int]:
    """Return {rucio dataset id: run number} for collision-data datasets."""
    with _DATASET_RUNS_FILE.open(encoding="utf-8") as handle:
        return {str(dataset): int(run) for dataset, run in json.load(handle).items()}


def collision_years_for_file(file_path: str) -> set[str] | None:
    """Return the DATA_YEAR_RUN_RANGES years of a file's run, or None if unknown."""
    match = _DATASET_PATTERN.search(str(file_path))
    run = dataset_runs().get(match.group(1)) if match else None
    if run is None:
        return None
    return {year for year, (lo, hi) in schemas.DATA_YEAR_RUN_RANGES.items() if lo <= run <= hi}


def _run_in_ranges(run: int) -> bool:
    return any(lo <= run <= hi for lo, hi in schemas.DATA_YEAR_RUN_RANGES.values())


def filter_collision_files_by_run(
    metadata: dict[str, list[str]],
    runs: dict[str, int] | None = None,
    good_runs: Collection[int] | None = None,
) -> dict[str, list[str]]:
    """Remove files of datasets whose run is outside DATA_YEAR_RUN_RANGES.

    With ``good_runs`` (the Good Runs List runs), also remove files of
    datasets whose run is not in it.
    """
    runs = dataset_runs() if runs is None else runs
    logger = logging.getLogger(__name__)
    filtered = {}
    for key, urls in metadata.items():
        kept, dropped_runs, unknown, unknown_files = [], set(), set(), 0
        bad_runs, bad_files = set(), 0
        for url in urls:
            match = _DATASET_PATTERN.search(url)
            run = runs.get(match.group(1)) if match else None
            if run is None:
                unknown.add(match.group(1) if match else url)
                unknown_files += 1
                kept.append(url)
            elif not _run_in_ranges(run):
                dropped_runs.add(run)
            elif good_runs is not None and run not in good_runs:
                bad_runs.add(run)
                bad_files += 1
            else:
                kept.append(url)
        if dropped_runs:
            logger.info(
                "%s: dropped %d / %d collision-data files from %d runs outside "
                "the MC-modelled ranges %s (runs %d-%d)",
                key, len(urls) - len(kept) - bad_files, len(urls), len(dropped_runs),
                schemas.DATA_YEAR_RUN_RANGES, min(dropped_runs), max(dropped_runs),
            )
        if bad_runs:
            logger.info(
                "%s: dropped %d / %d collision-data files from %d runs without "
                "Good Runs List lumiblocks: %s",
                key, bad_files, len(urls), len(bad_runs), sorted(bad_runs),
            )
        if unknown:
            logger.warning(
                "%s: %d collision-data file(s) from datasets without a known run "
                "are kept; their events are checked by run number during trigger "
                "selection: %s",
                key, unknown_files, sorted(unknown)[:10],
            )
        filtered[key] = kept
    return filtered
