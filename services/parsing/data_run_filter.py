"""Collision-data provenance and optional GRL-aware file prefiltering.

Recorded data is never filtered with MC ``RandomRunNumber`` ranges.  A known
dataset run may be skipped only when it has no run entry in the selected GRL;
the definitive GRL decision remains per event and lumiblock.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Collection

_DATASET_RUNS_FILE = Path(__file__).with_name("data_runs_2024r_pp.json")
_DATASET_PATTERN = re.compile(r"DAOD_PHYSLITE\.(\d+)\._\d+")


@lru_cache(maxsize=1)
def dataset_runs() -> dict[str, int]:
    """Return {rucio dataset id: run number} for collision-data datasets."""
    with _DATASET_RUNS_FILE.open(encoding="utf-8") as handle:
        return {str(dataset): int(run) for dataset, run in json.load(handle).items()}


def collision_year_for_file(file_path: str) -> str | None:
    """Return the data-taking year encoded by the immutable dataset path.

    This intentionally does not inspect an MC-derived run interval.  Callers
    must treat ``None`` as missing provenance, not as a physical rejection.
    """
    from services.parsing import schemas

    path = str(file_path)
    matches = {
        year for tag, year in schemas.DATA_PROVENANCE_YEARS.items() if tag in path
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def known_run_for_file(file_path: str) -> int | None:
    """Return the cached dataset run when the URI has a known Rucio DID."""
    match = _DATASET_PATTERN.search(str(file_path))
    return dataset_runs().get(match.group(1)) if match else None


def filter_collision_files_by_run(
    metadata: dict[str, list[str]],
    runs: dict[str, int] | None = None,
    good_runs: Collection[int] | None = None,
) -> dict[str, list[str]]:
    """Optionally skip known files whose run has no GRL lumiblocks.

    This is only an I/O optimization.  Unknown datasets and every known run
    when no GRL is supplied are retained, including early and future runs.
    """
    runs = dataset_runs() if runs is None else runs
    logger = logging.getLogger(__name__)
    filtered = {}
    for key, urls in metadata.items():
        kept, unknown, unknown_files = [], set(), 0
        bad_runs, bad_files = set(), 0
        for url in urls:
            match = _DATASET_PATTERN.search(url)
            run = runs.get(match.group(1)) if match else None
            if run is None:
                unknown.add(match.group(1) if match else url)
                unknown_files += 1
                kept.append(url)
            elif good_runs is not None and run not in good_runs:
                bad_runs.add(run)
                bad_files += 1
            else:
                kept.append(url)
        if bad_runs:
            logger.info(
                "%s: dropped %d / %d collision-data files from %d runs without "
                "Good Runs List lumiblocks: %s",
                key, bad_files, len(urls), len(bad_runs), sorted(bad_runs),
            )
        if unknown:
            logger.warning(
                "%s: %d collision-data file(s) from datasets without a known run "
                "are kept; their events are checked by GRL and trigger metadata "
                "during processing: %s",
                key, unknown_files, sorted(unknown)[:10],
            )
        filtered[key] = kept
    return filtered
