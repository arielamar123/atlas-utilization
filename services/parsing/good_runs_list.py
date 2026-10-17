"""Good Runs List (GRL) for the 2015/2016 collision data of the 2024r-pp release.

The 2024r-pp PHYSLITE collision data is not GRL-filtered: it also holds
lumiblocks with detector or data-quality problems.  ATLAS requires analyses
to use only GRL lumiblocks, and the per-run luminosity of the release refers
to them only.  The XML files next to this module are the standard
PHYS_StandardGRL_All_Good_25ns lists that ATLAS's PHYSLITEtoOpenData applies
(copies from atlas-outreach-data-tools/notebooks-collection-opendata,
for-research/limitations).  MC has no GRL.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from services.parsing import schemas

GRL_FILES = tuple(
    Path(__file__).with_name(name)
    for name in (
        "data15_13TeV.periodAllYear_DetStatus-v89-pro21-02_Unknown_PHYS_StandardGRL_All_Good_25ns.xml",
        "data16_13TeV.periodAllYear_DetStatus-v89-pro21-01_DQDefects-00-02-04_PHYS_StandardGRL_All_Good_25ns.xml",
    )
)

# Collision-data releases the GRL files above describe.
GRL_RELEASES = frozenset({"2024r-pp"})

# Lumiblock numbers are far below 2**32, so (run, lumiblock) packs into one
# sortable 64-bit key.
_LUMIBLOCK_BITS = 32


def release_has_grl(release_year: str) -> bool:
    return schemas.normalize_release_year(release_year) in GRL_RELEASES


def parse_grl_xml(path: Path) -> dict[int, list[tuple[int, int]]]:
    """Return {run: [(first lumiblock, last lumiblock), ...]} (inclusive ranges)."""
    good: dict[int, list[tuple[int, int]]] = {}
    for collection in ET.parse(path).getroot().iter("LumiBlockCollection"):
        run = int(collection.findtext("Run"))
        ranges = good.setdefault(run, [])
        for lb_range in collection.iter("LBRange"):
            ranges.append((int(lb_range.get("Start")), int(lb_range.get("End"))))
    return good


@lru_cache(maxsize=1)
def good_lumiblocks() -> Mapping[int, tuple[tuple[int, int], ...]]:
    """Merged GRL of every file in GRL_FILES."""
    merged: dict[int, list[tuple[int, int]]] = {}
    for path in GRL_FILES:
        for run, ranges in parse_grl_xml(path).items():
            merged.setdefault(run, []).extend(ranges)
    return {run: tuple(sorted(ranges)) for run, ranges in merged.items()}


def good_run_numbers() -> frozenset[int]:
    return frozenset(good_lumiblocks())


def _range_keys(
    good: Mapping[int, Iterable[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = sorted(
        (run, start, end) for run, ranges in good.items() for start, end in ranges
    )
    runs = np.array([b[0] for b in bounds], dtype=np.uint64)
    starts = np.array([b[1] for b in bounds], dtype=np.uint64)
    ends = np.array([b[2] for b in bounds], dtype=np.uint64)
    return (runs << _LUMIBLOCK_BITS) | starts, (runs << _LUMIBLOCK_BITS) | ends


def in_good_lumiblocks(
    runs: np.ndarray,
    lumiblocks: np.ndarray,
    good: Mapping[int, Iterable[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Per event: True if its (run, lumiblock) lies in a GRL lumiblock range."""
    good = good_lumiblocks() if good is None else good
    keys = (np.asarray(runs, dtype=np.uint64) << _LUMIBLOCK_BITS) | np.asarray(
        lumiblocks, dtype=np.uint64
    )
    starts, ends = _range_keys(good)
    if not len(starts):
        return np.zeros(len(keys), dtype=bool)
    # GRL ranges do not overlap, so only the last range starting at or
    # before an event's key can contain it.
    index = np.searchsorted(starts, keys, side="right") - 1
    return (index >= 0) & (keys <= ends[np.maximum(index, 0)])
