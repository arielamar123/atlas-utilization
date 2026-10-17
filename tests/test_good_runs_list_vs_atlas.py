"""The Good Runs List selection agrees with ATLAS's GRL-filtered Open Data.

Reference: the 2025e-13tev-beta Open Data for Education ntuples, which ATLAS
made from the same 2024r-pp PHYSLITE collision data with PHYSLITEtoOpenData
and the same GRL files (only GRL lumiblocks are kept).  For each reference
run, every education-ntuple event must survive this pipeline's GRL selection
of the run's PHYSLITE events (joined by eventNumber), and a run without GRL
lumiblocks must be absent from the ntuples.  Run 276336 has PHYSLITE events
in bad lumiblocks (16-35), so the selection is not trivially "keep
everything"; the PHYSLITE files of run 300418 hold only its GRL lumiblocks.

Reads only EventInfo branches (plus one full FileParser pass per run to check
the parser's run/lumiblock read) over XRootD, so the test is skipped unless
RUN_OPENDATA_GRL_TEST=1.  Run it in the project container:

    RUN_OPENDATA_GRL_TEST=1 pytest -s tests/test_good_runs_list_vs_atlas.py
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

np = pytest.importorskip("numpy")
ak = pytest.importorskip("awkward")
atom = pytest.importorskip("atlasopenmagic")

from services.parsing import schemas
from services.parsing.data_run_filter import dataset_runs
from services.parsing.event_selection import apply_good_runs_list
from services.parsing.file_parser import FileParser
from services.parsing.good_runs_list import good_run_numbers
from services.parsing.root_io import open_root_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OPENDATA_GRL_TEST") != "1",
    reason="reads remote Open Data files; set RUN_OPENDATA_GRL_TEST=1",
)

RELEASE = "2024r-pp"
ODEO_BASE = "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/opendata/"
EVENT_NUMBER_BRANCH = "EventInfoAuxDyn.eventNumber"

# run: 2025e 2to4lep data file holding the run.
# GRL lumiblocks: 276336 4-15, 36-54; 300418 281-310 only.
REFERENCE_RUNS = {
    276336: "ODEO_FEB2025_v0_2to4lep_data15_periodD.2to4lep.root",
    300418: "ODEO_FEB2025_v0_2to4lep_data16_periodB.2to4lep.root",
}
# Runs whose PHYSLITE files hold events outside the GRL.
RUNS_WITH_BAD_LUMIBLOCKS = {276336}
# A 2015 period-D run inside the MC-modelled range without GRL lumiblocks.
RUN_WITHOUT_GRL = (276330, "ODEO_FEB2025_v0_2to4lep_data15_periodD.2to4lep.root")

READ_RETRIES = 3
WORKERS = 4


def _physlite_urls(run):
    datasets = {dataset for dataset, dataset_run in dataset_runs().items() if dataset_run == run}
    assert datasets, f"run {run} is not in data_runs_2024r_pp.json"
    atom.set_release(RELEASE)
    urls = [
        url for url in atom.get_urls("data", protocol="root")
        if any(f"DAOD_PHYSLITE.{dataset}._" in url for dataset in datasets)
    ]
    assert urls, f"no PHYSLITE files found for run {run}"
    return urls


def _event_info(url):
    branches = [schemas.DATA_RUN_NUMBER_BRANCH, schemas.DATA_LUMI_BLOCK_BRANCH, EVENT_NUMBER_BRANCH]
    for attempt in range(READ_RETRIES):
        try:
            with open_root_file(url) as root_file:
                arrays = root_file["CollectionTree"].arrays(branches, library="np")
            break
        except OSError:
            if attempt == READ_RETRIES - 1:
                raise
    return ak.Array({
        "_dataRunNumber": arrays[schemas.DATA_RUN_NUMBER_BRANCH],
        "_dataLumiBlock": arrays[schemas.DATA_LUMI_BLOCK_BRANCH],
        "eventNumber": arrays[EVENT_NUMBER_BRANCH].astype(np.uint64),
    })


def _odeo_event_numbers(run, file_name):
    with open_root_file(ODEO_BASE + file_name) as root_file:
        tree = root_file["analysis"]
        runs = tree["runNumber"].array(library="np")
        entries = np.nonzero(runs == run)[0]
        if not len(entries):
            return np.zeros(0, dtype=np.uint64)
        start, stop = int(entries.min()), int(entries.max()) + 1
        numbers = tree["eventNumber"].array(entry_start=start, entry_stop=stop, library="np")
    return numbers[runs[start:stop] == run].astype(np.uint64)


@pytest.mark.parametrize("run", sorted(REFERENCE_RUNS))
def test_every_atlas_grl_event_survives_the_grl_selection(run):
    urls = _physlite_urls(run)
    with ThreadPoolExecutor(WORKERS) as pool:
        events = ak.concatenate(list(pool.map(_event_info, urls)))
    assert np.all(ak.to_numpy(events._dataRunNumber) == run)

    kept = apply_good_runs_list(events, file_path=f"run {run}")
    kept_numbers = ak.to_numpy(kept.eventNumber)
    all_numbers = ak.to_numpy(events.eventNumber)
    dropped_numbers = np.setdiff1d(all_numbers, kept_numbers)
    reference = _odeo_event_numbers(run, REFERENCE_RUNS[run])

    missing = reference[~np.isin(reference, kept_numbers)]
    print(
        f"\nrun {run}: {len(all_numbers)} PHYSLITE events, {len(kept_numbers)} in the GRL "
        f"({len(dropped_numbers)} dropped); {len(reference)} education-ntuple events, "
        f"{len(missing)} of them dropped by the GRL selection"
    )
    assert len(reference)
    assert bool(len(dropped_numbers)) == (run in RUNS_WITH_BAD_LUMIBLOCKS)
    assert not len(missing), f"run {run}: ATLAS GRL events dropped: {missing[:10]}"


@pytest.mark.parametrize("run", sorted(REFERENCE_RUNS))
def test_file_parser_reads_the_lumiblocks_used_by_the_grl(run):
    url = _physlite_urls(run)[0]
    for _ in range(READ_RETRIES):
        parsed = FileParser.parse_file(url, ["CollectionTree"], RELEASE, parse_mc=False, apply_grl=True)
        if parsed is not None:
            break
    else:
        raise RuntimeError(f"could not read {url}")

    direct = _event_info(url)
    assert ak.to_list(parsed._dataRunNumber) == ak.to_list(direct._dataRunNumber)
    assert ak.to_list(parsed._dataLumiBlock) == ak.to_list(direct._dataLumiBlock)
    assert len(apply_good_runs_list(parsed)) == len(apply_good_runs_list(direct))


def test_run_without_grl_lumiblocks_is_absent_from_atlas_ntuples():
    run, file_name = RUN_WITHOUT_GRL
    assert run not in good_run_numbers()
    assert dataset_runs() and run in dataset_runs().values()
    assert not len(_odeo_event_numbers(run, file_name))
