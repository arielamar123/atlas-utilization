"""Z->ll visible cross sections with this branch's collision-data lepton
trigger, compared with ATLAS's own trigger decisions for the same events.

Reference: the 2025e-13tev-beta Open Data for Education ntuples, which ATLAS
made from the same 2024r-pp PHYSLITE collision data with PHYSLITEtoOpenData
(https://gitlab.cern.ch/atlas-outreach-data-tools/physlitetoopendata).  Its
data/config.yaml uses exactly the chains of
``schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS``; ``trigE`` / ``trigM`` are the
TrigDecisionTool decisions (OR of the year's electron / muon chains),
``lep_isTrigMatched`` is the trigger-matching tool's per-lepton match, and
only events in the Good Runs List are kept.

For each reference run the test
1. selects Z->ee and Z->mumu candidates in the reference ntuple, so both
   sides use the same events and ATLAS object definitions;
2. counts the candidates accepted by ATLAS's trigger and by this branch's
   (FileParser + apply_trigger_selection on every PHYSLITE file of the run,
   events joined by eventNumber);
3. divides both counts by the run's GRL luminosity (PHYSLITEtoOpenData
   extras/lumitable_<year>.csv, "LAr Corrected" column) and compares the
   visible cross sections.

With identical events and luminosity, a difference can only come from the
trigger decision.  Each run reads ~1-2 GB over XRootD, so the test is
skipped unless RUN_OPENDATA_XSEC_TEST=1.  Run it in the project container:

    RUN_OPENDATA_XSEC_TEST=1 pytest -s tests/test_trigger_cross_section_vs_atlas.py
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pytest

np = pytest.importorskip("numpy")
ak = pytest.importorskip("awkward")
atom = pytest.importorskip("atlasopenmagic")

from services.parsing.data_run_filter import dataset_runs
from services.parsing.event_selection import apply_trigger_selection
from services.parsing.file_parser import FileParser
from services.parsing.root_io import open_root_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OPENDATA_XSEC_TEST") != "1",
    reason="reads remote Open Data files; set RUN_OPENDATA_XSEC_TEST=1",
)

RELEASE = "2024r-pp"
ODEO_BASE = "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/opendata/"

# run: (GRL luminosity [pb^-1], 2025e 2to4lep data file holding the run)
REFERENCE_RUNS = {
    276336: (0.678930, "ODEO_FEB2025_v0_2to4lep_data15_periodD.2to4lep.root"),
    300418: (5.955910, "ODEO_FEB2025_v0_2to4lep_data16_periodB.2to4lep.root"),
}

# Above the highest single-lepton threshold (2016: e26 / mu26).
LEPTON_PT_MIN_GEV = 27.0
Z_MASS_WINDOW_GEV = (66.0, 116.0)
ELECTRON, MUON = 11, 13

# Allowed relative difference between the two visible cross sections.  Both
# sides use the same chains, and Z leptons are above every trigger plateau,
# so the two should agree event by event.  A residual could come from
# matching differences: this branch matches any offline lepton to a chain
# that passed itself, while ATLAS matches its selected leptons to the OR of
# the year's chains.
RELATIVE_TOLERANCE = 0.01
# Allowed fraction of reference events outside the bounds of
# test_event_trigger_decisions_are_bracketed_by_atlas.
EVENT_DISAGREEMENT_TOLERANCE = 0.001

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


def _triggered_event_numbers(url):
    """Return (all eventNumbers, eventNumbers accepted by the trigger) of one file."""
    for _ in range(READ_RETRIES):
        events = FileParser.parse_file(
            url, ["CollectionTree"], RELEASE,
            enable_trigger_matching=True, parse_mc=False,
        )
        if events is not None:
            break
    else:
        raise RuntimeError(f"could not read {url}")

    with open_root_file(url) as root_file:
        numbers = root_file["CollectionTree"]["EventInfoAuxDyn.eventNumber"].array(library="np")
    assert len(numbers) == len(events), url
    events = ak.with_field(events, numbers.astype(np.uint64), "eventNumber")
    selected = apply_trigger_selection(events, release_year=RELEASE, file_path=url, parse_mc=False)
    return numbers.astype(np.uint64), ak.to_numpy(selected.eventNumber).astype(np.uint64)


def _reference_events(run, file_name):
    branches = [
        "eventNumber", "trigE", "trigM", "lep_type", "lep_charge", "lep_pt",
        "lep_eta", "lep_phi", "lep_isTightID", "lep_isTightIso", "lep_isTrigMatched",
    ]
    with open_root_file(ODEO_BASE + file_name) as root_file:
        tree = root_file["analysis"]
        runs = tree["runNumber"].array(library="np")
        entries = np.nonzero(runs == run)[0]
        assert len(entries), f"run {run} is not in {file_name}"
        start, stop = int(entries.min()), int(entries.max()) + 1
        events = tree.arrays(branches, entry_start=start, entry_stop=stop)
    return events[runs[start:stop] == run]


def _z_candidates(events, flavour):
    """Exactly two tight, isolated, same-flavour, opposite-charge leptons in the Z window."""
    two = events[ak.num(events.lep_pt) == 2]
    lep = two[["lep_type", "lep_charge", "lep_pt", "lep_eta", "lep_phi", "lep_isTightID", "lep_isTightIso"]]
    good = (
        ak.all(abs(lep.lep_type) == flavour, axis=1)
        & (ak.sum(lep.lep_charge, axis=1) == 0)
        & ak.all(lep.lep_pt > LEPTON_PT_MIN_GEV, axis=1)
        & ak.all(lep.lep_isTightID, axis=1)
        & ak.all(lep.lep_isTightIso, axis=1)
    )
    two = two[good]
    pt, eta, phi = (ak.to_numpy(two[f"lep_{v}"]).astype(np.float64) for v in ("pt", "eta", "phi"))
    mass = np.sqrt(2 * pt[:, 0] * pt[:, 1] * (np.cosh(eta[:, 0] - eta[:, 1]) - np.cos(phi[:, 0] - phi[:, 1])))
    lo, hi = Z_MASS_WINDOW_GEV
    return two[(mass > lo) & (mass < hi)]


def _atlas_trigger(events):
    """ATLAS single-lepton trigger: chain OR fired and a lepton of that flavour matched."""
    flavour = abs(events.lep_type)
    matched_e = ak.any(events.lep_isTrigMatched & (flavour == ELECTRON), axis=1)
    matched_mu = ak.any(events.lep_isTrigMatched & (flavour == MUON), axis=1)
    return ak.to_numpy((events.trigE & matched_e) | (events.trigM & matched_mu))


@lru_cache(maxsize=None)
def _run_comparison(run):
    """Return (reference events of the run, eventNumbers this branch's trigger accepts)."""
    with ThreadPoolExecutor(WORKERS) as pool:
        per_file = list(pool.map(_triggered_event_numbers, _physlite_urls(run)))
    all_numbers = np.concatenate([numbers for numbers, _ in per_file])
    ours = np.concatenate([selected for _, selected in per_file])
    reference = _reference_events(run, REFERENCE_RUNS[run][1])
    numbers = ak.to_numpy(reference.eventNumber).astype(np.uint64)
    missing = numbers[~np.isin(numbers, all_numbers)]
    assert not len(missing), f"run {run}: reference events absent from PHYSLITE: {missing[:10]}"
    return reference, ours


@pytest.mark.parametrize("run", sorted(REFERENCE_RUNS))
def test_z_visible_cross_section_matches_atlas_trigger(run):
    luminosity_pb = REFERENCE_RUNS[run][0]
    reference, ours = _run_comparison(run)

    failures = []
    for name, flavour in (("Z->ee", ELECTRON), ("Z->mumu", MUON)):
        candidates = _z_candidates(reference, flavour)
        numbers = ak.to_numpy(candidates.eventNumber).astype(np.uint64)
        atlas_pass = _atlas_trigger(candidates)
        our_pass = np.isin(numbers, ours)
        sigma_atlas = atlas_pass.sum() / luminosity_pb
        sigma_ours = our_pass.sum() / luminosity_pb
        relative = sigma_ours / sigma_atlas - 1
        print(
            f"\nrun {run} {name}: {len(numbers)} candidates, L = {luminosity_pb:.4f} pb^-1\n"
            f"  ATLAS trigger:  {atlas_pass.sum():6d} events  sigma_vis = {sigma_atlas:8.2f} pb  "
            f"(trigger eff. {atlas_pass.mean():.4f})\n"
            f"  this branch:    {our_pass.sum():6d} events  sigma_vis = {sigma_ours:8.2f} pb  "
            f"(trigger eff. {our_pass.mean():.4f})\n"
            f"  relative difference {relative:+.4%}; only ATLAS: {np.sum(atlas_pass & ~our_pass)}, "
            f"only this branch: {np.sum(our_pass & ~atlas_pass)}"
        )
        if abs(relative) > RELATIVE_TOLERANCE:
            failures.append(
                f"{name}: sigma_vis {sigma_ours:.2f} pb vs ATLAS {sigma_atlas:.2f} pb ({relative:+.2%}); "
                f"only ATLAS {numbers[atlas_pass & ~our_pass][:10]}, "
                f"only this branch {numbers[our_pass & ~atlas_pass][:10]}"
            )

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("run", sorted(REFERENCE_RUNS))
def test_event_trigger_decisions_are_bracketed_by_atlas(run):
    """Every reference event of the run, including low-pT leptons near threshold.

    Z candidates almost always fire the trigger, so their cross section is
    insensitive to a selection that accepts too much; most of these events
    fail the trigger.  ATLAS's ``lep_isTrigMatched`` also requires the
    offline lepton to be above the trigger plateau (about 1 GeV over the
    chain threshold), which neither this branch nor its MC path applies, so
    the decisions are compared as bounds:

        ATLAS (decision and matched lepton)  <=  this branch  <=  ATLAS decision
    """
    reference, ours = _run_comparison(run)
    numbers = ak.to_numpy(reference.eventNumber).astype(np.uint64)
    atlas_matched = _atlas_trigger(reference)
    atlas_decision = ak.to_numpy(reference.trigE | reference.trigM)
    our_pass = np.isin(numbers, ours)
    lost = atlas_matched & ~our_pass
    not_fired = our_pass & ~atlas_decision
    print(
        f"\nrun {run}: {len(numbers)} reference events; ATLAS decision {atlas_decision.sum()}, "
        f"ATLAS decision+match {atlas_matched.sum()}, this branch {our_pass.sum()}\n"
        f"  accepted by ATLAS but not this branch: {lost.sum()}; "
        f"accepted by this branch without an ATLAS decision: {not_fired.sum()}; "
        f"ATLAS decision without a match here: {np.sum(atlas_decision & ~our_pass)}"
    )
    assert atlas_decision.mean() < 0.5, "reference sample cannot tell the trigger decisions apart"
    assert lost.mean() <= EVENT_DISAGREEMENT_TOLERANCE, f"lost events {numbers[lost][:10]}"
    assert not_fired.mean() <= EVENT_DISAGREEMENT_TOLERANCE, f"not-fired events {numbers[not_fired][:10]}"
