"""Inspect and smoke-test 2024 ATLAS Open Data trigger decoding.

Example:
  python scripts/smoke_open_data_triggers.py URL_FOR_DATA15 URL_FOR_DATA16
"""

from __future__ import annotations

import argparse
import logging

import awkward as ak
import uproot

from services.parsing import schemas
from services.parsing.event_selection import apply_trigger_selection
from services.parsing.file_parser import FileParser


def _chain_counts(trigger_pass: ak.Array, year: str) -> tuple[int, int, int]:
    electron = ak.zeros_like(trigger_pass[trigger_pass.fields[0]], dtype=bool)
    muon = ak.zeros_like(electron)
    for chain in schemas.SINGLE_LEPTON_TRIGGER_CHAINS[year]["Electrons"]:
        if chain in trigger_pass.fields:
            electron = electron | trigger_pass[chain]
    for chain in schemas.SINGLE_LEPTON_TRIGGER_CHAINS[year]["Muons"]:
        if chain in trigger_pass.fields:
            muon = muon | trigger_pass[chain]
    return (
        int(ak.sum(electron & ~muon)),
        int(ak.sum(muon & ~electron)),
        int(ak.sum(electron & muon)),
    )


def smoke(url: str, max_events: int) -> None:
    # EOS HTTPS currently presents a local-intercept certificate in this
    # environment; this is only for the public structural validation script.
    with uproot.open(url, ssl=False, timeout=60) as root_file:
        tree = root_file["CollectionTree"]
        branches = set(tree.keys())
        metadata = root_file["MetaData"]
        metadata_branches = set(metadata.keys())
        configured_matching = set(schemas.get_all_trigger_branches())
        print(f"\nFILE: {url}")
        print("Tree branches:")
        for branch in sorted(branches):
            if (
                "TrigDecision" in branch
                or branch.endswith("runNumber")
                or branch in configured_matching
            ):
                print(f"  {branch}")
        print("Menu metadata branches:")
        for branch in sorted(metadata_branches):
            if "TriggerMenuJson_HLT" in branch:
                print(f"  {branch}")

        year = schemas.get_trigger_years("2024r-pp", url)[0]
        requested = FileParser._extract_branches_by_schema(
            branches, "2024r-pp", enable_trigger_matching=True, file_path=url
        )
        decision_branch = next(iter(requested.get("_triggerDecision", {})), None)
        smk_branch = next(iter(requested.get("_triggerSmk", {})), None)
        run_branch = next(iter(requested.get("_triggerRunNumber", {})), None)
        if not decision_branch or not smk_branch:
            raise RuntimeError("No usable xTrigDecision fallback branches were found")
        arrays = tree.arrays(
            [decision_branch, smk_branch, run_branch], entry_stop=max_events, library="ak"
        )
        trigger_pass = FileParser._decode_data_trigger_decisions(
            root_file,
            arrays[decision_branch], arrays[smk_branch], "2024r-pp", url, arrays[run_branch],
        )
        events = ak.zip({
            "_triggerPass": trigger_pass,
            "_triggerRunNumber": arrays[run_branch],
            "_triggerSource": ak.Array(["xTrigDecision"] * len(trigger_pass)),
        }, depth_limit=1)
        electron_only, muon_only, both = _chain_counts(trigger_pass, year)
        selected = apply_trigger_selection(events, "2024r-pp", url)
        print(
            f"source=xTrigDecision year={year} raw={len(events)} passing={len(selected)} "
            f"electron_only={electron_only} muon_only={muon_only} both={both}"
        )
        print(f"unique_smks={sorted(set(ak.to_list(arrays[smk_branch])))}")
        print(f"configured_chains_present={[chain for chain in trigger_pass.fields]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+", help="2015 and 2016 public PHYSLITE URLs")
    parser.add_argument("--max-events", type=int, default=1000)
    args = parser.parse_args()
    for url in args.urls:
        smoke(url, args.max_events)


if __name__ == "__main__":
    main()
