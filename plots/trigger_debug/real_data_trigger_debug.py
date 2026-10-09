"""Generate auditable plots for the readable collision-data trigger bits.

This module deliberately lives outside the production pipeline.  It receives
the same pre-selection event arrays as collision-data trigger selection and
creates one leading-lepton-pT plot per configured trigger chain.  The text
summary and the legends are calculated from the same counters.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import awkward as ak
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from services.parsing import schemas
from services.parsing.file_parser import FileParser


@dataclass(frozen=True)
class TriggerDefinition:
    """Human-readable requirements for one configured collision-data chain."""

    year: str
    flavour: str
    chain: str
    threshold_gev: float
    requirements: str


# These are the chain-name requirements, not extra cuts applied by this tool.
# Keep this table beside the debug plots so removing the diagnostics is a
# single-directory deletion and cannot affect the production selection.
_CHAIN_REQUIREMENTS = {
    "HLT_e24_lhmedium_L1EM20VH": (24, "medium likelihood ID; L1_EM20VH seed"),
    "HLT_e60_lhmedium": (60, "medium likelihood ID"),
    "HLT_e120_lhloose": (120, "loose likelihood ID"),
    "HLT_mu20_iloose_L1MU15": (20, "loose isolation; L1_MU15 seed"),
    "HLT_mu40": (40, "no ID/isolation qualifier in chain name"),
    "HLT_e26_lhtight_nod0_ivarloose": (26, "tight likelihood ID; no d0 requirement; variable-cone loose isolation"),
    "HLT_e60_lhmedium_nod0": (60, "medium likelihood ID; no d0 requirement"),
    "HLT_e140_lhloose_nod0": (140, "loose likelihood ID; no d0 requirement"),
    "HLT_mu26_ivarmedium": (26, "variable-cone medium isolation"),
    "HLT_mu50": (50, "no ID/isolation qualifier in chain name"),
}


def trigger_definitions() -> list[TriggerDefinition]:
    """Return every real-data trigger selected by the production code."""
    definitions = []
    for year, flavours in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS.items():
        for object_name, chains in flavours.items():
            flavour = "electron" if object_name == "Electrons" else "muon"
            for chain in chains:
                threshold, requirements = _CHAIN_REQUIREMENTS[chain]
                definitions.append(
                    TriggerDefinition(year, flavour, chain, threshold, requirements)
                )
    return definitions


def _leading_pt_gev(events: ak.Array, object_name: str) -> np.ndarray:
    """Return leading offline lepton pT in GeV for events containing it."""
    pts_mev = ak.max(events[object_name].pt, axis=1)
    return ak.to_numpy(pts_mev) / 1000.0


class RealDataTriggerDebugger:
    """Accumulate event-level trigger diagnostics and write plots plus counts."""

    def __init__(self, output_dir: str | Path, bins: int = 50, pt_max_gev: float = 200.0):
        self.output_dir = Path(output_dir)
        self.bins = bins
        self.pt_max_gev = pt_max_gev
        self._observations: dict[str, dict[str, list[np.ndarray] | TriggerDefinition]] = {}
        for definition in trigger_definitions():
            self._observations[definition.chain] = {
                "definition": definition, "pass": [], "fail": [],
            }

    def add_events(self, events: ak.Array) -> None:
        """Add parser output with decoded ``_triggerDecision`` fields.

        The denominator is event-based: an event must be in the chain's run
        period and contain at least one offline lepton of its flavour.  An
        event passes a chain as in ``apply_trigger_selection``: the chain's HLT
        physics decision fired and an offline lepton is matched to it.
        """
        required = {"_dataRunNumber", "_triggerDecision"}
        missing = required - set(events.fields)
        if missing:
            raise ValueError(f"Trigger debug input lacks fields: {sorted(missing)}")

        runs = events["_dataRunNumber"]
        decisions = events["_triggerDecision"]
        matches = events["_triggerMatch"] if "_triggerMatch" in events.fields else None
        for observation in self._observations.values():
            definition = observation["definition"]
            assert isinstance(definition, TriggerDefinition)
            object_name = "Electrons" if definition.flavour == "electron" else "Muons"
            if definition.chain not in decisions.fields or object_name not in events.fields:
                continue
            low, high = schemas.DATA_YEAR_RUN_RANGES[definition.year]
            relevant = (runs >= low) & (runs <= high) & (ak.num(events[object_name]) > 0)
            if not bool(ak.any(relevant)):
                continue
            relevant_events = events[relevant]
            pt = _leading_pt_gev(relevant_events, object_name)
            fired = ak.to_numpy(relevant_events["_triggerDecision"][definition.chain])
            branch = schemas.data_trigger_match_branch(definition.chain)
            if matches is not None and branch in matches.fields:
                fired = fired & ak.to_numpy(relevant_events["_triggerMatch"][branch])
            else:
                fired = np.zeros(len(fired), dtype=bool)
            observation["pass"].append(pt[fired])
            observation["fail"].append(pt[~fired])

    def save_state(self, path: str | Path) -> None:
        """Persist the raw pass/fail pT values so batch jobs can be merged."""
        arrays = {}
        for chain, observation in self._observations.items():
            arrays[f"{chain}__pass"] = self._joined(observation["pass"])
            arrays[f"{chain}__fail"] = self._joined(observation["fail"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def load_state(self, path: str | Path) -> None:
        """Add values written by :meth:`save_state` to this accumulator."""
        with np.load(path) as saved:
            for chain, observation in self._observations.items():
                for outcome in ("pass", "fail"):
                    key = f"{chain}__{outcome}"
                    if key in saved.files:
                        observation[outcome].append(saved[key])

    def plot_paths(self) -> list[Path]:
        """Return the PNG files :meth:`write` produces for non-empty chains."""
        paths = []
        for row in self._rows():
            if row["relevant_events"]:
                paths.append(self.output_dir / f"{row['chain']}_leading_pt.png")
                paths.append(self.output_dir / f"{row['chain']}_response.png")
        return paths

    @staticmethod
    def _joined(values: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(values) if values else np.array([], dtype=float)

    def _rows(self) -> list[dict[str, str | int | float]]:
        rows = []
        for observation in self._observations.values():
            definition = observation["definition"]
            assert isinstance(definition, TriggerDefinition)
            passed = len(self._joined(observation["pass"]))
            failed = len(self._joined(observation["fail"]))
            rows.append({
                "year": definition.year,
                "flavour": definition.flavour,
                "chain": definition.chain,
                "threshold_gev": definition.threshold_gev,
                "requirements": definition.requirements,
                "relevant_events": passed + failed,
                "passed_chain": passed,
                "did_not_pass_chain": failed,
            })
        return rows

    def _plot_chain(self, row: dict[str, str | int | float]) -> None:
        observation = self._observations[str(row["chain"])]
        passed = self._joined(observation["pass"])
        failed = self._joined(observation["fail"])
        if not len(passed) + len(failed):
            return
        edges = np.linspace(0, self.pt_max_gev, self.bins + 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(failed, bins=edges, histtype="stepfilled", alpha=0.45,
                color="tab:red", label=f"not retained ({len(failed):,})")
        ax.hist(passed, bins=edges, histtype="step", linewidth=2,
                color="tab:blue", label=f"fired & matched / retained ({len(passed):,})")
        ax.axvline(float(row["threshold_gev"]), color="black", linestyle="--", linewidth=1.5,
                   label=f"nominal HLT threshold ({row['threshold_gev']} GeV)")
        ax.set(xlabel=f"Leading offline {row['flavour']} $p_T$ [GeV]", ylabel="Events",
               title=f"{row['chain']} — {row['year']}\n{row['requirements']}")
        ax.legend(title=f"Relevant events: {row['relevant_events']:,}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{row['chain']}_leading_pt.png", dpi=150)
        plt.close(fig)

    def _plot_efficiency(self, row: dict[str, str | int | float]) -> None:
        """Plot the retained fraction versus the same pT quantity as the histogram."""
        observation = self._observations[str(row["chain"])]
        passed, failed = self._joined(observation["pass"]), self._joined(observation["fail"])
        all_pt = np.concatenate((passed, failed))
        if not len(all_pt):
            return
        edges = np.linspace(0, self.pt_max_gev, self.bins + 1)
        denominator, _ = np.histogram(all_pt, bins=edges)
        numerator, _ = np.histogram(passed, bins=edges)
        efficiency = np.divide(numerator, denominator, out=np.full(self.bins, np.nan), where=denominator > 0)
        centers = (edges[1:] + edges[:-1]) / 2
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.step(centers, efficiency, where="mid", color="tab:blue", label="fired & matched fraction")
        ax.axvline(float(row["threshold_gev"]), color="black", linestyle="--", label="nominal HLT threshold")
        ax.set(xlabel=f"Leading offline {row['flavour']} $p_T$ [GeV]", ylabel="Retained / relevant events",
               ylim=(-0.05, 1.05), title=f"{row['chain']} trigger response")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{row['chain']}_response.png", dpi=150)
        plt.close(fig)

    def write(self) -> list[dict[str, str | int | float]]:
        """Write histograms, response plots, and equivalent CSV/text counts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._rows()
        fields = list(rows[0]) if rows else ["year", "flavour", "chain", "threshold_gev", "requirements", "relevant_events", "passed_chain", "did_not_pass_chain"]
        with (self.output_dir / "trigger_counts.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (self.output_dir / "trigger_counts.txt").open("w", encoding="utf-8") as output:
            output.write("Real-data trigger debug counts\n")
            output.write("Relevant = in the chain's run range and has >=1 offline lepton of its flavour.\n")
            output.write("Passed chain = HLT physics decision (efPassedPhysics) fired AND an offline lepton "
                         "is trigger-matched; counts match each plot legend.\n\n")
            for row in rows:
                output.write(
                    f"{row['chain']} ({row['year']} {row['flavour']}; {row['requirements']})\n"
                    f"  nominal threshold: {row['threshold_gev']} GeV\n"
                    f"  relevant events: {row['relevant_events']}\n"
                    f"  fired & matched / retained: {row['passed_chain']}\n"
                    f"  not retained: {row['did_not_pass_chain']}\n\n"
                )
                self._plot_chain(row)
                self._plot_efficiency(row)
        return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug real Open Data lepton triggers without changing pipeline selection.")
    parser.add_argument("--input", action="append", required=True, help="ROOT URI or path; repeat for 2015 and 2016 files")
    parser.add_argument("--output-dir", default="plots/trigger_debug/generated", help="Directory for plots and count summaries")
    parser.add_argument("--batch-size", type=int, default=40_000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--pt-max-gev", type=float, default=200.0)
    args = parser.parse_args(argv)

    debugger = RealDataTriggerDebugger(args.output_dir, bins=args.bins, pt_max_gev=args.pt_max_gev)
    for source in args.input:
        events = FileParser.parse_file(
            source, ["CollectionTree"], "2024r-pp", batch_size=args.batch_size,
            enable_trigger_matching=True, parse_mc=False,
        )
        if events is None:
            raise RuntimeError(f"Could not parse trigger-debug input: {source}")
        debugger.add_events(events)
    rows = debugger.write()
    print(f"Wrote {len(rows)} trigger summaries to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
