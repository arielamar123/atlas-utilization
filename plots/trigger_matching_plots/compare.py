#!/usr/bin/env python3
"""Compare parsed collision-data outputs with trigger matching enabled/disabled.

The parser deliberately removes temporary trigger/run metadata before writing its
``events`` ROOT tree.  This tool therefore validates the *analysis-level*
effect of matching: event yields and object distributions in two otherwise
identical pipeline runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import awkward as ak
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PT_BINS_GEV = np.linspace(0.0, 200.0, 51)
MULTIPLICITY_BINS = np.arange(-0.5, 12.5, 1.0)


@dataclass
class DatasetSummary:
    """Histograms accumulated from parsed ``events`` trees."""

    files: list[str] = field(default_factory=list)
    event_count: int = 0
    leading_muon_pt: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PT_BINS_GEV) - 1, dtype=np.int64)
    )
    all_muon_pt: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PT_BINS_GEV) - 1, dtype=np.int64)
    )
    jet_multiplicity: np.ndarray = field(
        default_factory=lambda: np.zeros(len(MULTIPLICITY_BINS) - 1, dtype=np.int64)
    )
    bjet_multiplicity: np.ndarray | None = None
    leading_jet_pt: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PT_BINS_GEV) - 1, dtype=np.int64)
    )
    top_control_muon_pt: np.ndarray | None = None
    top_control_event_count: int | None = None


def discover_parsed_files(input_path: str | Path) -> list[Path]:
    """Return parsed event ROOT files below a pipeline output path.

    Histogram ROOT files are intentionally excluded: only ``parsed_*.root``
    files contain the event-level object branches needed for this comparison.
    """

    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("parsed_*.root"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = [file for file in candidates if file.name.startswith("parsed_")]
    if not files:
        raise FileNotFoundError(
            f"No parsed_*.root files found below {path}. Pass a pipeline run "
            "directory, its parsed_data directory, or one parsed ROOT file."
        )
    return files


def _histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=np.int64)
    return np.histogram(values, bins=bins)[0].astype(np.int64)


def _first_pt_gev(objects: ak.Array) -> np.ndarray:
    """Leading object pT, converting the pipeline's MeV representation to GeV."""

    leading = ak.firsts(objects)
    return np.asarray(ak.to_numpy(leading[~ak.is_none(leading)]), dtype=float) / 1000.0


def _flat_pt_gev(objects: ak.Array) -> np.ndarray:
    values = ak.flatten(objects, axis=None)
    return np.asarray(ak.to_numpy(values), dtype=float) / 1000.0


def _required_fields(tree: uproot.behaviors.TTree.TTree) -> set[str]:
    # ``cycle`` is not accepted by the uproot release bundled in the analysis
    # container; TTree branch names are already unambiguous here.
    fields = set(tree.keys())
    required = {"Muons_pt", "Jets_pt", "nJets"}
    missing = required - fields
    if missing:
        raise ValueError(
            f"Tree {tree.file.file_path}:events lacks required parsed branches: "
            f"{', '.join(sorted(missing))}"
        )
    return fields


def summarize(input_path: str | Path) -> DatasetSummary:
    """Stream parsed ROOT files and return comparison histograms."""

    summary = DatasetSummary()
    for root_file in discover_parsed_files(input_path):
        with uproot.open(root_file) as source:
            if "events" not in source:
                raise ValueError(f"{root_file} has no 'events' tree")
            tree = source["events"]
            fields = _required_fields(tree)
            branches = ["Muons_pt", "Jets_pt", "nJets"]
            has_bjets = "nBJets" in fields
            if has_bjets:
                branches.append("nBJets")
                if summary.bjet_multiplicity is None:
                    summary.bjet_multiplicity = np.zeros(
                        len(MULTIPLICITY_BINS) - 1, dtype=np.int64
                    )
                    summary.top_control_muon_pt = np.zeros(
                        len(PT_BINS_GEV) - 1, dtype=np.int64
                    )
                    summary.top_control_event_count = 0

            summary.files.append(str(root_file))
            summary.event_count += tree.num_entries
            for arrays in tree.iterate(branches, step_size="100 MB", library="ak"):
                muons = arrays["Muons_pt"]
                jets = arrays["Jets_pt"]
                leading_muon = _first_pt_gev(muons)
                summary.leading_muon_pt += _histogram(leading_muon, PT_BINS_GEV)
                summary.all_muon_pt += _histogram(_flat_pt_gev(muons), PT_BINS_GEV)
                summary.leading_jet_pt += _histogram(_first_pt_gev(jets), PT_BINS_GEV)

                n_jets = np.asarray(ak.to_numpy(arrays["nJets"]), dtype=float)
                summary.jet_multiplicity += _histogram(n_jets, MULTIPLICITY_BINS)

                if has_bjets:
                    n_bjets = np.asarray(ak.to_numpy(arrays["nBJets"]), dtype=float)
                    assert summary.bjet_multiplicity is not None
                    assert summary.top_control_muon_pt is not None
                    assert summary.top_control_event_count is not None
                    summary.bjet_multiplicity += _histogram(n_bjets, MULTIPLICITY_BINS)

                    leading_per_event = ak.firsts(muons)
                    muon_pt_gev = ak.fill_none(leading_per_event, -np.inf) / 1000.0
                    control_mask = (muon_pt_gev >= 27.0) & (n_jets >= 4) & (n_bjets >= 1)
                    control_pt = np.asarray(
                        ak.to_numpy(muon_pt_gev[control_mask]), dtype=float
                    )
                    summary.top_control_muon_pt += _histogram(control_pt, PT_BINS_GEV)
                    summary.top_control_event_count += int(np.count_nonzero(control_mask))
    return summary


def _normalised(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype=float)


def _plot_overlay(
    enabled: np.ndarray,
    disabled: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    output_file: Path,
) -> None:
    """Plot unit-area shapes plus enabled/disabled ratio."""

    enabled_shape = _normalised(enabled)
    disabled_shape = _normalised(disabled)
    fig, (top, ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(7.5, 6.3), gridspec_kw={"height_ratios": [3, 1]}
    )
    top.stairs(enabled_shape, bins, label="Trigger enabled", color="C0", linewidth=1.8)
    top.stairs(disabled_shape, bins, label="Trigger disabled", color="C1", linewidth=1.8)
    top.set_ylabel("Unit-area events")
    top.legend()
    top.grid(alpha=0.25)

    values = np.divide(
        enabled_shape,
        disabled_shape,
        out=np.full_like(enabled_shape, np.nan, dtype=float),
        where=disabled_shape > 0,
    )
    ratio.stairs(values, bins, color="black", linewidth=1.4)
    ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ratio.set_ylim(0.0, 2.0)
    ratio.set_ylabel("On / off")
    ratio.set_xlabel(xlabel)
    ratio.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def _plot_event_yields(enabled: DatasetSummary, disabled: DatasetSummary, output_file: Path) -> None:
    labels = ["All parsed events"]
    on_values = [enabled.event_count]
    off_values = [disabled.event_count]
    if enabled.top_control_event_count is not None and disabled.top_control_event_count is not None:
        labels.append(r"$\mu$ + $\geq4j$ + $\geq1b$ control")
        on_values.append(enabled.top_control_event_count)
        off_values.append(disabled.top_control_event_count)
    positions = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(positions - width / 2, on_values, width, label="Trigger enabled", color="C0")
    axis.bar(positions + width / 2, off_values, width, label="Trigger disabled", color="C1")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Events")
    axis.set_yscale("log")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def _summary_dict(summary: DatasetSummary) -> dict[str, object]:
    return {
        "files": summary.files,
        "event_count": summary.event_count,
        "top_control_event_count": summary.top_control_event_count,
        "histogram_entries": {
            "leading_muon_pt": int(summary.leading_muon_pt.sum()),
            "all_muon_pt": int(summary.all_muon_pt.sum()),
            "jet_multiplicity": int(summary.jet_multiplicity.sum()),
            "bjet_multiplicity": (
                int(summary.bjet_multiplicity.sum())
                if summary.bjet_multiplicity is not None
                else None
            ),
        },
    }


def build_comparison(
    enabled_path: str | Path, disabled_path: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Create validation plots and ``summary.json``; return its contents."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    enabled = summarize(enabled_path)
    disabled = summarize(disabled_path)

    _plot_event_yields(enabled, disabled, output / "01_event_yields.png")
    _plot_overlay(enabled.leading_muon_pt, disabled.leading_muon_pt, PT_BINS_GEV,
                  r"Leading muon $p_T$ [GeV]", output / "02_leading_muon_pt.png")
    _plot_overlay(enabled.all_muon_pt, disabled.all_muon_pt, PT_BINS_GEV,
                  r"Muon $p_T$ [GeV]", output / "03_all_muon_pt.png")
    _plot_overlay(enabled.jet_multiplicity, disabled.jet_multiplicity, MULTIPLICITY_BINS,
                  "Jet multiplicity", output / "04_jet_multiplicity.png")
    _plot_overlay(enabled.leading_jet_pt, disabled.leading_jet_pt, PT_BINS_GEV,
                  r"Leading jet $p_T$ [GeV]", output / "05_leading_jet_pt.png")

    if enabled.bjet_multiplicity is not None and disabled.bjet_multiplicity is not None:
        _plot_overlay(enabled.bjet_multiplicity, disabled.bjet_multiplicity,
                      MULTIPLICITY_BINS, "b-jet multiplicity",
                      output / "06_bjet_multiplicity.png")
    if enabled.top_control_muon_pt is not None and disabled.top_control_muon_pt is not None:
        _plot_overlay(enabled.top_control_muon_pt, disabled.top_control_muon_pt,
                      PT_BINS_GEV, r"Leading muon $p_T$ [GeV]",
                      output / "07_muon_4jet_1b_control_muon_pt.png")

    payload = {"trigger_enabled": _summary_dict(enabled), "trigger_disabled": _summary_dict(disabled)}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", required=True, help="Pipeline output with trigger matching enabled")
    parser.add_argument("--disabled", required=True, help="Equivalent pipeline output with matching disabled")
    parser.add_argument("--output-dir", default="plots/trigger_matching_plots/generated",
                        help="Directory for generated PNGs and summary.json")
    arguments = parser.parse_args(argv)
    payload = build_comparison(arguments.enabled, arguments.disabled, arguments.output_dir)
    print(f"Wrote plots to {Path(arguments.output_dir).resolve()}")
    print("Events (enabled / disabled): "
          f"{payload['trigger_enabled']['event_count']} / {payload['trigger_disabled']['event_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
