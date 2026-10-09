"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import awkward as ak
import logging
import numpy as np

from services.calculations import physics_calcs
from services.parsing import schemas

YAML_PARTICLE_KEYS: Dict[str, str] = {
    "electrons": "Electrons",
    "muons": "Muons",
    "jets": "Jets",
    "bjets": "BJets",
    "photons": "Photons",
    "taus": "Taus",
}

LIGHT_JET_FIELD = "Jets"
MAX_NON_JET_OBJECTS = 4

class TriggerInformationUnavailableError(RuntimeError):
    """A requested collision-data trigger decision could not be decoded."""

def canonical_particle_field_name(key: str) -> str:
    return YAML_PARTICLE_KEYS.get(key.lower(), key)


def normalize_yaml_kinematic_cuts(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ``pt_min`` / ``eta_max`` / ``rel_isolation_max`` into internal cut dict."""
    out: Dict[str, Any] = {}
    if "pt" in raw and isinstance(raw["pt"], dict):
        out["pt"] = dict(raw["pt"])
    elif "pt_min" in raw:
        out["pt"] = {"min": float(raw["pt_min"])}

    if "eta" in raw and isinstance(raw["eta"], dict):
        out["eta"] = dict(raw["eta"])
    elif "eta_max" in raw:
        em = float(raw["eta_max"])
        out["eta"] = {"min": -em, "max": em}

    if "phi" in raw and isinstance(raw["phi"], dict):
        out["phi"] = dict(raw["phi"])
    elif "phi_min" in raw or "phi_max" in raw:
        out["phi"] = {
            "min": float(raw.get("phi_min", -np.pi)),
            "max": float(raw.get("phi_max", np.pi)),
        }

    if "rel_isolation_max" in raw:
        out["rel_isolation_max"] = float(raw["rel_isolation_max"])

    return out


def apply_parsing_event_selection(
    events: ak.Array,
    particle_counts: Optional[Dict[str, Any]] = None,
    kinematic_cuts: Optional[Dict[str, Any]] = None,
    allowed_objects: Optional[tuple[str, ...]] = None,
) -> ak.Array:
    """
    Retain configured objects, apply per-object kinematic cuts, then apply
    event-level count ranges.  Events may contain unconfigured objects; those
    collections are discarded and do not participate in event selection.

    At most four retained non-light-jet objects are allowed per event.  Light
    jets are deliberately excluded from this total and have no upper bound.
    """
    if allowed_objects is not None:
        events = retain_objects_for_storage(events, allowed_objects)

    if kinematic_cuts:
        by_obj: Dict[str, Dict[str, Any]] = {}
        for key, val in kinematic_cuts.items():
            if not isinstance(val, dict):
                continue
            cname = canonical_particle_field_name(key)
            by_obj[cname] = normalize_yaml_kinematic_cuts(val)
        events = physics_calcs.filter_events_by_kinematics(events, by_obj)

    if particle_counts:
        mapped: Dict[str, Any] = {}
        for key, val in particle_counts.items():
            cname = canonical_particle_field_name(key)
            if cname == LIGHT_JET_FIELD and isinstance(val, dict):
                # Additional light jets must not reject an otherwise valid
                # final state.  Keep an optional lower bound, but remove any
                # configured upper bound.
                mapped[cname] = {**val, "max": float("inf")}
            else:
                mapped[cname] = val

    else:
        mapped = {}

    if mapped:
        events = physics_calcs.filter_events_by_particle_counts(
            events,
            mapped,
            is_exact_count=False,
            is_particle_counts_range=True,
        )

    return _filter_events_by_non_jet_object_total(events)


def _filter_events_by_non_jet_object_total(events: ak.Array) -> ak.Array:
    """Reject events with more than four retained non-light-jet objects."""
    non_jet_fields = [field for field in events.fields if field != LIGHT_JET_FIELD]
    if not non_jet_fields:
        return events

    total = ak.zeros_like(ak.num(events[non_jet_fields[0]]), dtype=np.int64)
    for field in non_jet_fields:
        total = total + ak.num(events[field])
    return events[total <= MAX_NON_JET_OBJECTS]


def retain_objects_for_storage(
    events: ak.Array, objects_to_store: tuple[str, ...]
) -> ak.Array:
    """Drop non-persisted physics collections before event selection."""
    return ak.zip(
        {
            field: events[field]
            for field in events.fields
            if field in objects_to_store
        },
        depth_limit=1,
    )


def apply_trigger_selection(
    events: ak.Array,
    release_year: str = "2024r-pp",
    file_path: str = "",
    parse_mc: bool = False,
) -> ak.Array:
    """
    Keep only events where at least one lepton fired a single-lepton trigger.

    The ``_triggerMatch`` field (if present) is a record of per-event booleans,
    one per trigger chain.  An event passes if ANY electron chain OR ANY muon
    chain is True.

    ``parse_mc`` is the authoritative mode supplied by parsing configuration.
    MC uses ``_runNumber`` (RandomRunNumber); collision data uses
    ``_dataRunNumber`` (runNumber).  Metadata fields are removed before the
    result reaches particle selection.
    """
    logger = logging.getLogger(__name__)

    trigger_field = "_triggerMatch" if parse_mc else "_triggerDecision"
    if trigger_field not in events.fields:
        message = (
            "No readable single-lepton trigger decision was supplied for "
            f"collision-data file {file_path}. "
            "The raw PHYSLITE AnalysisTrigMatch containers are xAOD "
            "TrigComposite objects, not uproot-readable AuxDyn decorations."
        )
        if not parse_mc:
            # Do not turn absent trigger metadata into an implicit pass (or an
            # implicit fail).  The official converter must first materialize
            # the EventInfo trigger decisions from the xAOD containers.
            raise TriggerInformationUnavailableError(message)
        logger.warning(
            "%s (%d MC events dropped)", message, len(events),
        )
        return events[:0]

    trig = events[trigger_field]
    chain_defs = (
        schemas.SINGLE_LEPTON_TRIGGER_CHAINS
        if parse_mc else schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS
    )

    if parse_mc:
        # MC: each event's trigger year is set by its random run number
        if "_runNumber" in events.fields:
            rrn = events["_runNumber"]
            trigger_years = [
                (year, (rrn >= lo) & (rrn <= hi))
                for year, (lo, hi) in schemas.YEAR_RUN_RANGES.items()
            ]
        else:
            # Keep the pre-existing MC fallback for unusual legacy samples.
            # It is deliberately unavailable to collision data below.
            logger.warning(
                "MC file %s has no %s; using its configured release menu",
                file_path, schemas.RANDOM_RUN_NUMBER_BRANCH,
            )
            trigger_years = [
                (year, True) for year in schemas.get_trigger_years(release_year, file_path)
            ]
    else:
        # Collision data is selected per event, not per filename/release.
        if "_dataRunNumber" not in events.fields:
            raise ValueError(
                "Collision-data trigger selection requires "
                f"'{schemas.DATA_RUN_NUMBER_BRANCH}', but it was not read from {file_path}"
            )
        data_run = events["_dataRunNumber"]
        trigger_years = [
            (year, (data_run >= lo) & (data_run <= hi))
            for year, (lo, hi) in schemas.DATA_YEAR_RUN_RANGES.items()
        ]

        valid_data_run = ak.zeros_like(data_run, dtype=bool)
        for _, in_year in trigger_years:
            valid_data_run = valid_data_run | in_year
        invalid_count = int(ak.sum(~valid_data_run))
        if invalid_count:
            logger.warning(
                "Rejecting %d / %d collision-data events with unsupported run numbers in %s",
                invalid_count, len(events), file_path,
            )

    # Build per-event booleans: did any electron / muon chain of the event's year fire?
    electron_pass = ak.zeros_like(ak.Array([False] * len(events)))
    muon_pass = ak.zeros_like(ak.Array([False] * len(events)))
    data_chain_masks: list[tuple[str, str, str, ak.Array]] = []
    for year, in_year in trigger_years:
        if year not in chain_defs:
            raise ValueError(
                f"No trigger chains defined for year '{year}'. "
                f"Supported: {sorted(chain_defs.keys())}"
            )
        year_chains = chain_defs[year]
        applicable_count = int(ak.sum(in_year)) if not isinstance(in_year, bool) else len(events)
        if not applicable_count:
            continue
        available_electrons = [
            chain for chain in year_chains.get("Electrons", [])
            if (chain + schemas.TRIGGER_BRANCH_SUFFIX if parse_mc else chain) in trig.fields
        ]
        available_muons = [
            chain for chain in year_chains.get("Muons", [])
            if (chain + schemas.TRIGGER_BRANCH_SUFFIX if parse_mc else chain) in trig.fields
        ]
        if not available_electrons and not available_muons:
            raise ValueError(
                f"No applicable trigger-match branches are available for {year} "
                f"events in {file_path}"
            )
        logger.info(
            "Trigger menu %s: %d events; electron chains=%s; muon chains=%s",
            year, applicable_count, available_electrons, available_muons,
        )
        for chain in available_electrons:
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX if parse_mc else chain
            if full_branch in trig.fields:
                chain_mask = in_year & trig[full_branch]
                electron_pass = electron_pass | chain_mask
                if not parse_mc:
                    data_chain_masks.append((year, "electron", chain, chain_mask))
        for chain in available_muons:
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX if parse_mc else chain
            if full_branch in trig.fields:
                chain_mask = in_year & trig[full_branch]
                muon_pass = muon_pass | chain_mask
                if not parse_mc:
                    data_chain_masks.append((year, "muon", chain, chain_mask))

    # Event passes if any lepton trigger fired.  Unsupported data run numbers
    # have no year mask and are consequently rejected explicitly.
    if not parse_mc:
        # xTrigDecision is event-level.  Bind each accepted decision to an
        # offline lepton of the corresponding flavour before OR'ing them, so
        # an electron trigger cannot retain a muon-only event (and vice versa).
        electron_pass = electron_pass & (
            ak.num(events["Electrons"]) > 0
            if "Electrons" in events.fields else ak.zeros_like(electron_pass)
        )
        muon_pass = muon_pass & (
            ak.num(events["Muons"]) > 0
            if "Muons" in events.fields else ak.zeros_like(muon_pass)
        )
        for year, flavour, chain, chain_mask in data_chain_masks:
            offline_mask = (
                ak.num(events["Electrons"]) > 0
                if flavour == "electron" and "Electrons" in events.fields
                else ak.num(events["Muons"]) > 0
                if flavour == "muon" and "Muons" in events.fields
                else ak.zeros_like(chain_mask)
            )
            fired = int(ak.sum(chain_mask))
            with_offline_lepton = int(ak.sum(chain_mask & offline_mask))
            logger.info(
                "Collision trigger %s (%s, %s): %d fired; %d with an offline %s",
                chain, year, file_path, fired, with_offline_lepton, flavour,
            )
    event_mask = electron_pass | muon_pass

    # Log trigger efficiency
    n_total = len(events)
    n_pass = int(ak.sum(event_mask))
    logger.info(
        "Trigger selection: %d / %d events pass (%.1f%%), "
        "electron-only: %d, muon-only: %d",
        n_pass, n_total, 100 * n_pass / n_total if n_total else 0,
        int(ak.sum(electron_pass & ~muon_pass)),
        int(ak.sum(muon_pass & ~electron_pass)),
    )

    filtered = events[event_mask]

    # Drop _triggerMatch from the output — it's event-level metadata that
    # would cause axis errors in downstream particle-level operations
    particle_fields = {
        f: filtered[f]
        for f in filtered.fields
        if f not in ("_triggerMatch", "_triggerDecision", "_runNumber", "_dataRunNumber")
    }
    return ak.zip(particle_fields, depth_limit=1)
