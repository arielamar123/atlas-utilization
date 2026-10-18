"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import awkward as ak
import logging
import numpy as np

from services.calculations import physics_calcs, consts
from services.parsing import good_runs_list, schemas

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

class TriggerInformationUnavailableError(ValueError):
    """Collision-data events lack the trigger information needed to select them."""

class LumiBlockInformationUnavailableError(ValueError):
    """Collision-data events lack the run/lumiblock needed for the Good Runs List."""

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

    events = _filter_events_by_non_jet_object_total(events)
    return _drop_events_without_objects(events)


def _drop_events_without_objects(events: ak.Array) -> ak.Array:
    """Reject events with no retained object.

    Invariant-mass calculation needs at least one object type
    (min_particles_in_combination >= 1), so such events would be dropped
    there anyway; dropping them here keeps them out of the parsed files.
    """
    object_fields = [field for field in events.fields if events[field].ndim > 1]
    if not object_fields:
        return events

    total = ak.zeros_like(ak.num(events[object_fields[0]]), dtype=np.int64)
    for field in object_fields:
        total = total + ak.num(events[field])
    return events[total > 0]


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


def apply_good_runs_list(
    events: ak.Array,
    file_path: str = "",
    good: Optional[Dict[int, Any]] = None,
) -> ak.Array:
    """
    Keep only collision-data events from Good Runs List lumiblocks.

    Collision data only; MC has no GRL.  Needs ``_dataRunNumber`` and
    ``_dataLumiBlock``.  ``_dataLumiBlock`` is dropped from the output;
    ``_dataRunNumber`` is kept for the trigger selection.  ``good`` is
    {run: [(first, last lumiblock), ...]} and defaults to the shipped GRLs.
    """
    missing = [f for f in ("_dataRunNumber", "_dataLumiBlock") if f not in events.fields]
    if missing:
        raise LumiBlockInformationUnavailableError(
            f"Good Runs List selection requires {missing}, but they were not "
            f"read from collision-data file {file_path}"
        )

    good = good_runs_list.good_lumiblocks() if good is None else good
    runs = ak.to_numpy(events["_dataRunNumber"])
    lumiblocks = ak.to_numpy(events["_dataLumiBlock"])
    keep = good_runs_list.in_good_lumiblocks(runs, lumiblocks, good)

    n_total = len(events)
    n_pass = int(np.sum(keep))
    logging.getLogger(__name__).info(
        "Good Runs List: %d / %d events in good lumiblocks (%.1f%%); "
        "%d events from runs outside the GRL in %s",
        n_pass, n_total, 100 * n_pass / n_total if n_total else 0,
        int(np.sum(~np.isin(runs, list(good)))), file_path,
    )

    filtered = events[keep]
    return ak.zip(
        {f: filtered[f] for f in filtered.fields if f != "_dataLumiBlock"},
        depth_limit=1,
    )


def apply_trigger_selection(
    events: ak.Array,
    release_year: str = "2024r-pp",
    file_path: str = "",
    parse_mc: bool = False,
    collision_kinematic_cuts: Optional[Dict[str, Any]] = None,
    collision_allowed_objects: Optional[tuple[str, ...]] = None,
) -> ak.Array:
    """
    Keep only events where at least one lepton fired a single-lepton trigger.

    The ``_triggerMatch`` field (if present) is a record of per-event booleans,
    one per trigger chain.  An event passes if ANY electron chain OR ANY muon
    chain is True.

    Returns the filtered events array (``_triggerMatch`` field is dropped
    from the output to avoid downstream issues with non-particle fields).

    ``parse_mc`` is the authoritative mode supplied by parsing configuration.
    Collision data is handled by ``_apply_collision_trigger_selection``; the
    MC path below is unchanged from master.
    """
    if not parse_mc:
        return _apply_collision_trigger_selection(
            events,
            file_path,
            collision_kinematic_cuts=collision_kinematic_cuts,
            collision_allowed_objects=collision_allowed_objects,
        )

    logger = logging.getLogger(__name__)

    if "_triggerMatch" not in events.fields:
        logger.warning(
            "Skipping file %s: no trigger-match branches on file (%d events dropped)",
            file_path, len(events),
        )
        return events[:0]

    trig = events["_triggerMatch"]
    chain_defs = schemas.SINGLE_LEPTON_TRIGGER_CHAINS

    if "_runNumber" in events.fields:
        # MC: each event's trigger year is set by its random run number
        rrn = events["_runNumber"]
        trigger_years = [
            (year, (rrn >= lo) & (rrn <= hi))
            for year, (lo, hi) in schemas.YEAR_RUN_RANGES.items()
        ]
    else:
        # Data: the file's year applies to every event
        trigger_years = [
            (year, True) for year in schemas.get_trigger_years(release_year, file_path)
        ]

    # Build per-event booleans: did any electron / muon chain of the event's year fire?
    electron_pass = ak.zeros_like(ak.Array([False] * len(events)))
    muon_pass = ak.zeros_like(ak.Array([False] * len(events)))
    for year, in_year in trigger_years:
        if year not in chain_defs:
            raise ValueError(
                f"No trigger chains defined for year '{year}'. "
                f"Supported: {sorted(chain_defs.keys())}"
            )
        year_chains = chain_defs[year]
        for chain in year_chains.get("Electrons", []):
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
            if full_branch in trig.fields:
                electron_pass = electron_pass | (in_year & trig[full_branch])
        for chain in year_chains.get("Muons", []):
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
            if full_branch in trig.fields:
                muon_pass = muon_pass | (in_year & trig[full_branch])

    # Event passes if any lepton trigger fired
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
    particle_fields = {f: filtered[f] for f in filtered.fields if f not in ("_triggerMatch", "_runNumber")}
    return ak.zip(particle_fields, depth_limit=1)


def _collision_selected_lepton_masks(
    events: ak.Array,
    kinematic_cuts: Optional[Dict[str, Any]],
    allowed_objects: Optional[tuple[str, ...]],
) -> Dict[str, ak.Array]:
    """Return the parsing-stage offline lepton masks without reordering them."""
    masks: Dict[str, ak.Array] = {}
    for field, yaml_name in (("Electrons", "electrons"), ("Muons", "muons")):
        particles = events[field] if field in events.fields else None
        if particles is None:
            continue
        # Test fixtures and callers that request no object cuts need not
        # materialize kinematic fields; local indices preserve the jagged
        # offline-object shape in either case.
        mask = ak.ones_like(ak.local_index(particles, axis=1), dtype=bool)
        if allowed_objects is not None and field not in allowed_objects:
            masks[field] = mask & False
            continue
        raw = (kinematic_cuts or {}).get(yaml_name, (kinematic_cuts or {}).get(field, {}))
        cuts = normalize_yaml_kinematic_cuts(raw) if isinstance(raw, dict) else {}
        if "pt" in cuts:
            if not hasattr(particles, "pt"):
                raise TriggerInformationUnavailableError(f"{field} is missing configured kinematic field 'pt'")
            mask = mask & (ak.values_astype(particles.pt, float) >= cuts["pt"]["min"])
        if "eta" in cuts:
            if not hasattr(particles, "eta"):
                raise TriggerInformationUnavailableError(f"{field} is missing configured kinematic field 'eta'")
            eta = ak.values_astype(particles.eta, float)
            mask = mask & (eta >= cuts["eta"]["min"]) & (eta <= cuts["eta"]["max"])
        if "phi" in cuts:
            if not hasattr(particles, "phi"):
                raise TriggerInformationUnavailableError(f"{field} is missing configured kinematic field 'phi'")
            phi = ak.values_astype(particles.phi, float)
            mask = mask & (phi >= cuts["phi"]["min"]) & (phi <= cuts["phi"]["max"])
        if field == "Electrons" and cuts.get("rel_isolation_max") is not None:
            iso_name = consts.ELECTRON_REL_ISOLATION_FIELD
            if not hasattr(particles, iso_name):
                raise TriggerInformationUnavailableError(
                    f"Electron trigger matching requires configured isolation field {iso_name!r}"
                )
            pt = ak.values_astype(particles.pt, float)
            iso = ak.values_astype(getattr(particles, iso_name), float)
            mask = mask & (iso / ak.where(pt > 0, pt, np.inf) < cuts["rel_isolation_max"])
        masks[field] = mask
    return masks


def _selected_match(mask: ak.Array, selected: Optional[ak.Array]) -> ak.Array:
    """Intersect a collision match mask with selected offline leptons.

    Scalar masks are retained as a compatibility fallback for old in-memory
    callers; parsed collision files always provide a jagged object mask.
    """
    if selected is None:
        return ak.any(mask, axis=1) if mask.ndim > 1 else mask
    if mask.ndim <= 1:
        return mask
    if not ak.all(ak.num(mask, axis=1) == ak.num(selected, axis=1)):
        raise TriggerInformationUnavailableError(
            "Collision trigger ElementLink mask does not align with its offline lepton collection"
        )
    return ak.any(mask & selected, axis=1)


def evaluate_collision_trigger_selection(
    events: ak.Array,
    file_path: str = "",
    collision_kinematic_cuts: Optional[Dict[str, Any]] = None,
    collision_allowed_objects: Optional[tuple[str, ...]] = None,
) -> ak.Array:
    """Evaluate collision triggers with explicit PASS/FAIL/UNDETERMINED state.

    A pass requires a fired configured chain and an ElementLink match to an
    electron or muon that survives the parsing-stage object cuts.  This is
    collision-only; the MC trigger path is intentionally separate.
    """
    if "_triggerDecision" not in events.fields:
        raise TriggerInformationUnavailableError(
            "No readable single-lepton trigger decision was supplied for "
            f"collision-data file {file_path}."
        )
    if "_dataRunNumber" not in events.fields:
        raise TriggerInformationUnavailableError(
            "Collision-data trigger selection requires "
            f"'{schemas.DATA_RUN_NUMBER_BRANCH}', but it was not read from {file_path}"
        )
    if "_dataYear" not in events.fields:
        raise TriggerInformationUnavailableError(
            "Collision-data trigger selection requires trusted dataset provenance "
            "(data15_13TeV or data16_13TeV); no data-taking year was supplied "
            f"for {file_path}."
        )

    decisions = events["_triggerDecision"]
    decision_available = decisions["_available"] if "_available" in decisions.fields else None
    decision_in_menu = decisions["_inMenu"] if "_inMenu" in decisions.fields else None
    matches = events["_triggerMatch"] if "_triggerMatch" in events.fields else None
    selected_masks = _collision_selected_lepton_masks(
        events, collision_kinematic_cuts, collision_allowed_objects
    )
    years = events["_dataYear"]
    empty = ak.zeros_like(events["_dataRunNumber"], dtype=bool)
    passed, undetermined, known_year = empty, empty, empty

    for year, menu in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS.items():
        in_year = years == year
        known_year = known_year | in_year
        if not bool(ak.any(in_year)):
            continue
        for flavour, flavour_chains in menu.items():
            for chain in flavour_chains:
                if chain not in decisions.fields:
                    # The decoder promises all configured chain fields.  A
                    # missing field is malformed metadata, never a false bit.
                    undetermined = undetermined | in_year
                    continue
                in_menu = (
                    decision_in_menu[chain] if decision_in_menu is not None and chain in decision_in_menu.fields
                    else ak.ones_like(empty, dtype=bool)
                )
                readable = (
                    decision_available[chain] if decision_available is not None and chain in decision_available.fields
                    else ak.ones_like(empty, dtype=bool)
                )
                fired = decisions[chain]
                branch = schemas.data_trigger_match_branch(chain)
                match_available = matches is not None and branch in matches.fields
                matched = (
                    _selected_match(matches[branch], selected_masks.get(flavour))
                    if match_available else empty
                )
                # A chain absent from a valid menu cannot have fired.  A
                # present-but-truncated HLT word, or a fired chain whose match
                # decoration is absent, is undecidable unless another chain
                # definitely passes.
                chain_pass = in_year & in_menu & readable & fired & matched
                passed = passed | chain_pass
                undetermined = undetermined | (
                    in_year & in_menu & (~readable | (fired & (not match_available)))
                )

    unsupported = ~known_year
    undetermined = undetermined | unsupported
    # A definite pass resolves an event even when another chain is unreadable.
    undetermined = undetermined & ~passed
    status = np.where(
        ak.to_numpy(passed), "PASS",
        np.where(ak.to_numpy(undetermined), "UNDETERMINED", "FAIL"),
    )
    return ak.zip({
        "event_trigger_pass": passed,
        "trigger_status": ak.Array(status),
        "metadata_available": ~undetermined,
        "selected_lepton_trigger_pass": passed,
    })


def _apply_collision_trigger_selection(
    events: ak.Array,
    file_path: str = "",
    collision_kinematic_cuts: Optional[Dict[str, Any]] = None,
    collision_allowed_objects: Optional[tuple[str, ...]] = None,
) -> ak.Array:
    """Apply only definite collision-data trigger passes.

    Metadata failures are raised distinctly instead of silently becoming a
    physical failure.  The handler records these as coverage failures.
    """
    result = evaluate_collision_trigger_selection(
        events,
        file_path,
        collision_kinematic_cuts=collision_kinematic_cuts,
        collision_allowed_objects=collision_allowed_objects,
    )
    undecidable = result.trigger_status == "UNDETERMINED"
    if bool(ak.any(undecidable)):
        raise TriggerInformationUnavailableError(
            f"Trigger metadata is insufficient for {int(ak.sum(undecidable))} / "
            f"{len(events)} collision-data event(s) in {file_path}"
        )
    filtered = events[result.event_trigger_pass]
    particle_fields = {
        f: filtered[f]
        for f in filtered.fields
        if f not in ("_triggerMatch", "_triggerDecision", "_runNumber", "_dataRunNumber", "_dataYear")
    }
    return ak.zip(particle_fields, depth_limit=1)
