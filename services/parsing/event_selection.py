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
) -> ak.Array:
    """
    Kinematic cuts are applied per particle type first, then event-level count ranges.
    """
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
            mapped[cname] = val
        events = physics_calcs.filter_events_by_particle_counts(
            events,
            mapped,
            is_exact_count=False,
            is_particle_counts_range=True,
        )

    return events


def apply_trigger_selection(
    events: ak.Array,
    release_year: str = "2024r-pp",
    file_path: str = "",
) -> ak.Array:
    """Keep only events passing a configured single-lepton trigger.

    ``_triggerPass`` contains per-chain event booleans.  It comes from genuine
    MC object matching or, for data fallback, HLT decision bits.  It does not
    restrict the offline leptons that later particle cuts may retain.
    """
    logger = logging.getLogger(__name__)

    if "_triggerPass" not in events.fields:
        raise RuntimeError(f"Trigger selection enabled but no usable trigger source in {file_path}")

    trig = events["_triggerPass"]
    chain_defs = schemas.SINGLE_LEPTON_TRIGGER_CHAINS
    mode = "MC" if release_year.endswith("_mc") else "data"
    source = (
        ak.to_list(events["_triggerSource"])[0]
        if "_triggerSource" in events.fields and len(events)
        else "unknown"
    )

    if release_year.endswith("_mc"):
        if "_triggerRunNumber" not in events.fields:
            raise RuntimeError(
                f"MC trigger selection requires EventInfoAuxDyn.RandomRunNumber in {file_path}"
            )
        # MC: each event's trigger year is set by its random run number
        rrn = events["_triggerRunNumber"]
        trigger_years = [
            (year, (rrn >= lo) & (rrn <= hi))
            for year, (lo, hi) in schemas.YEAR_RUN_RANGES.items()
        ]
        valid = ak.zeros_like(rrn, dtype=bool)
        for _year, mask in trigger_years:
            valid = valid | mask
        if not bool(ak.all(valid)):
            raise RuntimeError(
                f"MC trigger selection found missing or unsupported RandomRunNumber in "
                f"{file_path}: {ak.to_list(rrn[~valid])[:5]}"
            )
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
            if chain in trig.fields:
                electron_pass = electron_pass | (in_year & ak.fill_none(trig[chain], False))
        for chain in year_chains.get("Muons", []):
            if chain in trig.fields:
                muon_pass = muon_pass | (in_year & ak.fill_none(trig[chain], False))

    # Event passes if any lepton trigger fired
    event_mask = electron_pass | muon_pass

    # Log trigger efficiency
    n_total = len(events)
    n_pass = int(ak.sum(event_mask))
    logger.info(
        "Trigger selection (%s, source=%s, years=%s): %d / %d events pass (%.1f%%), "
        "electron-only: %d, muon-only: %d, both: %d",
        mode, source, [year for year, _mask in trigger_years],
        n_pass, n_total, 100 * n_pass / n_total if n_total else 0,
        int(ak.sum(electron_pass & ~muon_pass)),
        int(ak.sum(muon_pass & ~electron_pass)),
        int(ak.sum(electron_pass & muon_pass)),
    )
    for chain in trig.fields:
        logger.debug(
            "Trigger selection (%s, source=%s): chain %s passed %d / %d events",
            mode, source, chain, int(ak.sum(ak.fill_none(trig[chain], False))), n_total,
        )

    filtered = events[event_mask]

    # Drop trigger-only metadata before particle-level cuts.
    particle_fields = {
        f: filtered[f] for f in filtered.fields
        if f not in ("_triggerPass", "_triggerRunNumber", "_triggerSource")
    }
    if not particle_fields:
        # Useful for trigger-only validation and safe even though ordinary
        # parser output always includes particle fields.
        return ak.Array([{} for _ in range(len(filtered))])
    return ak.zip(particle_fields, depth_limit=1)
