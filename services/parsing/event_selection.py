"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import awkward as ak
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

PERSISTED_PARTICLE_FIELDS = frozenset({"pt", "eta", "phi", "mass", "charge"})
SUPPORTED_PARTICLE_COLLECTIONS = frozenset((*schemas.BASE_OBJECTS, "BJets"))


def canonical_particle_field_name(key: str) -> str:
    return YAML_PARTICLE_KEYS.get(key.lower(), key)


def _fields_for_collection(collection: str, release_year: str) -> tuple[str, ...]:
    """Return the stable fields needed downstream for one release/object."""
    source_collection = "Jets" if collection == "BJets" else collection
    try:
        release_schema = schemas.get_schema_for_release(release_year)
        configured_fields = release_schema.get("objects", {}).get(source_collection)
    except KeyError:
        configured_fields = None

    if configured_fields is None:
        configured_fields = schemas.BASE_OBJECTS.get(source_collection, ())

    return tuple(
        field for field in configured_fields if field in PERSISTED_PARTICLE_FIELDS
    )


def _empty_particle_collection(event_count: int, fields: Iterable[str]) -> ak.Array:
    """Build a typed jagged record collection containing no particles."""
    counts = np.zeros(event_count, dtype=np.int64)
    typed_fields = {}
    for field in fields:
        dtype = np.int8 if field == "charge" else np.float32
        typed_fields[field] = ak.unflatten(np.array([], dtype=dtype), counts)
    return ak.zip(typed_fields)


def normalize_particle_collections(
    events: ak.Array,
    objects_to_calculate: Iterable[str],
    release_year: str,
) -> ak.Array:
    """
    Make parsing output follow ``objects_to_calculate`` exactly.

    Missing collections become typed empty jagged arrays, so files with
    different top-level schemas can be concatenated without losing a requested
    collection. Unrequested collections and selection-only auxiliary fields are
    discarded before chunk accumulation.
    """
    requested = tuple(objects_to_calculate)
    unknown = [name for name in requested if name not in SUPPORTED_PARTICLE_COLLECTIONS]
    if unknown:
        raise ValueError(
            "Unsupported objects_to_calculate collection(s): " + ", ".join(unknown)
        )
    if len(set(requested)) != len(requested):
        raise ValueError("objects_to_calculate contains duplicate collections")

    normalized = {}
    for collection_name in requested:
        expected_fields = _fields_for_collection(collection_name, release_year)
        if collection_name not in events.fields:
            normalized[collection_name] = _empty_particle_collection(
                len(events), expected_fields
            )
            continue

        particles = events[collection_name]
        missing_fields = [
            field for field in expected_fields if field not in particles.fields
        ]
        if missing_fields:
            raise ValueError(
                f"{collection_name} in {release_year} is missing required parsed "
                f"field(s): {', '.join(missing_fields)}"
            )
        normalized[collection_name] = ak.zip(
            {field: particles[field] for field in expected_fields}
        )

    return ak.zip(normalized, depth_limit=1)


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
