"""Parsing drops events left without objects, which mass calculation rejects anyway."""

import pytest

ak = pytest.importorskip("awkward")

from services.calculations.im_calculator import IMCalculator
from services.parsing.event_selection import apply_parsing_event_selection


def _obj(pt_gev):
    return {"pt": pt_gev * 1000.0, "eta": 0.1, "phi": 0.2, "mass": 0.0}


EVENTS = ak.Array({
    "Electrons": [[_obj(30)], [_obj(10)], [], []],
    "Muons": [[], [], [_obj(40)], []],
    "Jets": [[], [_obj(20)], [_obj(50)], []],
})
CUTS = {"electrons": {"pt_min": 25000.0}, "muons": {"pt_min": 25000.0}, "jets": {"pt_min": 30000.0}}
COUNTS = {"electrons": {"min": 0, "max": 4}, "muons": {"min": 0, "max": 4}, "jets": {"min": 0, "max": 4}}


def _select(events):
    return apply_parsing_event_selection(
        events, particle_counts=COUNTS, kinematic_cuts=CUTS,
        allowed_objects=("Electrons", "Muons", "Jets"),
    )


def test_events_without_objects_after_cuts_are_dropped():
    selected = _select(EVENTS)

    # Event 1 loses its 10 GeV electron and 20 GeV jet; event 3 has nothing.
    assert ak.to_list(ak.num(selected.Electrons)) == [1, 0]
    assert ak.to_list(ak.num(selected.Muons)) == [0, 1]
    assert ak.to_list(ak.num(selected.Jets)) == [0, 1]


def test_dropped_events_are_exactly_those_mass_calculation_rejects():
    def counts(events):
        return IMCalculator(events, min_events_per_fs=1, min_k=1, max_k=4, min_n=1, max_n=4).final_state_counts()

    from services.calculations import physics_calcs
    from services.parsing.event_selection import canonical_particle_field_name, normalize_yaml_kinematic_cuts
    without_drop = physics_calcs.filter_events_by_kinematics(
        EVENTS, {canonical_particle_field_name(k): normalize_yaml_kinematic_cuts(v) for k, v in CUTS.items()}
    )
    assert len(without_drop) == 4
    assert counts(_select(EVENTS)) == counts(without_drop)
