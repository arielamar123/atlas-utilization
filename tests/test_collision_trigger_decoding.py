"""Collision-data HLT decisions are decoded from efPassedPhysics by chain counter."""

import pytest

ak = pytest.importorskip("awkward")

from services.parsing import schemas
from services.parsing.file_parser import FileParser

E24, MU20 = "HLT_e24_lhmedium_L1EM20VH", "HLT_mu20_iloose_L1MU15"


def _words(*bits, size=8):
    words = [0] * size
    for bit in bits:
        words[bit // 32] |= 1 << (bit % 32)
    return words


def _decode(smk, words, menus, chains=(E24, MU20)):
    return FileParser.decode_hlt_physics_decisions(
        ak.Array(smk), ak.Array(words), menus, list(chains), "data.root"
    )


def test_hlt_bits_are_read_at_each_events_own_menu_counter():
    menus = {
        1: {"chains": {E24: {"counter": "64"}, MU20: {"counter": "200"}}},
        2: {"chains": {E24: {"counter": "3"}, MU20: {"counter": "64"}}},
    }
    decoded = _decode(
        [1, 1, 2, 2],
        [_words(64), _words(200), _words(3), _words(64)],
        menus,
    )

    assert ak.to_list(decoded[E24]) == [True, False, True, False]
    assert ak.to_list(decoded[MU20]) == [False, True, False, True]


def test_hlt_counter_word_boundaries_and_menu_change_are_exact():
    chains = ("HLT_a", "HLT_b", "HLT_c", "HLT_d")
    menus = {
        10: {"chains": {chain: {"counter": str(counter)} for chain, counter in zip(chains, (31, 32, 63, 64))}},
        11: {"chains": {chain: {"counter": str(counter)} for chain, counter in zip(chains, (64, 63, 32, 31))}},
    }
    decoded = _decode([10, 11], [_words(31, 32, 63, 64), _words(31, 32, 63, 64)], menus, chains)
    assert ak.to_list(decoded["HLT_a"]) == [True, True]
    assert ak.to_list(decoded["HLT_d"]) == [True, True]
    assert ak.to_list(decoded._available["HLT_c"]) == [True, True]


def test_chain_missing_from_menu_is_not_fired_and_short_bitset_is_unavailable():
    menus = {7: {"chains": {E24: {"counter": "255"}}}}
    decoded = _decode([7, 7], [_words(255), _words(size=2)], menus)

    assert ak.to_list(decoded[E24]) == [True, False]
    assert ak.to_list(decoded[MU20]) == [False, False]
    assert ak.to_list(decoded._available[E24]) == [True, False]
    assert ak.to_list(decoded._inMenu[MU20]) == [False, False]


def test_event_smk_without_menu_is_an_explicit_error():
    with pytest.raises(ValueError, match="No TriggerMenuJson_HLT entry for SMK 9"):
        _decode([9], [_words(1)], {1: {"chains": {}}})


def test_data_reads_hlt_physics_not_level1_bitset():
    assert schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH.endswith(".efPassedPhysics")


def test_data_2015_menu_uses_documented_mu50_without_changing_mc():
    assert "HLT_mu50" in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS["2015"]["Muons"]
    assert "HLT_mu40" not in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS["2015"]["Muons"]
    assert any("mu40" in chain for chain in schemas.SINGLE_LEPTON_TRIGGER_CHAINS["2015"]["Muons"])


class _FakeTree:
    def __init__(self, data):
        self.data = data

    def arrays(self, branches, entry_start, entry_stop, library):
        return self.data[sorted(branches)][entry_start:entry_stop]


def test_hlt_bits_are_decoded_batch_by_batch_without_keeping_bitsets():
    smk_b = schemas.TRIGGER_DECISION_SMK_BRANCH
    hlt_b = schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH
    menus = {1: {"chains": {E24: {"counter": "64"}}}, 2: {"chains": {E24: {"counter": "3"}}}}
    smk = [1, 1, 2, 2, 1]
    words = [_words(64), _words(), _words(3), _words(64), _words(64)]
    tree = _FakeTree(ak.Array({smk_b: smk, hlt_b: words, "x": [1, 2, 3, 4, 5]}))
    obj_branches = {"_triggerDecisionRaw": {smk_b: "smk", hlt_b: "hlt"}, "Other": {"x": "x"}}
    calls = []

    def reduce(batch):
        calls.append(len(batch))
        return FileParser.decode_hlt_physics_decisions(batch[smk_b], batch[hlt_b], menus, [E24])

    result, error = FileParser._read_file_in_batches(
        tree, {smk_b, hlt_b, "x"}, obj_branches, 5, 2,
        {"_triggerDecisionRaw": ("_triggerDecision", reduce)},
    )

    assert error is None and calls == [2, 2, 1]
    assert "_triggerDecisionRaw" not in result
    assert ak.to_list(result["_triggerDecision"][E24]) == [True, False, True, False, True]
    assert ak.to_list(result["Other"]["x"]) == [1, 2, 3, 4, 5]


def _entry(*inner_sizes):
    """Serialized vector<vector<ElementLink>>: header, outer size, inner vectors."""
    link = bytes.fromhex("40000018 000067bf b0734000 000e0000 feb3df9e 3902fec0 00000000".replace(" ", ""))
    body = len(inner_sizes).to_bytes(4, "big")
    for n in inner_sizes:
        body += n.to_bytes(4, "big") + link * n
    return bytes.fromhex("4000") + (len(body) + 2).to_bytes(2, "big") + bytes.fromhex("0009") + body


def test_trigger_matches_are_read_from_raw_entry_bytes():
    raw = ak.Array([list(_entry()), list(_entry(1)), list(_entry(0)), list(_entry(0, 2))])

    assert FileParser.trigger_matches_from_raw_entries(raw).tolist() == [False, True, False, True]
    assert FileParser.trigger_matches_from_raw_entries(raw[:0]).tolist() == []


def test_collision_raw_decoder_preserves_the_matched_offline_index():
    index_one = _entry(1)[:-4] + (1).to_bytes(4, "big")
    raw = ak.Array([list(index_one), list(_entry())])
    assert FileParser.trigger_match_masks_from_raw_entries(raw, [2, 3]) == [
        [False, True], [False, False, False],
    ]


def test_unexpected_raw_match_layout_is_reported():
    truncated = list(_entry(1))[:-3]
    assert FileParser.trigger_matches_from_raw_entries(ak.Array([truncated])) is None
