"""Good Runs List (GRL) selection of 2015/2016 collision data."""

import pytest

ak = pytest.importorskip("awkward")
np = pytest.importorskip("numpy")
uproot = pytest.importorskip("uproot")

from domain.config import ParsingConfig, PipelineConfig
from domain.events import EventBatch
from orchestration.context import PipelineContext
from orchestration.handlers.parsing_handler import ParsingHandler
from orchestration.states import PipelineState
from services.parsing import good_runs_list, schemas
from services.parsing.data_run_filter import filter_collision_files_by_run
from services.parsing.event_accumulator import EventAccumulator
from services.parsing.event_selection import (
    LumiBlockInformationUnavailableError,
    apply_good_runs_list,
    apply_trigger_selection,
)
from services.parsing.file_parser import FileParser, InvalidFileContentError
from utils.paths import update_config_paths_with_run_dir

GOOD = {276262: [(72, 84), (86, 110)], 300000: [(5, 5)]}
D15 = "HLT_e24_lhmedium_L1EM20VH"


def _events(runs, lumiblocks):
    return ak.Array({
        "event_id": list(range(len(runs))),
        "_dataRunNumber": runs,
        "_dataLumiBlock": lumiblocks,
        "Electrons": [[{"pt": 30_000.0, "eta": 0.1, "phi": 0.2}] for _ in runs],
    })


def test_lumiblock_ranges_are_inclusive_and_per_run():
    runs = [276262] * 7 + [300000] * 3 + [276263]
    lumiblocks = [71, 72, 84, 85, 86, 110, 111, 4, 5, 6, 80]

    keep = good_runs_list.in_good_lumiblocks(np.array(runs), np.array(lumiblocks), GOOD)

    assert keep.tolist() == [False, True, True, False, True, True, False, False, True, False, False]


def test_good_runs_list_selection_keeps_run_number_and_drops_lumiblock(caplog):
    caplog.set_level("INFO")
    events = _events([276262, 276262, 300000, 276263], [72, 85, 5, 72])

    selected = apply_good_runs_list(events, file_path="data.root", good=GOOD)

    assert ak.to_list(selected.event_id) == [0, 2]
    assert "_dataLumiBlock" not in selected.fields
    # The trigger selection still needs the recorded run number.
    assert ak.to_list(selected._dataRunNumber) == [276262, 300000]
    assert ak.to_list(selected.Electrons.pt) == [[30_000.0], [30_000.0]]
    assert "Good Runs List: 2 / 4 events in good lumiblocks" in caplog.text
    assert "1 events from runs outside the GRL in data.root" in caplog.text


def test_shipped_grls_are_the_standard_2015_2016_lists():
    good = good_runs_list.good_lumiblocks()
    runs_2015 = [run for run in good if run <= schemas.DATA_YEAR_RUN_RANGES["2015"][1]]
    runs_2016 = [run for run in good if run >= schemas.DATA_YEAR_RUN_RANGES["2016"][0]]

    assert (len(runs_2015), len(runs_2016)) == (66, 151)
    assert len(good) == 217
    assert min(good) == 276262 and max(good) == 311481
    assert good[276262] == ((72, 84), (86, 110), (118, 288), (292, 379))
    assert all(
        any(lo <= run <= hi for lo, hi in schemas.DATA_YEAR_RUN_RANGES.values())
        for run in good
    )
    for ranges in good.values():
        assert all(start <= end for start, end in ranges)
        assert all(prev[1] < cur[0] for prev, cur in zip(ranges, ranges[1:]))


def test_default_grl_rejects_bad_lumiblocks_and_runs_outside_the_grl():
    # 276262: good LBs 72-84, 86-110; 276330 is a 2015 run without GRL lumiblocks.
    events = _events([276262, 276262, 276262, 276330], [72, 85, 110, 100])

    assert ak.to_list(apply_good_runs_list(events).event_id) == [0, 2]


def test_missing_lumiblock_is_an_explicit_error():
    events = ak.Array({"_dataRunNumber": [276262], "Electrons": [[]]})
    with pytest.raises(LumiBlockInformationUnavailableError, match="_dataLumiBlock"):
        apply_good_runs_list(events, file_path="data.root")


def test_trigger_selection_runs_on_grl_selected_events():
    events = ak.with_field(
        _events([276262, 276262], [72, 85]), ak.Array({D15: [True, True]}), "_triggerDecision"
    )
    events = ak.with_field(
        events, ak.Array({schemas.data_trigger_match_branch(D15): [True, True]}), "_triggerMatch"
    )

    selected = apply_trigger_selection(apply_good_runs_list(events), parse_mc=False)

    assert ak.to_list(selected.event_id) == [0]
    assert not {"_dataRunNumber", "_dataLumiBlock"} & set(selected.fields)


@pytest.mark.parametrize("trigger", [False, True])
def test_collision_data_reads_run_and_lumiblock_for_the_grl(trigger):
    branches = {
        schemas.DATA_RUN_NUMBER_BRANCH,
        schemas.DATA_LUMI_BLOCK_BRANCH,
        schemas.RANDOM_RUN_NUMBER_BRANCH,
    }

    read = FileParser._extract_branches_by_schema(
        branches, "2024r-pp", enable_trigger_matching=trigger, parse_mc=False, apply_grl=True
    )
    assert read["_dataRunNumber"] == {schemas.DATA_RUN_NUMBER_BRANCH: "_dataRunNumber"}
    assert read["_dataLumiBlock"] == {schemas.DATA_LUMI_BLOCK_BRANCH: "_dataLumiBlock"}

    without_grl = FileParser._extract_branches_by_schema(
        branches, "2024r-pp", enable_trigger_matching=trigger, parse_mc=False, apply_grl=False
    )
    assert "_dataLumiBlock" not in without_grl
    assert ("_dataRunNumber" in without_grl) == trigger


def test_mc_never_reads_grl_branches():
    branches = {
        schemas.DATA_RUN_NUMBER_BRANCH,
        schemas.DATA_LUMI_BLOCK_BRANCH,
        schemas.RANDOM_RUN_NUMBER_BRANCH,
    }
    for trigger in (False, True):
        read = FileParser._extract_branches_by_schema(
            branches, "2024r-pp_mc", enable_trigger_matching=trigger, parse_mc=True, apply_grl=True
        )
        assert not {"_dataRunNumber", "_dataLumiBlock"} & set(read)


def test_collision_file_without_lumiblock_cannot_be_grl_selected():
    read = {"_dataRunNumber": {schemas.DATA_RUN_NUMBER_BRANCH: "_dataRunNumber"}}
    with pytest.raises(InvalidFileContentError, match="lumiBlock"):
        FileParser._require_good_runs_branches(read, "data.root")


def test_runs_without_good_lumiblocks_are_dropped_before_the_file_budget(caplog):
    caplog.set_level("INFO")
    base = "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/data15_13TeV/"
    urls = [f"{base}DAOD_PHYSLITE.{d}._000001.pool.root.1" for d in (1, 2, 3, 4)]
    runs = {"1": 266904, "2": 276262, "3": 276330}

    kept = filter_collision_files_by_run({"2024r-pp": urls}, runs, good_runs={276262})

    # 266904: outside the MC ranges; 276330: no GRL lumiblocks; 4: unknown run.
    assert kept["2024r-pp"] == [urls[1], urls[3]]
    assert "dropped 1 / 4 collision-data files from 1 runs outside" in caplog.text
    assert "dropped 1 / 4 collision-data files from 1 runs without Good Runs List" in caplog.text
    assert filter_collision_files_by_run({"2024r-pp": urls}, runs)["2024r-pp"] == urls[1:]


def test_grl_is_only_defined_for_the_2024r_pp_release():
    assert good_runs_list.release_has_grl("2024r-pp")
    assert not good_runs_list.release_has_grl("2016e-8tev")
    assert not good_runs_list.release_has_grl("2025e-13tev-beta")
    assert not good_runs_list.release_has_grl("record_30511")


def _config_dict(run_dir, parse_mc=False, **parsing):
    return update_config_paths_with_run_dir({
        "tasks": {"do_parsing": True},
        "trigger_config": {"enabled": False},
        "parsing_task_config": {
            "output_path": "./output/parsed_data",
            "file_urls_path": "./output/metadata_cache.json",
            "jobs_logs_path": "./output/logs",
            "release_years": ["2024r-pp"],
            "parse_mc": parse_mc,
            "show_progress_bar": False,
            **parsing,
        },
    }, str(run_dir))


def test_apply_grl_defaults_on_and_must_be_boolean(tmp_path):
    assert PipelineConfig.from_dict(_config_dict(tmp_path)).parsing_config.apply_grl is True
    off = PipelineConfig.from_dict(_config_dict(tmp_path, apply_grl=False))
    assert off.parsing_config.apply_grl is False
    with pytest.raises(ValueError, match="apply_grl"):
        ParsingConfig(output_path="o", file_urls_path="f", jobs_logs_path="l", apply_grl="yes")


class _StubProcessor:
    """Yields one in-memory file; records the apply_grl the handler asked for."""

    def __init__(self, events):
        self.events = events
        self.apply_grl = []

    def process_files(self, file_urls, release_year, apply_grl=False, **_):
        self.apply_grl.append(apply_grl)
        for file_id, url in enumerate(file_urls):
            yield EventBatch(
                events=self.events, file_id=file_id, release_year=release_year,
                size_bytes=self.events.layout.nbytes, event_count=len(self.events),
                processing_time_sec=0.0, file_url=url,
            )


def _parse(run_dir, events, parse_mc=False, **parsing):
    key = "2024r-pp_mc" if parse_mc else "2024r-pp"
    processor = _StubProcessor(events)
    context = PipelineContext(
        config=PipelineConfig.from_dict(_config_dict(run_dir, parse_mc=parse_mc, **parsing)),
        current_state=PipelineState.PARSING,
        metadata={key: ["file.root"]},
    )
    context, _ = ParsingHandler(None, processor, EventAccumulator(10**9)).handle(context)
    assert len(context.parsed_files) == 1
    with uproot.open(context.parsed_files[0]) as handle:
        parsed = handle["events"].arrays()
    return processor.apply_grl, parsed


def test_handler_applies_the_grl_to_collision_data_without_trigger_selection(tmp_path):
    events = _events([276262, 276262, 276262, 276330], [72, 85, 110, 100])

    apply_grl, parsed = _parse(tmp_path / "on", events)
    assert apply_grl == [True]
    assert len(parsed) == 2
    assert not {"_dataRunNumber", "_dataLumiBlock"} & set(parsed.fields)

    apply_grl, parsed = _parse(tmp_path / "off", events, apply_grl=False)
    assert apply_grl == [False]
    assert len(parsed) == 4


def test_handler_never_applies_the_grl_to_mc(tmp_path):
    events = ak.Array({"Electrons": [[{"pt": 30_000.0, "eta": 0.1, "phi": 0.2}]] * 3})

    apply_grl, parsed = _parse(tmp_path, events, parse_mc=True, apply_grl=True)

    assert apply_grl == [False]
    assert len(parsed) == 3
