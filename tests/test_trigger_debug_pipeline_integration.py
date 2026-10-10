"""The real-data trigger debugger runs automatically inside ParsingHandler."""

import csv

import pytest

ak = pytest.importorskip("awkward")
np = pytest.importorskip("numpy")
uproot = pytest.importorskip("uproot")

from domain.config import PipelineConfig
from domain.events import EventBatch
from orchestration.context import PipelineContext
from orchestration.handlers.parsing_handler import ParsingHandler
from orchestration.states import PipelineState
from pipeline.executor import PipelineExecutor
from plots.trigger_debug import pipeline_hook, real_data_trigger_debug
from plots.trigger_debug.pipeline_hook import TriggerDebugHook
from services.parsing import schemas
from services.parsing.event_accumulator import EventAccumulator
from utils.paths import update_config_paths_with_run_dir

CHAINS = sorted({
    chain
    for flavours in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS.values()
    for chains in flavours.values()
    for chain in chains
})
RUN_2015, RUN_2016 = 276300, 300000


def _lepton(pt_gev):
    return {"pt": pt_gev * 1000.0, "eta": 0.1, "phi": 0.2}


def _events(rows):
    """rows: (run, electron pTs, muon pTs, fired chains[, fired chains without a matched lepton])."""
    rows = [row if len(row) == 5 else (*row, set()) for row in rows]
    return ak.Array({
        "_dataRunNumber": [run for run, *_ in rows],
        "_triggerDecision": {
            chain: [chain in fired for _, _, _, fired, _ in rows] for chain in CHAINS
        },
        "_triggerMatch": {
            schemas.data_trigger_match_branch(chain): [
                chain in fired and chain not in unmatched for _, _, _, fired, unmatched in rows
            ]
            for chain in CHAINS
        },
        "Electrons": [[_lepton(pt) for pt in electrons] for _, electrons, _, _, _ in rows],
        "Muons": [[_lepton(pt) for pt in muons] for _, _, muons, _, _ in rows],
    })


# Two files, i.e. two parsed batches, covering both run years.
FILES = {
    "data_2015.root": _events([
        (RUN_2015, [30.0], [], {"HLT_e24_lhmedium_L1EM20VH"}),
        (RUN_2015, [18.0], [], set()),
        (RUN_2015, [], [32.0], {"HLT_mu20_iloose_L1MU15"}),
    ]),
    "data_2016.root": _events([
        (RUN_2016, [40.0], [50.0], {"HLT_e26_lhtight_nod0_ivarloose"}),
        (RUN_2016, [10.0], [], set()),
        (RUN_2016, [], [], set()),
        # Fired, but no offline muon is matched to the chain: not retained.
        (RUN_2016, [], [45.0], {"HLT_mu26_ivarmedium"}, {"HLT_mu26_ivarmedium"}),
    ]),
}

# (relevant, passed, failed) over both files; failures stay in the denominator.
EXPECTED = {
    "HLT_e24_lhmedium_L1EM20VH": (2, 1, 1),
    "HLT_e60_lhmedium": (2, 0, 2),
    "HLT_e120_lhloose": (2, 0, 2),
    "HLT_mu20_iloose_L1MU15": (1, 1, 0),
    "HLT_mu40": (1, 0, 1),
    "HLT_e26_lhtight_nod0_ivarloose": (2, 1, 1),
    "HLT_e60_lhmedium_nod0": (2, 0, 2),
    "HLT_e140_lhloose_nod0": (2, 0, 2),
    "HLT_mu26_ivarmedium": (2, 0, 2),
    "HLT_mu50": (2, 0, 2),
}


class StubProcessor:
    """Yields the in-memory FILES exactly as ThreadedFileProcessor would."""

    def process_files(self, file_urls, release_year, on_success=None, **_):
        for file_id, url in enumerate(file_urls):
            events = FILES[url]
            if on_success:
                on_success(url, len(events), 0.0)
            yield EventBatch(
                events=events, file_id=file_id, release_year=release_year,
                size_bytes=events.layout.nbytes, event_count=len(events),
                processing_time_sec=0.0, file_url=url,
            )


def _config(run_dir, parse_mc=False, trigger=True, batch=None):
    run_metadata = {}
    if batch is not None:
        run_metadata = {"batch_job_index": batch, "total_batch_jobs": 2}
    config_dict = update_config_paths_with_run_dir({
        "tasks": {"do_parsing": True},
        "trigger_config": {"enabled": trigger},
        "run_metadata": run_metadata,
        "parsing_task_config": {
            "output_path": "./output/parsed_data",
            "file_urls_path": "./output/metadata_cache.json",
            "jobs_logs_path": "./output/logs",
            "release_years": ["2024r-pp"],
            "parse_mc": parse_mc,
            "show_progress_bar": False,
        },
    }, str(run_dir))
    return PipelineConfig.from_dict(config_dict)


def _run(run_dir, **options):
    context = PipelineContext(
        config=_config(run_dir, **options),
        current_state=PipelineState.PARSING,
        metadata={"2024r-pp": list(FILES), "2024r-pp_mc": list(FILES)},
    )
    handler = ParsingHandler(None, StubProcessor(), EventAccumulator(10**9))
    context, _ = handler.handle(context)
    return context


def _counts(directory):
    with (directory / "trigger_counts.csv").open(encoding="utf-8") as handle:
        return {
            row["chain"]: (
                int(row["relevant_events"]),
                int(row["passed_chain"]),
                int(row["did_not_pass_chain"]),
            )
            for row in csv.DictReader(handle)
        }


def _parsed(context):
    events = []
    for path in context.parsed_files:
        with uproot.open(path) as handle:
            events.extend(ak.to_list(handle["events"].arrays()))
    return events


def test_normal_data_run_writes_debug_plots_and_counts(tmp_path, caplog):
    caplog.set_level("INFO")
    context = _run(tmp_path)

    output = tmp_path / "plots" / "trigger_debug"
    assert _counts(output) == EXPECTED
    assert (output / "trigger_counts.txt").exists()
    plots = sorted(path.name for path in output.glob("*.png"))
    assert plots == sorted(
        f"{chain}_{kind}.png" for chain in EXPECTED for kind in ("leading_pt", "response")
    )
    assert f"created {len(plots)} plot(s)" in caplog.text
    assert str(output.resolve()) in caplog.text
    # Trigger selection itself is unaffected: three events fired a chain
    # with an offline lepton of the matching flavour.
    assert context.parsing_stats.trigger_events_before == 7
    assert context.parsing_stats.trigger_events_after == 3


def test_parsing_output_is_identical_with_and_without_debugger(tmp_path, monkeypatch):
    with_debug = _parsed(_run(tmp_path / "with"))

    monkeypatch.setattr(
        TriggerDebugHook, "from_parsing",
        classmethod(lambda cls, *args: cls(None, "disabled by test")),
    )
    without_debug = _parsed(_run(tmp_path / "without"))

    assert with_debug == without_debug
    assert len(with_debug) == 3
    assert not (tmp_path / "without" / "plots" / "trigger_debug").exists()


@pytest.mark.parametrize(
    ("parse_mc", "trigger", "reason"),
    [(True, True, "parse_mc is true"), (False, False, "trigger_config.enabled is false")],
)
def test_mc_or_disabled_trigger_never_invokes_debugger(tmp_path, monkeypatch, caplog, parse_mc, trigger, reason):
    def fail(*_args, **_kwargs):
        raise AssertionError("real-data trigger debugger must not be constructed")

    monkeypatch.setattr(real_data_trigger_debug, "RealDataTriggerDebugger", fail)
    caplog.set_level("INFO")
    _run(tmp_path, parse_mc=parse_mc, trigger=trigger)

    assert not (tmp_path / "plots" / "trigger_debug").exists()
    assert f"Real-data trigger debug plots skipped: {reason}" in caplog.text


def test_batch_jobs_do_not_collide_and_merge_to_single_job_counts(tmp_path):
    for batch in (1, 2):
        _run(tmp_path, batch=batch)

    output = tmp_path / "plots" / "trigger_debug"
    batch_dirs = sorted(path.name for path in output.glob("batch_*"))
    assert batch_dirs == ["batch_1", "batch_2"]
    assert not (output / "trigger_counts.csv").exists()

    # The existing --merge-only entry point combines the batch outputs.
    PipelineExecutor(_config(tmp_path)).merge_outputs(str(tmp_path))
    assert _counts(output) == EXPECTED

    # Failing events are kept with their pT so they enter the histograms.
    merged_fail = []
    for batch_dir in batch_dirs:
        with np.load(output / batch_dir / pipeline_hook.STATE_FILE) as state:
            merged_fail.extend(state["HLT_e60_lhmedium__fail"].tolist())
    assert sorted(merged_fail) == [18.0, 30.0]


def test_file_without_trigger_information_is_skipped_not_fatal(tmp_path, monkeypatch, caplog):
    no_decision = ak.Array({
        "_dataRunNumber": [RUN_2015],
        "Electrons": [[_lepton(30.0)]],
        "Muons": [[]],
    })
    monkeypatch.setitem(FILES, "broken.root", no_decision)
    caplog.set_level("INFO")

    context = _run(tmp_path)

    assert "Skipping 1 events of broken.root" in caplog.text
    assert context.parsing_stats.trigger_events_before == 8
    assert context.parsing_stats.trigger_events_after == 3
    assert _counts(tmp_path / "plots" / "trigger_debug") == EXPECTED


def test_run_with_no_surviving_events_does_not_write_an_empty_chunk(tmp_path, monkeypatch, caplog):
    for name in list(FILES):
        monkeypatch.setitem(FILES, name, _events([(RUN_2015, [18.0], [], set())]))
    caplog.set_level("INFO")

    context = _run(tmp_path)

    assert context.parsed_files == []
    assert not list((tmp_path / "parsed_data").glob("*.root"))
    assert "No events survived the parsing selections" in caplog.text


@pytest.mark.parametrize(("parse_mc", "trigger", "filtered"), [
    (False, True, True), (False, False, True), (True, True, False), (True, False, False),
])
def test_collision_data_uses_the_same_runs_with_and_without_trigger_selection(
    tmp_path, monkeypatch, parse_mc, trigger, filtered
):
    import orchestration.handlers.parsing_handler as handler_module

    calls = []

    def record(metadata):
        calls.append(sorted(metadata))
        return metadata

    monkeypatch.setattr(handler_module, "filter_collision_files_by_run", record)
    _run(tmp_path, parse_mc=parse_mc, trigger=trigger)

    assert bool(calls) == filtered
