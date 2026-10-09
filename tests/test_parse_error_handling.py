"""Only network/read failures are retried; content errors skip the file at once."""

import pytest

ak = pytest.importorskip("awkward")

from services.parsing import file_parser as fp
from services.parsing import schemas
from services.parsing import threaded_processor as tp
from services.parsing.file_parser import FileParser, InvalidFileContentError


def _parse_raising(monkeypatch, error):
    def boom(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(fp, "open_root_file", boom)
    return FileParser.parse_file("root://host//file.root", ["CollectionTree"], "2024r-pp")


def test_network_errors_return_none_for_retry(monkeypatch):
    assert _parse_raising(monkeypatch, OSError("[ERROR] Operation expired")) is None
    assert _parse_raising(monkeypatch, TimeoutError("read timed out")) is None


@pytest.mark.parametrize("error", [KeyError("CollectionTree"), ValueError("bad menu"), FileNotFoundError("gone")])
def test_content_errors_raise_without_retry(monkeypatch, error):
    with pytest.raises(InvalidFileContentError):
        _parse_raising(monkeypatch, error)


class _Parser:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def parse_file(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _run(parser, monkeypatch):
    sleeps, errors, successes = [], [], []
    monkeypatch.setattr("time.sleep", sleeps.append)
    processor = tp.ThreadedFileProcessor(parser, max_threads=1, show_progress=False)
    batches = list(processor.process_files(
        ["f.root"], ["CollectionTree"], "2024r-pp",
        on_success=lambda *a: successes.append(a), on_error=lambda url, e: errors.append(e),
    ))
    return batches, sleeps, errors, successes


def test_processor_retries_only_read_failures(monkeypatch):
    events = ak.Array({"x": [1, 2]})
    parser = _Parser([None, None, events])
    batches, sleeps, errors, successes = _run(parser, monkeypatch)

    assert parser.calls == 3 and sleeps == [30, 60]
    assert len(batches) == 1 and not errors and len(successes) == 1


def test_processor_skips_content_errors_immediately(monkeypatch, caplog):
    parser = _Parser([InvalidFileContentError("No TriggerMenuJson_HLT entry for SMK 9")])
    batches, sleeps, errors, successes = _run(parser, monkeypatch)

    assert parser.calls == 1 and sleeps == []
    assert batches == [] and not successes
    assert len(errors) == 1 and "SMK 9" in str(errors[0])
    assert "SMK 9" in caplog.text


def test_collision_file_without_decision_branches_is_rejected():
    run_only = {"_dataRunNumber": {schemas.DATA_RUN_NUMBER_BRANCH: "_dataRunNumber"}}
    with pytest.raises(InvalidFileContentError, match="efPassedPhysics"):
        FileParser._require_collision_trigger_branches(run_only, "data.root")

    complete = dict(run_only, _triggerDecisionRaw={
        schemas.TRIGGER_DECISION_SMK_BRANCH: "smk",
        schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH: "hlt",
    })
    FileParser._require_collision_trigger_branches(complete, "data.root")
