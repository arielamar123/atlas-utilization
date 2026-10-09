"""Connect the real-data trigger debugger to a normal pipeline run.

Production code touches only :meth:`TriggerDebugHook.from_parsing`,
:meth:`TriggerDebugHook.observe`, :meth:`TriggerDebugHook.finish` and
:func:`merge_batch_outputs`.  Deleting this directory and those calls removes
the diagnostics.  Failures here are logged and never change parsing results.

Output layout (``<run_dir>/plots/trigger_debug/``):

- single job: plots, ``trigger_counts.txt`` and ``trigger_counts.csv``
  directly in that directory;
- batch job N: the same files in ``batch_N/`` plus a raw-value state file,
  which ``python main.py --merge-only`` combines into the parent directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEBUG_SUBDIR = Path("plots") / "trigger_debug"
STATE_FILE = "trigger_debug_state.npz"


def debug_dir(run_dir: str | Path) -> Path:
    """Return the trigger-debug directory inside a pipeline run directory."""
    return Path(run_dir) / DEBUG_SUBDIR


def _write(debugger, source: str) -> None:
    rows = debugger.write()
    plots = [path for path in debugger.plot_paths() if path.exists()]
    output_dir = debugger.output_dir.resolve()
    if plots:
        logger.info(
            "Real-data trigger debug: created %d plot(s) plus trigger_counts.txt/.csv "
            "for %d chain(s) from %s in %s",
            len(plots), len(rows), source, output_dir,
        )
    else:
        logger.warning(
            "Real-data trigger debug: created 0 plots from %s because no event was in "
            "a configured chain's run range with an offline lepton of its flavour; "
            "counts written to %s",
            source, output_dir,
        )


class TriggerDebugHook:
    """Accumulate pre-trigger-selection collision-data events across a run."""

    def __init__(
        self,
        output_dir: Path | None,
        skip_reason: str | None = None,
        state_path: Path | None = None,
    ):
        self.skip_reason = skip_reason
        self.state_path = state_path
        self._batches = 0
        self._debugger = None
        if output_dir is not None:
            # Imported lazily so MC and trigger-disabled runs do not load it.
            from .real_data_trigger_debug import RealDataTriggerDebugger
            self._debugger = RealDataTriggerDebugger(output_dir)

    @classmethod
    def from_parsing(cls, parsing_config, trigger_enabled: bool, batch_job_index=None):
        """Create an active hook only for collision data with trigger selection."""
        if parsing_config.parse_mc:
            return cls(None, "parse_mc is true (Monte Carlo); the debugger is for collision data only")
        if not trigger_enabled:
            return cls(None, "trigger_config.enabled is false")
        # main.py places jobs_logs_path at <run_dir>/logs.
        output_dir = debug_dir(Path(parsing_config.jobs_logs_path).resolve().parent)
        if batch_job_index is None:
            return cls(output_dir)
        output_dir = output_dir / f"batch_{batch_job_index}"
        return cls(output_dir, state_path=output_dir / STATE_FILE)

    @property
    def active(self) -> bool:
        return self._debugger is not None

    def observe(self, events) -> None:
        """Record one batch; call before ``apply_trigger_selection``."""
        if self._debugger is None:
            return
        try:
            self._debugger.add_events(events)
            self._batches += 1
        except Exception:
            logger.exception(
                "Real-data trigger debugging disabled after a failure; parsing continues unchanged"
            )
            self.skip_reason = "the debugger failed while accumulating events (see error above)"
            self._debugger = None

    def finish(self) -> None:
        """Write plots and counts for everything observed, or log why not."""
        if self._debugger is None:
            logger.info("Real-data trigger debug plots skipped: %s", self.skip_reason)
            return
        try:
            _write(self._debugger, f"{self._batches} parsed batch(es)")
            if self.state_path is not None:
                self._debugger.save_state(self.state_path)
                logger.info(
                    "Real-data trigger debug state for merging saved to %s",
                    self.state_path.resolve(),
                )
        except Exception:
            logger.exception("Failed to write real-data trigger debug output")


def merge_batch_outputs(run_dir: str | Path) -> None:
    """Combine every batch job's state into ``<run_dir>/plots/trigger_debug/``."""
    output_dir = debug_dir(run_dir)
    states = sorted(output_dir.glob(f"batch_*/{STATE_FILE}"))
    if not states:
        logger.info(
            "Real-data trigger debug merge skipped: no batch state files under %s",
            output_dir.resolve(),
        )
        return
    try:
        from .real_data_trigger_debug import RealDataTriggerDebugger
        debugger = RealDataTriggerDebugger(output_dir)
        for state in states:
            debugger.load_state(state)
        _write(debugger, f"{len(states)} batch job(s)")
    except Exception:
        logger.exception("Failed to merge real-data trigger debug output")
