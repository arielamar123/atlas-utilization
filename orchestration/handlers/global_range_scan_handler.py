"""Global histogram-range scan state.

This state is an internal prerequisite of BumpNet histogram creation. It scans
the processed SQLite shards exactly once and writes the shared bin ranges before
the histogram state fills any bins.
"""

from pathlib import Path
from typing import Optional

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler


class GlobalRangeScanHandler(StateHandler):
    """Compute and persist global histogram ranges for processed SQLite data."""

    @staticmethod
    def scan(
        input_dir: str,
        output_path: str,
        exclude_outliers: bool,
        file_list: Optional[list[str]] = None,
    ) -> dict:
        from services.pipelines.histograms_pipeline import (
            compute_global_ranges,
            save_global_ranges,
        )

        input_path = Path(input_dir)
        if file_list is None:
            sqlite_files = sorted(path.name for path in input_path.glob("*.sqlite"))
        else:
            sqlite_files = sorted({
                Path(filename).name
                for filename in file_list
                if filename.endswith(".sqlite")
                and (input_path / Path(filename).name).exists()
            })

        if not sqlite_files:
            raise RuntimeError(
                f"No processed SQLite files available for the global-range scan in {input_path}"
            )

        ranges = compute_global_ranges(
            sqlite_files,
            str(input_path),
            exclude_outliers=exclude_outliers,
        )
        if not ranges:
            raise RuntimeError(
                f"Global-range scan produced no histogram ranges from {len(sqlite_files)} shard(s)"
            )

        range_path = Path(output_path)
        range_path.parent.mkdir(parents=True, exist_ok=True)
        save_global_ranges(ranges, str(range_path))
        return {
            "path": str(range_path),
            "input_files": sqlite_files,
            "range_count": len(ranges),
        }

    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        self._log_state_entry(context)

        hc = context.config.histogram_creation_config
        if hc is None:
            raise RuntimeError("histogram_creation_config is required for global-range scan")
        if not hc.global_ranges_path:
            raise RuntimeError("global_ranges_path is required for BumpNet histogram creation")

        selected_files = context.processed_files or None
        self.logger.info(
            f"Scanning processed histogram inputs once for global ranges: {hc.input_dir}"
        )
        scan_summary = self.scan(
            input_dir=hc.input_dir,
            output_path=hc.global_ranges_path,
            exclude_outliers=hc.exclude_outliers,
            file_list=selected_files,
        )
        self.logger.info(
            f"Global-range scan complete: {scan_summary['range_count']} ranges from "
            f"{len(scan_summary['input_files'])} SQLite shard(s)"
        )

        updated = context.with_custom_data("global_range_scan", scan_summary)
        next_state = self._determine_next_state(updated)
        self._log_state_exit(context, next_state)
        return updated, next_state
