"""Temporary diagnostics for collision-data lepton triggers.

The parsing stage calls only ``pipeline_hook``; it never changes parsing
results.  To remove the diagnostics, delete this directory and the
``TriggerDebugHook`` / ``merge_batch_outputs`` calls in
``orchestration/handlers/parsing_handler.py`` and ``pipeline/executor.py``.
"""
