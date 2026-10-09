# Real-data lepton-trigger debugging

These diagnostics run automatically during `python main.py` when parsing
collision data (`parse_mc: false`) with `trigger_config.enabled: true`.  No
extra command is needed.  They do not change MC trigger matching or the
production event selection.  For MC runs or with trigger selection disabled,
the parsing log says why the plots were skipped.

The parsing handler passes each parsed batch to `pipeline_hook.py` *before*
`apply_trigger_selection()`.  Counts and histograms therefore cover the same
pre-selection events, accumulated over every file in the run.

Pipeline output goes to `<run_dir>/plots/trigger_debug/`.  Batch jobs
(`--batch-job-index N`) write to `batch_N/` in that directory, along with
`trigger_debug_state.npz`.  `python main.py --merge-only --run-dir <run_dir>`
then merges every batch into the parent directory.

To remove the diagnostics, delete this directory and the `TriggerDebugHook` /
`merge_batch_outputs` calls in `orchestration/handlers/parsing_handler.py` and
`pipeline/executor.py`.

## Standalone use

The module can also be run on chosen files, inside the project Docker
container:

```bash
python -m plots.trigger_debug.real_data_trigger_debug \
  --input 'root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/data15_13TeV/DAOD_PHYSLITE.37001628._000029.pool.root.1' \
  --input 'ROOT_URI_FOR_A_2016_FILE'
```

Standalone output is written to `plots/trigger_debug/generated/`.  Both modes produce:

- `trigger_counts.txt` and `trigger_counts.csv` are the authoritative,
  per-chain counters.
- `*_leading_pt.png` shows the same pass/fail counts in its legend, for events
  in that trigger's run range containing the corresponding offline flavour.
- `*_response.png` is the same decision expressed as a binned event-level
  fired fraction versus leading offline lepton pT.

The dashed line is the nominal HLT pT threshold encoded in the trigger name.
It is a debugging reference, not an offline-cut boundary: trigger ID,
isolation, L1 seed, detector response, and the event-level nature of readable
`xTrigDecision` bits mean the distributions need not turn on sharply there.

An event passes a chain, as in the real-data selection, when the chain's HLT
physics decision (`xTrigDecision.efPassedPhysics`) fired and an offline lepton
is trigger-matched to it (`AnalysisTrigMatch_<chain>AuxDyn.TrigMatchedObjects`,
as in MC).  Only runs inside the MC-modelled ranges are counted.
