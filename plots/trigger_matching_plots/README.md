# Collision-data trigger-matching validation plots

Run the same collision-data file list and analysis configuration twice: once
with lepton trigger matching enabled, and once with it disabled. Generate a
separate plot directory for each completed run:

```bash
python plots/trigger_matching_plots/compare.py \
  --input /path/to/trigger_enabled_run \
  --output-dir plots/trigger_matching_plots/enabled

python plots/trigger_matching_plots/compare.py \
  --input /path/to/trigger_disabled_run \
  --output-dir plots/trigger_matching_plots/disabled
```

The command writes PNGs and `summary.json` to
`plots/trigger_matching_plots/generated` by default. The figures use identical
fixed binning and formatting, so compare the enabled and disabled directories
visually.

The figures deliberately include both raw yields (`01_event_yields.png`) and
unit-area shapes:

- leading and inclusive muon \(p_T\), where the enabled sample should show the
  trigger turn-on and should be stable on the plateau;
- jet and b-jet multiplicity, plus leading-jet \(p_T\), which should not be
  radically reshaped on the muon plateau;
- a data control region requiring a leading muon with \(p_T\geq27\) GeV,
  \(\geq4\) jets, and \(\geq1\) b-jet. Its yield and muon spectrum are useful
  semileptonic-top-enriched sanity checks.

These are collision-data comparisons, not truth-efficiency plots. The final
parsed `events` tree intentionally excludes temporary `_triggerMatch` and
`_dataRunNumber` fields, so it cannot make a run-number-versus-matched-event
plot or independently re-read the source trigger decorations. For that
lower-level audit, inspect raw DAOD_PHYSLITE inputs before parsing. Also keep
the run inputs identical and, for a strict selected-data comparison, use files
that carry the relevant trigger-decoration branch; undecorated real-data files
are intentionally retained unfiltered by the parser.
