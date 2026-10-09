# Real-data lepton-trigger debugging

This is deliberately isolated from the pipeline so it can be deleted once the
collision-data trigger investigation is complete.  It does not change MC
trigger matching or production event selection.

Run inside the project Docker container, supplying at least one 2015 and one
2016 PHYSLITE file:

```bash
python -m plots.trigger_debug.real_data_trigger_debug \
  --input 'root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/data15_13TeV/DAOD_PHYSLITE.37001628._000029.pool.root.1' \
  --input 'ROOT_URI_FOR_A_2016_FILE'
```

Output is written to `plots/trigger_debug/generated/`:

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

The current real-data selection decodes `xTrigDecision.TAV` bits and requires
an offline lepton of the corresponding flavour.  It cannot inspect the raw
PHYSLITE `AnalysisTrigMatch` ElementLinks with uproot, so these plots do not
claim object-level matching.
