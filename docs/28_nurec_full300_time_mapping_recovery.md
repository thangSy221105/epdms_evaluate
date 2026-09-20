# NuRec full-300 time-mapping recovery

`recover_nurec_time_mapping_full300.py` inventories local time-related artifacts
and writes a new recovery report. It never estimates an offset from timestamp
minima, overlap, nearest numeric values, or a global statistic.

The accepted five-clip evidence is treated as immutable. Its established method
is a per-clip rebase validated by an explicit PAI/NuRec pose timeline: a stable
per-clip offset, unit scale, equal relative duration, at least two recorded
relative timestamp correspondences, and the independent trajectory check. Only
the current-300 clip that is actually present in that accepted artifact can be
reused automatically.

Example:

```powershell
python scripts/recover_nurec_time_mapping_full300.py
```

The canonical `configs/nurec_time_contract_full300.jsonl` contains only rows
whose mapping is verified. Unresolved clips are represented in the audit and
minimal-data-plan reports, not as fake contract rows. `sequence_tracks.json` is
used only for post-hoc contradiction/coverage diagnostics; it cannot prove a
clock origin or create an offset.
