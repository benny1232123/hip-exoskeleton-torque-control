# Legacy scripts

Superseded, kept for provenance only. Do not use these for new results.

| Script | Scope | Superseded by |
|---|---|---|
| `pan2023_reproduction_v2.py` | Part 1 only (≈7 subjects) controller demo + Gaussian component plots | `pan2023_full_dataset.py` (22 subjects) |
| `visualize_gait.py` | Part 1 only per-subject angle/moment curves | `pan2023_full_dataset.py` Figure 1 |

They are retained because they were the first working versions of the
controller demo and their output filenames (`*_demo.png`,
`*_profiles.png`, `*_part1.png`) are referenced in earlier notes.

Both read Part 1 only, i.e. `processed/levelground_part1.csv`, which is not
tracked by git — regenerate it with `python process_part1.py` first.

Paths here are resolved relative to this repository (no absolute paths), so the
scripts remain runnable after cloning.
