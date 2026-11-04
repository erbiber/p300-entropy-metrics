# Legacy outputs — retained for provenance, not regenerable

These four files were produced by an earlier version of `run_phase013.py` whose output
suffix did not include the `_K{k}` component. The current script always writes
`phase013_<quantity>_<config>_K<k>[...].csv`, so **these filenames can no longer be
produced by the deposited pipeline**.

They are kept here so that the figures and tables built from them remain traceable.
Nothing in `scripts/` reads this directory.

## Status of each file

| file | status |
|---|---|
| `phase013_dbeta_config4.csv` | **Superseded.** Contains 4 models x 2 datasets (8 rows). Every value is identical, to the last digit, to the corresponding rows of `../phase013_dbeta_config4_K1000.csv`, which contains all 9 models (18 rows). Use the K1000 file. |
| `phase013_dbeta_config2.csv` | Legacy. No `_K1000` counterpart exists for config-2 Delta beta, so these values have no regenerable replacement. Regenerate with `python scripts/run_phase013.py --config config2 --k 1000` if the config-2 contrast is needed. |
| `phase013_interaction_config4.csv` | Legacy. Superseded by `../phase013_interaction_config4_K1000.csv`. |
| `phase013_interaction_config2.csv` | Legacy. No regenerable counterpart; see the config-2 note above. |

Verified 2026-11-04: `phase013_dbeta_config4.csv` and the K1000 file agree exactly on all
eight shared rows across dbeta, beta_real, beta_pseudo_mean, bca_lo, bca_hi, surrogate_p,
n_used and draws_median.
