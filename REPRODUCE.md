# Reproducing the analysis

## 1. Environment

```
python -m venv venv
. venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data

Neither dataset is redistributed here.

- ERP CORE visual P3: https://doi.org/10.18115/D5JW4R
- OpenNeuro ds006018 v1.2.2: https://openneuro.org/datasets/ds006018/versions/1.2.2 (read through EEGDash)

Paths are set in `pipeline/config.py` and can be overridden by environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `P300_BASE` | folder holding the data and results folders | two levels above `pipeline/config.py` |
| `P300_ERPCORE` | ERP CORE P3 folder | `$P300_BASE/data/erp_core_P3` |
| `P300_DS006018` | ds006018 folder | `$P300_BASE/data_ds006018` |
| `P300_RESULTS` | results folder | `$P300_BASE/results/P300` |

Every step also accepts `--results DIR`.

## 3. Full run

```
cd pipeline
python run_all.py --results ../results
```

`run_all.py` runs the 13 steps in a fixed order, skips steps whose outputs exist (`--force` reruns them), validates every output (participant and trial counts, missing and non-finite values, convergence, draw counts) and writes `pipeline_manifest_<time>.csv` and `pipeline_summary_<time>.txt`. Steps 6 and 13 rerun every time. Useful options: `--dry-run` (print the plan), `--steps 9,11` (step numbers), `--only robustness,simulation` (step ids), `--stop-on-fail`.

## 4. Rebuilding the reported values from the included results

The included `results/` folder is sufficient to rerun steps 6, 9, 11, 12 and 13 and to redraw every figure without the EEG data:

```
cd pipeline
python run_all.py --results ../results --only followup,summaries
cd ../figures
python fig1_waveforms.py ../results fig1.png
python fig2_profile_agreement.py ../results/paper_values.json fig2.png
python fig3_reference_free.py ../results/paper_values.json fig3.png
python fig4_complexity.py ../results/paper_values.json fig4.png
python figS1_persistence.py ../results/paper_values.json figS1.png
```

When the step 5 trial files are absent, step 6 leaves `followup_lags_seconds.csv`, `followup_soa.csv` and `followup_timescale_bounds.csv` unchanged and rewrites `followup_summary.txt` without the lag section. Steps 1, 5, 7 and 8 and the per-dataset part of step 10 need the EEG data; steps 2–4 need the trial-level files of step 1, which are included only for the six cells listed in `README.md`.

Steps 9, 11, 12 and 13 use fixed seeds (12345; step 12 seeds each replication from the seed, the replication number and a CRC-32 of the scenario name), so a rerun gives identical files, independent of the number of workers in step 12.

## 5. Tables and figures

Every value in the tables and figures is read from `results/paper_values.json`, written by step 13 from the files below. Tag `r150_g05_ic20` is the recording reference as distributed; `_cz`, `_mastoid` and `_average` mark the other references; `_nobase` marks features without baseline correction. "Reference-specific pipeline" files are `matched_*`, `canonical_lmm_*`, `mechanism_*`, `symmetry_*`, `erp_*` and `features_*`; "common cleaning" files are `common_*`, `reffree_*`, `robustness_*` and `simulation_*`.

| Item | Content | Result files | Step |
|---|---|---|---|
| Table 1 | studies relating pre-stimulus or early activity to later evoked responses (literature) | none | – |
| Table 2 | matched-contrast battery | `matched_erpcore_r150_g05_ic20.csv` | 3 |
| Table 3 | participants and trials | `common_subjects_{dataset}.csv`, `common_persubject_{dataset}_{free,onset}.csv` | 8 |
| Table 4 | positive control | `mechanism_*.csv`, `erp_*.csv` (Pz, 300–600 ms), `features_*_log.csv` | 4, 7, 1 |
| Table 5 | profile agreement after common cleaning | `common_transport_free.csv`, `common_transport_onset.csv` | 9 |
| Table 6 | neighbouring-trial adjustment | `robustness_neighbour_adjustment.csv`, `robustness_neighbour_share.csv` | 11 |
| Table 7 | complexity measures, equivalence | `common_dbeta_free.csv` | 9 |
| Table S1 | agreement of Δβ over 21 models | `common_persubject_{dataset}_free.csv` | 9, 13 |
| Table S2 | pooled mixed models | `canonical_lmm_erpcore_r150_g05_ic20.csv` | 2 |
| Table S3 | Δβ for all 21 measures | `common_dbeta_free.csv` | 9 |
| Table S4 | lags in seconds | `followup_lags_seconds.csv` | 6 |
| Table S5 | location statistics under re-referencing | `common_location_free.csv` | 9 |
| Table S6 | absolute disagreement and interaction | `common_absdis_free.csv`, `robustness_absolute_interaction.csv` | 9, 11 |
| Table S7 | baseline correction and reference | `matched_persubject_*[_nobase]_ic20.csv` | 3, 13 |
| Table S8 | background symmetry | `symmetry_*.csv`, `matched_persubject_*.csv` | 7, 13 |
| Table S9 | battery composition | `common_robust_free.csv` | 9 |
| Table S10 | condition check | `common_condition_summary.csv` | 9 |
| Table S11 | sensitivity to the cleaning pipeline | `common_persubject_{dataset}_free.csv`, `matched_persubject_*.csv` | 13 |
| Table S12 | Fisher-z agreement | `common_transport_free.csv`, `common_transport_onset.csv` | 9 |
| Table S13 | minimum target trials | `common_threshold_free.csv` | 9 |
| Table S14 | reference-free derivation | `reffree_agreement.csv`, `reffree_dbeta.csv`, `reffree_poscontrol_summary.csv` | 10, 13 |
| Table S15 | event timing and real-trial count | `reffree_agreement.csv`, `reffree_dbeta.csv`, `reffree_poscontrol_summary.csv` | 10, 13 |
| Table S16 | size of real-trial slopes, noise floor | `robustness_summary.csv` | 11 |
| Table S17 | percentile and BCa intervals | `robustness_bca_agreement.csv` | 11 |
| Table S18 | trial-level persistence | `features_{dataset}_r150_g05_{mastoid,average}_ic20.csv`, `mechanism_*.csv`, `followup_soa.csv` | 1, 4, 6, 13 |
| Table S19 | simulation: calibration | `simulation_calibration.csv`, `simulation_summary.csv` | 12 |
| Table S20 | simulation: profile agreement | `simulation_agreement.csv` | 12 |
| Table S21 | distribution of participant-level slopes, robust location estimates | `robustness_distribution.csv`, `robustness_location_agreement.csv` | 11 |
| Figure 1 | grand-average waveforms | `erp_*.csv` | 7 |
| Figure 2 | coupling profiles | `common_profiles_free.csv`, `common_transport_free.csv` | 9 |
| Figure 3 | reference-free profiles | `reffree_dbeta.csv`, `reffree_agreement.csv` | 10 |
| Figure 4 | complexity measures | `common_dbeta_free.csv` | 9 |
| Figure S1 | trial-level persistence | as Table S18 | 13 |

The cross-dataset agreement of Δβ under the reference-specific pipeline is in `followup_reference_agreement.csv` (step 6). Holm adjustment over 21 and 17 models with interval-inversion and null-centred bootstrap p values (section 2.5) is in `robustness_tests.csv` (step 11); the difference between group-mean slopes and group-mean correlations of the complexity measures (section 2.9) is the `slope_correlation` entry of `paper_values.json` (step 13).
