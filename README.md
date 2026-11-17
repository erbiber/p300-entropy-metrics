# Cross-channel Single-trial P300 Coupling Follows the EEG Reference and Disappears Without One

Analysis code, result files and figure scripts for the study *Cross-channel Single-trial P300 Coupling Follows the EEG Reference and Disappears Without One* 

Seventeen summaries of early (0–150 ms) and pre-stimulus activity at Fz are related to single-trial P300 amplitude at Pz in two public visual-oddball datasets, in real trials and in event-free pseudotrials, under the recording reference as distributed, linked mastoids, the common average, Cz and a reference-free surface Laplacian, on identical trials. Agreement between datasets is the correlation, and the absolute difference, between the two datasets' coefficient profiles.

- As distributed, the datasets disagree (r = −0.83 [−0.91, −0.35]); with either common reference they agree (+0.96 and +0.97), and within each dataset the mastoid and average profiles are mirror images.
- Event-free pseudotrials reverse in the same way, so background signal shared by both electrodes is sufficient to produce the change.
- Without a reference the P300 remains, but all 17 cross-channel coefficients are equivalent to zero within ±0.05 and their mean |β| is close to the sampling-noise floor.



## Repository layout

```
.
├── README.md
├── REPRODUCE.md            step-by-step reproduction, and a map from every table and figure to its result files
├── requirements.txt        package versions used for the reported results
├── LICENSE
├── pipeline/               analysis code (Python)
│   ├── run_all.py          runs steps 1–13 in a fixed order, skips completed steps, validates every output
│   ├── config.py           paths, filters, windows, thresholds (all overridable by environment variables)
│   ├── spec.py             definitions of the 21 matched models and the 17 pooled models
│   ├── estimators.py       robust standardisation, OLS slopes, BCa intervals
│   ├── placement.py        pseudotrial placement
│   ├── lib/                data access, preprocessing, feature extraction, parity features
│   └── run/                01_features.py … 13_summaries.py
├── results/                every result file used in the paper (CSV), run summaries and paper_values.json
└── figures/                scripts and PNG files for Figures 1–4 and S1
```

## Datasets

The EEG data are not included. Both datasets are public and de-identified.

| Dataset | Source | Recording | Used here |
|---|---|---|---|
| ERP CORE, active visual oddball (P3) (Kappenman et al. 2021) | https://doi.org/10.18115/D5JW4R | BioSemi ActiveTwo, 1024 Hz, no reference electrode (CMS at PO1, DRL at PO2) | 40 listed; 35 in the primary (event-free) analysis, 38 in the comparison-rule analysis |
| OpenNeuro ds006018 v1.2.2, visual oddball (Isbell et al. 2025) | https://openneuro.org/datasets/ds006018/versions/1.2.2, read through EEGDash | Brain Products actiCHamp Plus, 500 Hz, online reference Cz, mastoids LM and RM | 98 listed; 97 analysed |

Event codes 11, 22, 33, 44 and 55 are targets and the 20 off-diagonal codes are standards in both datasets.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_features.py` | preprocessing, ICA, epoching and trial features under one reference |
| 2 | `02_canonical_lmm.py` | pooled linear mixed models on real trials |
| 3 | `03_matched.py` | matched real-minus-pseudotrial contrast (Δβ) with BCa intervals |
| 4 | `04_mechanism.py` | trial-level positive control and within-trial shared-state tests |
| 5 | `05_parity_features.py` | time-reversal parity features and event onsets |
| 6 | `06_followup.py` | equivalence, cross-dataset agreement, parity decision, lags in seconds, sampling-rate check |
| 7 | `07_from_cache.py` | features without baseline correction, participant ERP averages, background symmetry |
| 8 | `08_common.py` | one ICA cleaning, references derived after cleaning, identical trials, event-free pseudotrials |
| 9 | `09_common_analysis.py` | profile agreement, absolute disagreement, location statistics, battery robustness, Δβ with Holm and equivalence |
| 10 | `10_reference_free.py` | surface Laplacian (CSD and CSD18), delay-corrected and subsampled real trials, positive control |
| 11 | `11_robustness.py` | equivalence and noise floor of real-trial slopes, effective number of statistics, absolute-scale interaction, BCa agreement, Holm adjustment with both bootstrap p values, distribution of participant-level slopes with trimmed means and medians, neighbouring-trial adjustment |
| 12 | `12_simulation.py` | calibration of the estimators by simulation with known true slopes |
| 13 | `13_summaries.py` | collects every value reported in the tables and figures into `results/paper_values.json` |

`python run_all.py` runs all steps. Before any step, a gate checks that the pseudotrial feature code reproduces the real-trial feature code at 1024 and 500 Hz. See `REPRODUCE.md` for options, run times and the table and figure map.

## Results folder

`results/` contains the outputs of steps 1–13 from the run of 15 November 2026 (`pipeline_summary_20261115_214001.txt`, `pipeline_manifest_20261115_214001.csv`), the step 11 and 12 outputs, and `paper_values.json`. Trial-level feature files are included for the six cells used by steps 11 and 13 (`features_{erpcore,ds006018}_r150_g05[_mastoid|_average]_ic20.csv`); the remaining trial-level files of steps 1, 5 and 7 are regenerated from the raw data.

## Requirements

Python 3 with the versions in `requirements.txt` (MNE-Python 1.12.1, NumPy 2.3.1, SciPy 1.16.0, pandas 2.2.3, statsmodels 0.14.5, scikit-learn 1.7.0, antropy 0.1.9, EEGDash 0.8.2, Matplotlib 3.11.2).

## Licence

MIT (see `LICENSE`).
