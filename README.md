## Overview

This repository contains the complete analysis pipeline for a study examining whether conventional early-window endpoint-summary measures (mean amplitude, RMS, entropy, Hjorth parameters, etc.) capture genuine stimulus-locked information about P300 amplitude in the active visual oddball paradigm. Every coupling estimate is accompanied by a matched pseudotrial control estimate; the inferential quantity is the difference in standardized coupling between real and pseudotrials (Δβ), reported with a subject-cluster bootstrap confidence interval and a surrogate p-value.

Three families of measures are examined:

- **Amplitude/energy** (M1–M12): cross-channel and same-channel models
- **Signal complexity** (M13–M15): permutation entropy, sample entropy, Lempel–Ziv
- **Shape/robust/Hjorth** (M16–M23): distributional statistics, slope, Hjorth parameters

Three findings emerge, each supported by the matched pseudotrial contrast and tested in an independent dataset:

1. **Cross-channel amplitude and energy couplings are near-zero and dataset-specific.** Their matched contrast does not indicate a stimulus-locked contribution, and a formal dataset-by-feature interaction shows they differ between datasets — consistent with dependence on the background structure of the signal rather than a stable population effect.
2. **Same-channel coupling is general within-trial temporal continuity.** It is large (R² ≈ 0.31) but statistically unchanged under pseudotrial substitution (Δβ ≈ 0), present at every electrode including the eye channels, and quantitatively equivalent across datasets once trial composition is matched — not a P300-specific process.
3. **Complexity measures carry at most a small, dataset-dependent population coupling, with no reliable individual-level structure.** Per-participant slopes have no demonstrable split-half reliability at the available trial counts, and their cross-measure agreement does not survive correction for shared estimation noise.

The autocorrelation ratio (AUR = |β_pseudo/β_real|) is retained only as a descriptive summary read against a calibrated background band, because a ratio's confidence interval becomes unbounded as its denominator approaches zero and is uninformative at the small effect sizes of the cross-channel and complexity measures.

---

## Datasets

### Primary: ERP CORE Visual P3 (Kappenman et al., 2021)

- Source: https://github.com/lucklab/ERP_CORE/tree/master/P3
- *N* = 27 retained (of 40); 1,084 epochs (213 target + 871 standard; both conditions retained, condition as covariate)
- BioSemi ActiveTwo, 1024 Hz, 30 scalp + 3 EOG channels, ±100 µV rejection
- Event coding: two-digit XY where X = block target letter (1–5), Y = shown letter; diagonal {11,22,33,44,55} = target, off-diagonal = standard

### Independent dataset: OpenNeuro ds006018 (Isbell et al., 2025)

- Source: https://openneuro.org/datasets/ds006018
- *N* = 90 retained (of 127); 3,130 target trials for real-trial models; *N* = 84 for pseudotrial fits at the primary ±150 µV configuration (73 at ±100 µV)
- Brain Products actiCHamp Plus, 500 Hz, 26 scalp channels, ±150 µV rejection
- Access via EEGDash: `pip install eegdash`
- Same active visual oddball (letters A–E) as ERP CORE

**Data are not included in this repository.** Download from the sources above and set the environment variables (see Configuration).

---

## Repository structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   │
│   │   -- configuration --
│   ├── config.py                            # Primary-dataset parameters (paths, filters,
│   │                                        #   windows, REJECT_THRESHOLD = 100e-6,
│   │                                        #   PSEUDOTRIAL_MIN_GAP_FROM_REAL = 1.0)
│   ├── config_ds006018.py                   # Independent-dataset parameters; also carries
│   │                                        #   PSEUDOTRIAL_CONFIGS (all four) and both
│   │                                        #   rejection thresholds
│   │
│   │   -- main numbered pipeline --
│   ├── 01_extract_features.py               # Preprocessing + feature extraction (ERP CORE)
│   ├── 02_run_lmms.py                       # Canonical LMMs (real trials)
│   ├── 03_per_electrode_lmm.py              # Per-electrode LMMs for topographic validation
│   ├── 04_pseudotrial_correction.py         # Config-1 pseudotrial run (M1, M4a, M9a, M12)
│   ├── 05_pseudotrial_diagnostic.py         # 4-config sensitivity sweep (Table 3)
│   ├── 06_pseudotrial_extended.py           # Config-4 pseudotrial run (amplitude models)
│   ├── 07_entropy_pseudotrial.py            # Config-4 pseudotrial, entropy/Hjorth
│   ├── 08_extended_endpoint_pseudotrial.py  # Config-4 pseudotrial, shape/robust/trend
│   ├── 09_interelectrode_validation.py      # Full-montage topographic analysis
│   ├── 10_crossval_ds006018.py              # Independent-dataset run
│   ├── 11_entropy_heterogeneity.py          # Per-subject complexity slope analysis
│   ├── 12_model_diagnostics.py              # Residual diagnostics + AIC comparison
│   ├── 13_make_figures.py                   # Main figures 1-7
│   ├── 14_additinal_figures.py              # Delta-beta forest plot and per-subject figure
│   │                                        #   (filename typo retained as deposited)
│   │
│   │   -- matched-contrast (Delta-beta) subsystem --
│   ├── run_phase013.py                      # Driver. Defines all four placement configs and
│   │                                        #   the +/-150 uV matched threshold; --config,
│   │                                        #   --clean-pseudo, --targets-only, --resample
│   ├── phase013_engine.py                   # Core estimator: per-subject slopes, Delta-beta,
│   │                                        #   BCa bootstrap, surrogate p, clean_pseudo_mask()
│   ├── phase013_erpcore.py                  # Primary-dataset driver
│   ├── phase013_ds006018.py                 # Independent-dataset driver
│   ├── phase013_cache.py                    # Epoch/feature caching
│   ├── phase013_smoketest.py                # Integration test for the engine
│   │
│   │   -- standalone validity checks --
│   ├── target_standard_validity.py          # Behavioural validity (accuracy, RT)
│   ├── distributional_comparability.py      # Real vs pseudotrial ITI / position / coverage
│   │
│   │   -- added in this revision --
│   ├── 15_calibration_simulation.py         # AUR null bands + power (see Known gaps)
│   ├── 16_slope_reliability.py              # Split-half / Spearman-Brown; cross-measure
│   ├── 17_overlap_audit.py                  # Pseudotrial overlap diagnostics
│   └── 18_figure_S2_calibration.py          # Supplementary Figure S2
│
└── results/
    │   -- canonical real-trial fits --
    ├── lmm_summary_canonical.csv                    (  17 rows) Canonical LMMs, N=27/1,084
    ├── per_electrode_canonical.csv                  (  30 rows) Per-electrode coefficients
    ├── model_diagnostics.csv                        (  18 rows) Log-lik, AIC, BIC by structure
    │
    │   -- single-draw pseudotrial runs --
    ├── pseudotrial_lmm_summary.csv                  (   4 rows) Script 04; M1, M4a, M9a, M12
    ├── pseudotrial_sensitivity.csv                  (  16 rows) Script 05; 4 configs x 4 models
    ├── entropy_pseudotrial_results.csv              (  10 rows) Script 07; real + pseudo
    ├── extended_endpoint_pseudotrial_results.csv    (  42 rows) Script 08; real + pseudo
    │
    │   -- matched-contrast (Delta-beta) outputs --
    ├── phase013_dbeta_config4.csv                   (   8 rows) PRIMARY. Both datasets
    ├── phase013_dbeta_config2.csv                   (   8 rows) +/-100 uV robustness
    ├── phase013_dbeta_config4_K1000_clean.csv       (   4 rows) Evoked-clean, erp_core only
    ├── phase013_dbeta_config4_K1000_rs500.csv       (   8 rows) 500 Hz decimation
    ├── phase013_interaction_config4.csv             (   4 rows) Dataset x feature interaction
    ├── phase013_interaction_config2.csv             (   4 rows)
    ├── phase013_interaction_config4_K1000_targets.csv (  4 rows) Condition-matched
    ├── phase013_interaction_config4_K1000_rs500.csv (   4 rows) 500 Hz
    ├── phase013_marginal_r2_config4_K1000.csv       (   8 rows) Marginal R2 robustness
    ├── phase013_marginal_r2_config2_K1000.csv       (   8 rows)
    ├── phase013_persubject_config4_K1000.csv        ( 492 rows) Per-subject real/pseudo slopes
    ├── phase013_persubject_config2_K1000.csv        ( 444 rows)
    │
    │   -- topography --
    ├── interelectrode_val1_cross_to_Pz.csv          (  66 rows) Cross-channel
    ├── interelectrode_val2_same_channel.csv         (  66 rows) Same-channel
    ├── interelectrode_val3_shape_to_Pz.csv          ( 132 rows) Complexity
    ├── interelectrode_all.csv                       ( 264 rows) All three combined
    │
    │   -- trial-level and independent dataset --
    ├── heterogeneity_primary_trials.csv             (1084 rows) Trial-level complexity + P300
    ├── heterogeneity_ds006018_checkpoint.csv        (3130 rows) Same, independent dataset
    ├── crossval_ds006018_results.csv                (  14 rows) Independent-dataset LMMs
    │
    │   -- validity checks --
    ├── target_standard_validity.csv                 (  40 rows) Per-subject accuracy and RT
    ├── distributional_comparability.csv             (  40 rows) ITI / position / coverage
    │
    │   -- added in this revision --
    ├── slope_reliability.csv                        (  10 rows) Split-half reliability
    ├── cross_measure_concordance.csv                (  20 rows) Same-trial vs disjoint-half
    │
    └── logs/                                        Duplicate copies of nine files above,
                                                     byte-identical to the top-level versions.
                                                     Retained for backward compatibility with
                                                     earlier scripts that read results/logs/.
```

**Two notes on the layout.** `config.py` and `config_ds006018.py` live inside `scripts/`, not at the repository root. There is no top-level `results_ds006018/` directory — `crossval_ds006018_results.csv` is in `results/`.

---

## Configuration

Both config files expose paths via environment variables so the pipeline runs without editing source:

```bash
# Primary dataset
export ERP_CORE_P3_DATA=/path/to/erp_core_P3
export DOF_SCRIPT_DIR=/path/to/this/repo/scripts
export DOF_RESULTS_DIR=/path/to/output/results

# Independent dataset
export DS006018_DATA=/path/to/ds006018
export DS006018_RESULTS=/path/to/output/results_ds006018
```

If the variables are not set, the scripts fall back to the hardcoded Windows paths in `scripts/config.py` / `scripts/config_ds006018.py`; update those for your system.

**Where each parameter actually lives.** Preprocessing settings, analysis windows and the real-trial rejection threshold (`REJECT_THRESHOLD = 100e-6`) are in the per-dataset config files. The four pseudotrial placement configurations and the relaxed ±150 µV threshold used for the matched contrast are defined in `run_phase013.py`, not in `config.py`. Note that `config.py` sets `PSEUDOTRIAL_MIN_GAP_FROM_REAL = 1.0`, which corresponds to configs 1 and 3 — the primary 0.5 s gap is selected through `run_phase013.py --config config4`. `config_ds006018.py` additionally carries an explicit `PSEUDOTRIAL_CONFIGS` list and both thresholds, so the two config files are not symmetric.

---

## Installation

```bash
pip install mne eegdash numpy scipy pandas statsmodels matplotlib antropy
```

Tested with Python 3.10+, MNE 1.7+, statsmodels 0.14+.

---

## Running order

The numbered scripts run in order; each reads outputs from earlier steps. The `phase013` subsystem runs after step 11 and produces the matched-contrast estimates the paper reports as primary.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `01_extract_features.py` | Raw ERP CORE BIDS | `trial_features_canonical.csv` |
| 2 | `02_run_lmms.py` | features | `lmm_summary_canonical.csv`, `model_diagnostics.csv` |
| 3 | `03_per_electrode_lmm.py` | features | `per_electrode_canonical.csv` |
| 4 | `04_pseudotrial_correction.py` | raw EEG | `pseudotrial_lmm_summary.csv` (Config-1) |
| 5 | `05_pseudotrial_diagnostic.py` | raw EEG | `pseudotrial_sensitivity.csv` (Table 3) |
| 6 | `06_pseudotrial_extended.py` | raw EEG | `extended_endpoint_pseudotrial_results.csv` |
| 7 | `07_entropy_pseudotrial.py` | raw EEG | `entropy_pseudotrial_results.csv` |
| 8 | `08_extended_endpoint_pseudotrial.py` | raw EEG | appends to `extended_endpoint_pseudotrial_results.csv` |
| 9 | `09_interelectrode_validation.py` | raw EEG | `interelectrode_val1/2/3`, `interelectrode_all.csv` |
| 10 | `10_crossval_ds006018.py` | ds006018 EEG | `crossval_ds006018_results.csv` |
| 11 | `11_entropy_heterogeneity.py` | features + ds006018 | `heterogeneity_primary_trials.csv`, `heterogeneity_ds006018_checkpoint.csv` |
| 12 | `run_phase013.py --config config4` | raw EEG (both datasets) | `phase013_dbeta_*`, `phase013_interaction_*`, `phase013_marginal_r2_*`, `phase013_persubject_*` |
| 13 | `target_standard_validity.py` | raw ERP CORE | `target_standard_validity.csv` |
| 14 | `distributional_comparability.py` | raw EEG | `distributional_comparability.csv` |
| 15 | `16_slope_reliability.py` | heterogeneity files | `slope_reliability.csv`, `cross_measure_concordance.csv` |
| 16 | `17_overlap_audit.py` | raw EEG | `overlap_audit_<dataset>_<config>.csv` |
| 17 | `12_model_diagnostics.py` | LMM fits | `model_diagnostics.csv` + Supplementary Figure S1 |
| 18 | `13_make_figures.py`, `14_additinal_figures.py`, `18_figure_S2_calibration.py` | all results | Figures 1–7, Supplementary Figure S2 |

`15_calibration_simulation.py` is self-contained and can be run at any point; it needs no EEG data.

Useful `run_phase013.py` flags: `--config {config1..config4|all}`, `--clean-pseudo` (drop pseudotrials whose measurement windows overlap a real evoked interval), `--targets-only` (condition-matched interaction), `--resample 500` (sampling-rate control).

---

## Key methodological parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Early window | 0–150 ms | At Fz (cross-channel) or Pz (same-channel) |
| P300 window | 300–600 ms | At Pz |
| Pseudotrial min gap | 0.5 s (Config-4 primary) | Also tested at 1.0 s (configs 1, 3) |
| Pseudotrial threshold | ±150 µV (Config-4 primary) | Also tested at ±100 µV (config 2) |
| Matched-contrast N | 28 primary / 84 independent | Config-4; 17 / 73 at ±100 µV |
| Surrogate placements | ≥ 1,000 per participant | `run_phase013.py -K` |
| Bootstrap | 4,000 replicates, subject-cluster BCa | `phase013_engine.py` |
| Model | Random-intercept LMM, REML | `statsmodels MixedLM` |
| Standardization | Robust z (median / 1.4826×MAD) | Within-participant |
| R² | Nakagawa & Schielzeth (2013) | Marginal (fixed effects only) |
| AUR | \|β_pseudo\| / \|β_real\| | Descriptive only; see the note below |
| Entropy parameters | PE: m=3, τ=1, normalized; SE: m=2, r=0.2×SD; LZ: median-threshold binarization | `antropy` |
| ΔAIC (RI vs RIS) | 32–54 in favour of random-intercept | From `model_diagnostics.csv` |

**Two AUR estimators exist in this repository and they are not interchangeable.** The *single-draw* estimator divides the canonical real-trial β by the config-4 pseudo β from `pseudotrial_sensitivity.csv`; it gives M1 = 0.30 and M4a = 1.02, and it is the estimator behind Table 9 and the corresponding Results text. The *matched-contrast* estimator uses `phase013_dbeta_config4.csv` (β_pseudo_mean / β_real over ≥ 1,000 placements); it gives M1 = 0.47, M4a = 1.37, M8 = 0.07, M9a = 0.99, and it is what Supplementary Figure S2 plots. For M1 the two fall on opposite sides of the calibrated band floor, which is why both are shown in that figure. This instability is the reason Δβ, not the ratio, is the reported estimand.

---

## Pseudotrial configurations (Table 3)

Table 3 reports the 4-config sensitivity sweep produced by **script 05** (`pseudotrial_sensitivity.csv`). The sweep uses a liberal subject-inclusion gate (a subject is retained if at least `MIN_TRIALS_REQUIRED` = 10 pseudotrials could be placed), which is what allows all 27 subjects to contribute under config 4.

Scripts 06–08 use a stricter gate: a subject is included only if a full complement of pseudotrials matching their real-trial count could be placed. This produces a slightly smaller pool (943 trials, 26 participants) but more closely matched per-subject sample sizes. The two config-4 runs are therefore not interchangeable.

| Config | Min gap | Threshold | Pseudotrials (% of real) | Participants | M9a *R*²_pseudo | M1 *R*²_pseudo |
|--------|---------|-----------|--------------------------|--------------|----------------|---------------|
| 1 | 1.0 s | ±100 µV | 157 (14%) | 11 | 0.298 | 0.004 |
| 2 | 0.5 s | ±100 µV | 455 (42%) | 16 | 0.279 | 0.002 |
| 3 | 1.0 s | ±150 µV | 426 (39%) | 20 | 0.303 | 0.012 |
| **4 (primary)** | **0.5 s** | **±150 µV** | **952 (88%)** | **27** | **0.229** | **0.001** |

**Deposited matched-contrast configs.** Only configs 2 and 4 were run through the `phase013` subsystem; there are no `phase013_*_config1` or `_config3` files. Configs 1 and 3 use the 1.0 s gap, which is infeasible for this paradigm's ~1.5 s stimulus-onset asynchrony, and appear only in the script-05 sweep.

---

## Result file reference

### Canonical real-trial fits

**`lmm_summary_canonical.csv`** — one row per model, real trials only, N = 27 / 1,084, ±100 µV.
Columns: `model`, `predictor`, `beta`, `SE`, `z`, `p`, `CI_low`, `CI_high`, `R2_marginal`, `R2_conditional`, `n_trials`, `n_subjects`, `fit_method`, `note`.
Contains M1–M12 including the a/b competitive and m mean-variant models (17 rows). **M13–M23 are not in this file**; they are in `entropy_pseudotrial_results.csv` and `extended_endpoint_pseudotrial_results.csv`, which come from a separate run — coefficients for models present in both differ in the fourth decimal place.

**`per_electrode_canonical.csv`** — per-electrode coefficients. Columns: `channel`, `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`.

**`model_diagnostics.csv`** — Columns: `model`, `family`, `structure`, `ll`, `aic`, `bic`. Source of the ΔAIC range.

### Matched-contrast (Δβ) outputs

**`phase013_dbeta_config4.csv`** — the primary inferential file. Columns: `dataset`, `config`, `model`, `n_subjects_processed`, `n_used`, `n_dropped`, `draws_min`, `draws_median`, `beta_real`, `beta_pseudo_mean`, `beta_pseudo_sd`, `dbeta`, `bca_lo`, `bca_hi`, `surrogate_p`. Four models × two datasets.

**`phase013_interaction_*.csv`** — Columns: `config`, `model`, `datasets`, `interaction_beta`, `interaction_p`, `n_trials`.

**`phase013_marginal_r2_*.csv`** — Columns: `dataset`, `config`, `model`, `marginal_r2`, `lmm_beta`, `n_subjects`, `n_trials`. A robustness check under robust-z preprocessing at two thresholds; the ±150 µV run retains N = 33 / 1,625 and the ±100 µV run N = 27 / 1,084, because they are different rejection cutoffs. **These are not the canonical R² values** — those are in `lmm_summary_canonical.csv`.

**`phase013_persubject_*.csv`** — Columns: `dataset`, `config`, `model`, `subject`, `real_slope`, `pseudo_mean`, `d`, `n_pseudo_draws`, `used`. Contains only the four headline amplitude/energy models (M1, M4a, M8, M9a) — no complexity measure.

### Single-draw pseudotrial runs

**`pseudotrial_sensitivity.csv`** — 4-config sweep, M1/M4a/M9a/M12. Columns: `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `model`, `config`. Source of Table 3 and of the single-draw AUR values.

**`entropy_pseudotrial_results.csv`** — real and pseudo rows for M13, M14, M15, M22, M23.
**`extended_endpoint_pseudotrial_results.csv`** — real and pseudo rows for the shape/robust/trend family and several amplitude models. Both use columns `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `model`, `kind`, `config`, `formula`.

**`pseudotrial_lmm_summary.csv`** — script 04, config 1. Columns include `beta_real`, `beta_pseudo`, `R2_real`, `R2_pseudo`, `beta_ratio_pseudo_over_real`. Retained for provenance; superseded by the matched contrast.

### Topography

**`interelectrode_val1/2/3_*.csv`** and **`interelectrode_all.csv`** — coupling at each of 33 electrodes (30 scalp + 3 EOG), real and pseudo. Columns: `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `electrode`, `kind`, `measure`.

### Trial-level and independent dataset

**`heterogeneity_primary_trials.csv`** / **`heterogeneity_ds006018_checkpoint.csv`** — trial-level complexity features and P300 amplitude for per-subject slope estimation. Columns: `subject`, `pe`, `se`, `lz`, `hjorth_mob`, `hjorth_cplx`, `p300` (the primary file additionally carries a condition column).

**`crossval_ds006018_results.csv`** — Columns: `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `model`, `kind`, `config`.

### Validity checks

**`target_standard_validity.csv`** — 40 subjects (`sub-001`–`sub-040`), i.e. the full ERP CORE sample **before** the EEG quality-control exclusions that produce N = 27. Columns: `n_target`, `n_standard`, `target_accuracy`, `target_median_rt`, `standard_accuracy`, `standard_median_rt`, `overall_accuracy`, `overall_median_rt`, `subject`.

**`distributional_comparability.csv`** — 40 subjects, same caveat. Columns: `n_real`, `fs`, `dur_s`, `real_iti_median`, `real_iti_q10`, `real_iti_q90`, `pseudo_gap_median`, `pseudo_gap_q10`, `pseudo_gap_q90`, `real_pos_mean`, `pseudo_pos_mean`, `real_pos_sd`, `pseudo_pos_sd`, `subject`.

### Added in this revision

**`slope_reliability.csv`** — split-half reliability behind Table 8's reliability column. Columns: `dataset`, `measure`, `n_subjects`, `slope_mean`, `slope_sd`, `odd_even_r`, `spearman_brown`, `reportable`, `sb_defined`. The Spearman–Brown correction is undefined for negative odd–even correlations (it returns values outside [−1, 1]); such cases are flagged `sb_defined = False` and reported as ≤ 0.

**`cross_measure_concordance.csv`** — Columns: `dataset`, `measure_a`, `measure_b`, `r_same_trials`, `r_disjoint_a_odd`, `r_disjoint_b_odd`, `r_disjoint_mean`. Both split directions are given because the assignment of which measure takes the odd half is arbitrary.

---

## Known gaps

- **`15_calibration_simulation.py` does not reproduce the published calibration bands.** It implements the specification in the paper's Section 2.12, but yields [0.60, 2.67] at R² ≈ 0.01 and [0.89, 1.12] at R² ≈ 0.31, against the published [0.37, 2.94] and [0.85, 1.17]. The qualitative structure is the same — wide at small R², narrow and centred on 1 at large R² — but the numbers differ, so this script is a reimplementation from the written specification and not the original. The specification does not pin down the per-participant trial count, the surrogate-averaging depth entering the ratio, or how sampling rate enters the window statistic, and each of those moves the band.
- **`17_overlap_audit.py` has not been run.** It needs the raw recordings, because it depends on true stimulus onsets and realised pseudotrial placements. The overlap percentages quoted in the paper (27.5%, 19.8%, ~80% retained, ~118 clean surrogates per placement) are therefore not yet reproduced from a deposited file. The evoked-clean *contrast* is deposited and does reproduce.
- **Configs 1 and 3 were not run through the matched-contrast subsystem**, so no `phase013_*_config1/3` files exist.
- **`13_make_figures.py` and `14_additinal_figures.py` both define Figure 3 and Figure 7** by different methods (`figure_3_pseudotrial` / `figure_3_dbeta_forest`, and `figure_7_entropy_heterogeneity` / `figure_7_persubject`). The paper uses the Δβ forest plot for Figure 3, and Figure 7 shows per-participant complexity slopes — note that `14_additinal_figures.py:figure_7_persubject` selects `model == 'M9a_mean_Pz'`, an amplitude model, and reads its second panel from hardcoded constants.
- **`results/logs/` duplicates nine files** that are already in `results/`, byte-identically.

---

## Reproducibility

All stochastic operations use fixed seeds (`RANDOM_SEED = 42`, `PSEUDOTRIAL_SEED = 12345`; the matched-contrast bootstrap and surrogate draws are separately seeded in `phase013_engine.py`). Filter settings, epoch windows, artifact thresholds, pseudotrial parameters and trial-count criteria are version-controlled and applied with the same values to both datasets, though they are distributed across `config.py`, `config_ds006018.py` and `run_phase013.py` rather than held in one file (see Configuration).

Each numerical result in the paper is traceable to a specific output CSV through the running order above.

---

## Citation

Biber, E. (2025). *Matched pseudotrial controls for early-window scalar associations with P300 amplitude: evidence from two visual oddball EEG datasets.* bioRxiv. https://doi.org/10.64898/2025.12.17.694588

---

## License

Code: MIT License. See `LICENSE` for details.  
Data: ERP CORE — CC BY 4.0 (Kappenman et al., 2021). ds006018 — CC0 (Isbell et al., 2025).