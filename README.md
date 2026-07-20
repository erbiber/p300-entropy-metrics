## Overview

This repository contains the complete analysis pipeline for a study examining whether conventional early-window endpoint-summary measures (mean amplitude, RMS, entropy, Hjorth parameters, etc.) capture genuine stimulus-locked information about P300 amplitude in the active visual oddball paradigm. Every coupling estimate is accompanied by a pseudotrial control estimate; the direction of change under pseudotrial substitution is the core diagnostic.

Three families of measures are examined:

- **Amplitude/energy** (M1–M12): cross-channel and same-channel models
- **Signal complexity** (M13–M15): permutation entropy, sample entropy, Lempel–Ziv
- **Shape/robust/Hjorth** (M16–M23): distributional statistics, slope, Hjorth parameters


Three findings emerge, each supported by the matched pseudotrial contrast and cross-validated in an independent dataset:

1. **Cross-channel amplitude and energy couplings are near-zero and dataset-specific.** Their matched contrast does not indicate a stimulus-locked contribution, and a formal dataset-by-feature interaction shows they differ between datasets — consistent with dependence on the background structure of the signal rather than a stable population effect.
2. **Same-channel coupling is general within-trial temporal continuity.** It is large (R² ≈ 0.31) but statistically unchanged under pseudotrial substitution (Δβ ≈ 0), present at every electrode including the eye channels, and quantitatively equivalent across datasets once trial composition is matched — not a P300-specific process.
3. **Complexity measures carry at most a small, dataset-dependent population coupling, with no reliable individual-level structure.** Per-participant slopes have no demonstrable split-half reliability at the available trial counts, and their cross-measure agreement does not survive correction for shared estimation noise; the sign-heterogeneous pattern is better explained by estimation noise around a near-zero population mean than by a stable person-level trait.

The autocorrelation ratio (AUR = |β_pseudo/β_real|) is retained only as a descriptive summary read against a calibrated background band, because a ratio's confidence interval becomes unbounded as its denominator approaches zero and is uninformative at the small effect sizes of the cross-channel and complexity measures.

---

## Datasets

### Primary: ERP CORE Visual P3 (Kappenman et al., 2021)

- Source: https://github.com/lucklab/ERP_CORE/tree/master/P3
- *N* = 27 retained (of 40); 1,084 epochs (213 target + 871 standard; both conditions retained, condition as covariate)
- BioSemi ActiveTwo, 1024 Hz, 30 scalp + 3 EOG channels, ±100 µV rejection
- Event coding: two-digit XY where X = block target letter (1–5), Y = shown letter; diagonal {11,22,33,44,55} = target, off-diagonal = standard

### Cross-validation: OpenNeuro ds006018 (Isbell et al., 2025)

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
├── config.py                         # Primary dataset parameters
├── config_ds006018.py                # Cross-validation dataset parameters
├── scripts/
│   ├── 01_extract_features.py        # Preprocessing + feature extraction (ERP CORE)
│   ├── 02_run_lmms.py                # Canonical LMMs (M1–M23, real trials)
│   ├── 03_per_electrode_lmm.py       # Per-electrode LMMs for topographic validation
│   ├── 04_pseudotrial_correction.py  # Config-1 pseudotrial run (M1, M4a, M9a, M12)
│   ├── 05_pseudotrial_diagnostic.py  # 4-config sensitivity sweep (Table 3)
│   ├── 06_pseudotrial_extended.py    # Config-4 primary pseudotrial run (all amplitude models)
│   ├── 07_entropy_pseudotrial.py     # Config-4 pseudotrial for entropy/complexity (M13–M15, M22–M23)
│   ├── 08_extended_endpoint_pseudotrial.py  # Config-4 pseudotrial for shape/robust/trend/Hjorth
│   ├── 09_interelectrode_validation.py      # Full-montage topographic analysis (Figures 4–6)
│   ├── 10_crossval_ds006018.py       # Cross-validation (ds006018; M1, M4a, M8, M9a, M13–M15)
│   ├── 11_entropy_heterogeneity.py   # Per-subject slope analysis + Figure 7
│   ├── 12_model_diagnostics.py       # Residual diagnostics + AIC comparison (Figure S1)
│   └── 13_make_figures.py            # All publication figures (Figures 1–7)
├── results/
│   ├── lmm_summary_canonical.csv     # Full canonical LMM output (M1–M23, real trials)
│   ├── model_diagnostics.csv         # AIC comparison and residual metrics
│   ├── pseudotrial_lmm_summary.csv   # Script-04 Config-1 pseudotrial estimates
│   ├── per_electrode_canonical.csv   # Per-electrode canonical LMM coefficients
│   └── logs/
│       ├── entropy_pseudotrial_results.csv           # Script-07 Config-4 entropy pseudotrials
│       ├── extended_endpoint_pseudotrial_results.csv # Script-08 Config-4 amplitude pseudotrials
│       ├── pseudotrial_sensitivity.csv               # 4-config sensitivity sweep (script 05)
│       ├── heterogeneity_primary_trials.csv          # Per-subject slopes, primary sample
│       ├── heterogeneity_ds006018_checkpoint.csv     # Per-subject slopes, ds006018
│       ├── interelectrode_val1_cross_to_Pz.csv       # Cross-channel topo (early → Pz)
│       ├── interelectrode_val2_same_channel.csv      # Same-channel topo (early → late, same elec)
│       ├── interelectrode_val3_shape_to_Pz.csv       # Complexity topo (PE + Hjorth mobility → Pz)
│       └── interelectrode_all.csv                    # All interelectrode results combined
└── results_ds006018/
    └── crossval_ds006018_results.csv # Cross-validation LMM output (real + pseudotrials)
```

---

## Configuration

Both config files expose paths via environment variables so the pipeline runs without editing source:

```bash
# Primary dataset
export ERP_CORE_P3_DATA=/path/to/erp_core_P3
export DOF_SCRIPT_DIR=/path/to/this/repo/scripts
export DOF_RESULTS_DIR=/path/to/output/results_update

# Cross-validation dataset
export DS006018_DATA=/path/to/ds006018
export DS006018_RESULTS=/path/to/output/results_ds006018
```

If the variables are not set, the scripts fall back to the hardcoded Windows paths in `config.py`/`config_ds006018.py`; update those for your system.

---

## Installation

```bash
pip install mne eegdash numpy scipy pandas statsmodels matplotlib antropy
```

Tested with Python 3.10+, MNE 1.7+, statsmodels 0.14+.

---

## Running order

Scripts must be run in the numbered order. Each script reads outputs from earlier scripts.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `01_extract_features.py` | Raw ERP CORE BIDS | `trial_features_canonical.csv` |
| 2 | `02_run_lmms.py` | features | `lmm_summary_canonical.csv`, `model_diagnostics.csv` |
| 3 | `03_per_electrode_lmm.py` | features | `per_electrode_canonical.csv` |
| 4 | `04_pseudotrial_correction.py` | raw EEG | `pseudotrial_lmm_summary.csv` (Config-1) |
| 5 | `05_pseudotrial_diagnostic.py` | raw EEG | `pseudotrial_sensitivity.csv` (Table 3) |
| 6 | `06_pseudotrial_extended.py` | raw EEG | `extended_endpoint_pseudotrial_results.csv` (Config-4) |
| 7 | `07_entropy_pseudotrial.py` | raw EEG | `entropy_pseudotrial_results.csv` (Config-4) |
| 8 | `08_extended_endpoint_pseudotrial.py` | raw EEG | appends to `extended_endpoint_pseudotrial_results.csv` |
| 9 | `09_interelectrode_validation.py` | raw EEG | `interelectrode_val1/2/3`, `interelectrode_all.csv` |
| 10 | `10_crossval_ds006018.py` | ds006018 EEG | `crossval_ds006018_results.csv` |
| 11 | `11_entropy_heterogeneity.py` | features + ds006018 | `heterogeneity_primary_trials.csv`, `heterogeneity_ds006018_checkpoint.csv` |
| 12 | `12_model_diagnostics.py` | LMM fits | `model_diagnostics.csv` + Figure S1 |
| 13 | `13_make_figures.py` | all results | Figures 1–7 (PNG + SVG) |

---

## Key methodological parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Early window | 0–150 ms | At Fz |
| P300 window | 300–600 ms | At Pz |
| Pseudotrial min gap | 0.5 s (Config-4 primary) | Also tested at 1.0 s |
| Pseudotrial threshold | ±150 µV (Config-4 primary) | Also tested at ±100 µV |
| Primary pseudotrial N | 943 (87% of 1,084) | 26 of 27 participants (primary Config-4 run, scripts 06–08) |
| Model | Random-intercept LMM, REML | `statsmodels MixedLM` |
| Standardization | Robust z (median / 1.4826×MAD) | Within-participant |
| R² | Nakagawa & Schielzeth (2013) | Marginal (fixed effects only) |
| AUR | \|β_pseudo\| / \|β_real\| | β-based throughout; R²-ratio used only for Table 8 region rows (stated in paper §2.8) |
| Entropy parameters | PE: m=3, τ=1, normalized; SE: m=2, r=0.2×SD; LZ: median-threshold binarization | `antropy` library |
| ΔAIC (RI vs RIS) | 32–54 in favor of random-intercept | Computed from `model_diagnostics.csv`; range across M1, M4a, M9a, M12, M13, M15 |

---

## Pseudotrial configurations (Table 3)

Table 3 reports the 4-config sensitivity sweep produced by **script 05** (`pseudotrial_sensitivity.csv`). The sweep uses a liberal subject-inclusion gate (subject retained if at least `MIN_TRIALS_REQUIRED` = 10 pseudotrials could be placed), which is what allows all 27 subjects to contribute even under the strictest configurations.

Scripts 06–08 (the primary Config-4 run) use a stricter gate: a subject is included only if a full complement of pseudotrials matching their real-trial count could be placed. This produces a slightly smaller pool (943 trials, 26 participants) but yields more closely matched real/pseudo sample sizes per subject. The two Config-4 runs are therefore not interchangeable; Table 3 uses the sweep throughout for consistency.

| Config | Min gap | Threshold | Pseudotrials (% of real) | Participants | M9a *R*²_pseudo | M1 *R*²_pseudo |
|--------|---------|-----------|--------------------------|--------------|----------------|---------------|
| 1 | 1.0 s | ±100 µV | 157 (14%) | 11 | 0.298 | 0.004 |
| 2 | 0.5 s | ±100 µV | 455 (42%) | 16 | 0.279 | 0.002 |
| 3 | 1.0 s | ±150 µV | 426 (39%) | 20 | 0.303 | 0.012 |
| **4 (primary)** | **0.5 s** | **±150 µV** | **952 (88%)** | **27** | **0.229** | **0.001** |

Config-4 is used as the primary comparison throughout. Scripts 06–10 all use Config-4 parameters. Config-1 (script 04) and the sensitivity sweep (script 05) confirm conclusions are unchanged across configurations.

> **Note on deposited files:** `pseudotrial_lmm_summary.csv` in `results/` contains Config-1 estimates for M1, M4a, M9a, M12 (script 04). All tables and figures in the paper use Config-4 values. For most models these come from `extended_endpoint_pseudotrial_results.csv` and `entropy_pseudotrial_results.csv` (scripts 06–08). The M1, M4a, M9a, and M12 Config-4 pseudotrial estimates reported in Tables 4–5 of the paper (β_pseudo = −0.106, +0.221, +0.582, +0.589 respectively; *n* = 943, 26 participants) were produced by script 06's primary run but are not separately deposited in the CSV above, which covers the remaining amplitude and shape models. The Config-1 file (`pseudotrial_lmm_summary.csv`) is retained for provenance of the original sensitivity check.

---

## Result files

### `results/lmm_summary_canonical.csv`
One row per model (M1–M23). Columns: `model`, `predictor`, `beta`, `SE`, `z`, `p`, `CI_low`, `CI_high`, `R2_marginal`, `R2_conditional`, `n_trials`, `n_subjects`, `fit_method`, `note`.

### `results/logs/extended_endpoint_pseudotrial_results.csv`
Config-4 pseudotrial estimates for amplitude/energy/shape models (M2, M3, M4b, M5–M11, M9b, and the shape/Hjorth models M16–M23 via script 08). Columns: `model`, `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `kind` (real/pseudo), `config`. Note: M1, M4a, M9a, and M12 are not in this file; see the note in the Pseudotrial configurations section above.

### `results/logs/entropy_pseudotrial_results.csv`
Config-4 pseudotrial estimates for M13 (permutation entropy), M14 (sample entropy), M15 (Lempel–Ziv), M22 (Hjorth mobility), M23 (Hjorth complexity). Same columns as above plus `measure`.

### `results/logs/pseudotrial_sensitivity.csv`
4-config sensitivity sweep (script 05) for M1, M4a, M9a, M12. Source for Table 3 in the paper. Columns: `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `model`, `config`.

### `results/logs/interelectrode_val1/2/3_*.csv`
Full-montage coupling estimates at each of the 33 electrodes (30 scalp + 3 EOG). Columns: `electrode`, `beta`, `R2_marginal`, `p`, `kind`, `measure`.

### `results/logs/heterogeneity_primary_trials.csv` / `heterogeneity_ds006018_checkpoint.csv`
Trial-level data with per-trial complexity features and P300 amplitude, used for per-subject slope estimation (Figure 7, Table 9). Columns include `subject`, `p300`, `pe`, `se`, `lz`, `hjorth_mob`, `hjorth_cplx`.

### `results_ds006018/crossval_ds006018_results.csv`
Cross-validation output. Columns: `model`, `beta`, `SE`, `z`, `p`, `R2_marginal`, `n_trials`, `n_subjects`, `kind` (real/pseudo), `config`.

---

## Reproducibility

All stochastic operations use fixed seeds (`RANDOM_SEED = 42`, `PSEUDOTRIAL_SEED = 12345`). Filter settings, epoch windows, artifact thresholds, pseudotrial parameters, and trial-count criteria are fixed in `config.py` / `config_ds006018.py` and applied identically across datasets.

Each numerical result in the paper is traceable to a specific output CSV through the running order above. The preprint at https://doi.org/10.64898/2025.12.17.694588 contains a provenance section mapping every reported statistic to its source file and script.

---

## Citation

Biber, E. (2025). *Matched pseudotrial controls for early-window scalar associations with P300 amplitude: evidence from two visual oddball EEG datasets.* bioRxiv. https://doi.org/10.64898/2025.12.17.694588

---

## License

Code: MIT License. See `LICENSE` for details.  
Data: ERP CORE — CC BY 4.0 (Kappenman et al., 2021). ds006018 — CC0 (Isbell et al., 2025).
