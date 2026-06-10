# Single-trial endpoint-summary measures and the P300

Analysis code and outputs for the manuscript *"Endpoint-summary measures do not capture consistent population-level P300 coupling in the visual oddball: a pseudotrial-controlled, cross-validated study."*

This repository contains the complete analysis pipeline for the **static endpoint-summary** investigation. Each reported number in the manuscript traces to an output file in `results/`; the mapping is documented in `docs/number_provenance.md`.

## What the study does

Single-trial coupling between an early-window (0–150 ms) summary measure and the later P300 (300–600 ms) is estimated with linear mixed-effects models, and every model is re-estimated on **pseudotrials** (windows placed at random latencies in the same continuous recording) to separate stimulus-locked coupling from background EEG autocorrelation. Twenty-three models (M1–M23) span amplitude, energy, distributional-shape, complexity, and Hjorth measures. Findings are cross-validated on an independent dataset (OpenNeuro ds006018) and examined for per-subject heterogeneity.

## Repository layout

```
scripts/       Analysis pipeline (run in numerical order) + config files
                 config.py            shared settings (paths, windows, electrodes)
                 config_ds006018.py   cross-validation dataset settings
results/       Output CSVs (the canonical numbers)
results/logs/  Per-analysis detailed outputs

```

The config files live alongside the scripts so that `from config import ...` resolves with no `PYTHONPATH` setup — run each script from within `scripts/`.

## Pipeline order

Run in numerical order from within `scripts/` after setting the data path (see Setup). Scripts share `config.py`.

| Script | Purpose | Key output |
|---|---|---|
| `01_extract_features.py` | Preprocess ERP CORE P3, extract single-trial features | `trial_features_canonical.csv` |
| `02_run_lmms.py` | Fit M1–M12 mixed-effects coupling models | `lmm_summary_canonical.csv` |
| `03_per_electrode_lmm.py` | Per-electrode coupling map (cross-channel β at every site) | `per_electrode_canonical.csv` |
| `04_pseudotrial_correction.py` | Primary pseudotrial control — four configurations | `pseudotrial_lmm_summary.csv` |
| `05_pseudotrial_diagnostic.py` | Pseudotrial placement diagnostics, yield per configuration | `logs/pseudotrial_diagnostic.csv` |
| `06_pseudotrial_extended.py` | Extend pseudotrial test to additional amplitude models (Config 4 only) | `logs/pseudotrial_extended_results.csv` |
| `07_entropy_pseudotrial.py` | Pseudotrial control for entropy/complexity family (M13–M15) plus Hjorth mobility/complexity (Config 4 only) | `logs/entropy_pseudotrial_results.csv` |
| `08_extended_endpoint_pseudotrial.py` | Pseudotrial test for shape/Hjorth/robust family (M16–M23, Config 4 only) | `logs/extended_endpoint_pseudotrial_results.csv` |
| `09_interelectrode_validation.py` | Full-montage topographic validation — three mechanisms | `logs/interelectrode_*.csv` |
| `10_crossval_ds006018.py` | Cross-validation on ds006018 (population-level models) | `crossval_ds006018_results.csv` |
| `11_entropy_heterogeneity.py` | Per-subject slope analysis (entropy + Hjorth mobility/complexity) — both datasets (N=27, N=90) | `logs/heterogeneity_*.csv` |
| `12_model_diagnostics.py` | RI vs RIS AIC comparison + residual diagnostics, all families | `model_diagnostics.csv`, `figS1_diagnostics.png` |
| `13_make_figures.py` | All manuscript figures (Figures 1–8) | `figures/fig1–fig8.png/svg` |
| `inspect_ds006018.py` | Inspect ds006018 channels/codes/sampling rate (utility) | console |

## Two sample sizes for the cross-validation dataset

ds006018 appears with two participant counts, both correct, because two scripts process it for different purposes:

- **N = 56** (51 with pseudotrials) — the population-level coupling models in `10_crossval_ds006018.py`, which require matched pseudotrials.
- **N = 90** — the per-subject heterogeneity analysis in `11_entropy_heterogeneity.py`, which needs only real-trial features.

The primary ERP CORE sample is **N = 27** (13 of 40 excluded for insufficient clean trials; see `results/logs/subject_exclusions.csv`).

## Setup

```bash
pip install -r requirements.txt
```

The three paths are set to their default values in `scripts/config.py`:

| Variable | Default (Erkan's machine) |
|---|---|
| `DATA_ROOT` | `C:\Users\erkan\Documents\dof_validation\data\erp_core_P3` |
| `SCRIPT_DIR` | `C:\Users\erkan\Documents\dof_validation\scripts\Brain\Final\Update` |
| `RESULTS_DIR` | `C:\Users\erkan\Documents\dof_validation\results_update` |

To run on a different machine, override with environment variables (no editing needed):

```bash
# Windows (PowerShell)
$env:ERP_CORE_P3_DATA  = "D:\your\path\to\erp_core_P3"
$env:DOF_RESULTS_DIR   = "D:\your\path\to\results"

# Mac / Linux
export ERP_CORE_P3_DATA="/your/path/to/erp_core_P3"
export DOF_RESULTS_DIR="/your/path/to/results"
```

Run scripts from within the `scripts/` directory so `from config import` resolves:

```bash
cd scripts
python 01_extract_features.py
python 02_run_lmms.py
# ... continue in order through 13_entropy_heterogeneity.py
```

ERP CORE data: download from the ERP CORE OSF repository (Kappenman et al., 2021) and place the `.set` / `.fdt` files in `DATA_ROOT`. The ds006018 cross-validation dataset is downloaded automatically by script 12 via EEGDash.

## Data availability

This repository ships the analysis **outputs** (`results/`) so the manuscript numbers can be reproduced and checked without re-running preprocessing. The raw EEG is not redistributed here: ERP CORE and ds006018 are obtained from their original open repositories as described above.

## Citation

If you use this code, please cite the manuscript (details to follow on publication) and the two datasets:

- Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). ERP CORE: An open resource for human event-related potential research. *NeuroImage, 225*, 117465.
- Isbell, E., Peters, A. N., Richardson, D. M., & Rodas De León, N. E. (2025). Cognitive electrophysiology in socioeconomic context in adulthood. *Scientific Data, 12*, 841.
