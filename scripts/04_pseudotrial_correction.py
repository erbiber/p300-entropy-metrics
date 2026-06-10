\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import mne

from config import (
    DATA_ROOT, RESULTS_DIR, FILES,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    BASELINE, REJECT_THRESHOLD,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT,
    ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, P300_WINDOW, BASELINE_CONTROL_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
    MIN_TRIALS_REQUIRED,
    PSEUDOTRIAL_MIN_GAP_FROM_REAL,
    PSEUDOTRIAL_SEED,
    get_channel_index, banner,
)

warnings.filterwarnings('ignore', category=Warning)
mne.set_log_level('WARNING')

                                                                            
                                                             
                                                                            
def feature_block(trace):
    mean_amp = float(np.mean(trace))
    sd_amp = float(np.std(trace - mean_amp, ddof=0))
    rms_amp = float(np.sqrt(np.mean(trace ** 2)))
    return mean_amp, sd_amp, rms_amp

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def robust_z_within_subject(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

                                                                            
                       
                                                                            
def generate_pseudotrial_samples(
        n_pseudo, sfreq, n_continuous_samples,
        real_event_samples, min_gap_seconds,
        tmin, tmax, rng):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    min_gap_samples = int(min_gap_seconds * sfreq)
    epoch_samples = int((tmax - tmin) * sfreq)
    pre_buffer = int(abs(tmin) * sfreq) + 1
    post_buffer = int(tmax * sfreq) + 1

                        
    valid_low = pre_buffer
    valid_high = n_continuous_samples - post_buffer
    if valid_high <= valid_low:
        return np.array([], dtype=int)

                                               
    forbidden = np.zeros(n_continuous_samples, dtype=bool)
    for ev in real_event_samples:
        lo = max(0, int(ev) - min_gap_samples)
        hi = min(n_continuous_samples, int(ev) + min_gap_samples)
        forbidden[lo:hi] = True

    placed_samples = []
                                                                               
    n_attempts = max(n_pseudo * 10, 1000)
    placed_set_for_gap = np.zeros(n_continuous_samples, dtype=bool)
    for _ in range(n_attempts):
        if len(placed_samples) >= n_pseudo:
            break
        cand = int(rng.integers(valid_low, valid_high))
        if forbidden[cand]:
            continue
        if placed_set_for_gap[cand]:
            continue
        placed_samples.append(cand)
                                                           
        lo = max(0, cand - epoch_samples)
        hi = min(n_continuous_samples, cand + epoch_samples)
        placed_set_for_gap[lo:hi] = True

    placed_samples.sort()
    return np.array(placed_samples, dtype=int)

                                                                            
                          
                                                                            
def process_subject_pseudotrials(sub_id, rng):
\
\
\
\
\
\
\
       
    set_path = os.path.join(
        DATA_ROOT, f"sub-{sub_id}", "ses-P3", "eeg",
        f"sub-{sub_id}_ses-P3_task-P3_eeg.set",
    )
    if not os.path.exists(set_path):
        return None, None, "file_not_found"

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design=FILTER_DESIGN, verbose=False)

                       
    if DO_ICA:
        try:
            raw_for_ica = raw.copy().filter(
                ICA_HIGHPASS_FOR_FIT, None,
                fir_design=FILTER_DESIGN, verbose=False)
            ica = mne.preprocessing.ICA(
                n_components=ICA_N_COMPONENTS,
                method=ICA_METHOD,
                random_state=ICA_RANDOM_STATE,
                max_iter='auto',
            )
            ica.fit(raw_for_ica, verbose=False)
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
            except Exception:
                ica.exclude = []
            raw = ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"  [sub-{sub_id}] ICA failed, continuing without: {e}")

    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    events, _ = mne.events_from_annotations(raw, verbose=False)
    real_sample_indices = events[:, 0]
    real_codes = events[:, 2]

                                                                
    epochs_real = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=REJECT_THRESHOLD),
        preload=True, verbose=False,
    )
    if len(epochs_real) < MIN_TRIALS_REQUIRED:
        return None, None, f"insufficient_real_trials_{len(epochs_real)}"
    real_codes_kept = epochs_real.events[:, 2]
    valid_real = np.isin(real_codes_kept, STANDARD_CODES + TARGET_CODES)
    epochs_real = epochs_real[valid_real]
    if len(epochs_real) < MIN_TRIALS_REQUIRED:
        return None, None, f"insufficient_valid_real_{len(epochs_real)}"
    n_real_retained = len(epochs_real)

                                                
                                                          
    n_pseudo_target = n_real_retained
    pseudo_samples = generate_pseudotrial_samples(
        n_pseudo=n_pseudo_target,
        sfreq=sfreq,
        n_continuous_samples=n_samples,
        real_event_samples=real_sample_indices,
        min_gap_seconds=PSEUDOTRIAL_MIN_GAP_FROM_REAL,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        rng=rng,
    )
    if len(pseudo_samples) < MIN_TRIALS_REQUIRED:
        return None, None, f"insufficient_pseudotrials_{len(pseudo_samples)}"

                                                    
                                                                          
    pseudo_events = np.column_stack([
        pseudo_samples,
        np.zeros(len(pseudo_samples), dtype=int),
        np.full(len(pseudo_samples), 99, dtype=int),
    ])

                                      
    epochs_pseudo = mne.Epochs(
        raw, pseudo_events, event_id={'pseudo': 99},
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=REJECT_THRESHOLD),
        preload=True, verbose=False,
    )
    if len(epochs_pseudo) < MIN_TRIALS_REQUIRED:
        return None, None, f"insufficient_pseudo_after_reject_{len(epochs_pseudo)}"

                                                   
    data_p = epochs_pseudo.get_data()
    times = epochs_pseudo.times
    ch_names = epochs_pseudo.ch_names

    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return None, None, "missing_channels_in_pseudo"

    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)
    m_basectrl = window_mask(times, BASELINE_CONTROL_WINDOW)

    rows_p = []
                                                                        
                                                                            
    for i in range(len(epochs_pseudo)):
        fz_tr = data_p[i, fz_idx, :]
        pz_tr = data_p[i, pz_idx, :]
        m_fz, sd_fz, rms_fz = feature_block(fz_tr[m_early])
        m_pz_e, sd_pz_e, rms_pz_e = feature_block(pz_tr[m_early])
        m_basef, _, rms_basef = feature_block(fz_tr[m_basectrl])
        m_basep, _, rms_basep = feature_block(pz_tr[m_basectrl])
        p300_amp = float(np.mean(pz_tr[m_p300]))
        rows_p.append({
                     : f'sub-{sub_id}',
                        : int(i),
                       : 'Pseudo',
                           : m_fz, 'sd_early_fz': sd_fz, 'rms_early_fz': rms_fz,
                           : m_pz_e, 'sd_early_pz': sd_pz_e, 'rms_early_pz': rms_pz_e,
                          : m_basef, 'rms_base_fz': rms_basef,
                          : m_basep, 'rms_base_pz': rms_basep,
                      : p300_amp,
        })

    df_pseudo = pd.DataFrame(rows_p)
    return None, df_pseudo, None                                  

                                                                            
                                           
                                                                            
def fit_RI(df, formula, predictor):
    for kwargs in [
        dict(method='lbfgs', reml=True),
        dict(method='nm', maxiter=2000, reml=True),
    ]:
        try:
            md = smf.mixedlm(formula, df, groups=df['subject'])
            fit = md.fit(**kwargs)
            ll = float(fit.llf)
            if not np.isfinite(ll):
                continue
            beta = float(fit.params[predictor])
            se = float(fit.bse[predictor])
            z = beta / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            X = np.asarray(fit.model.exog)
            betas = np.asarray(fit.fe_params)
            var_f = float(np.var(X @ betas))
            var_e = float(fit.scale)
            try:
                cov_re = np.atleast_2d(np.asarray(fit.cov_re))
                var_r = float(np.trace(cov_re))
            except Exception:
                var_r = 0.0
            denom = var_f + var_r + var_e
            r2m = var_f / denom if denom > 0 else np.nan
            return dict(beta=beta, SE=se, z=z, p=p, R2_marginal=r2m)
        except Exception:
            continue
    return dict(beta=np.nan, SE=np.nan, z=np.nan, p=np.nan, R2_marginal=np.nan)

                                                                            
      
                                                                            
def main():
    banner("04_pseudotrial_correction.py — autocorrelation control")
    print("Steinfath et al. (2025) procedure: random triggers in the")
    print("continuous data, matched to real trial count, with minimum gap")
    print(f"from real events ({PSEUDOTRIAL_MIN_GAP_FROM_REAL} s).\n")

                                                                            
    real_path = os.path.join(RESULTS_DIR, FILES['lmm_summary'])
    if not os.path.exists(real_path):
        print(f"ERROR: canonical LMM summary not found: {real_path}")
        print("Run 02_run_lmms.py first.")
        return

    real_summary = pd.read_csv(real_path)
                                                    
    target_models = ['M1_RMS_Fz_0_150', 'M4a_competitive_Fz_mean',
                                              , 'M12_Pz_mean_with_baseline_cov']
    real_compare = real_summary[real_summary['model'].isin(target_models)][
        ['model', 'beta', 'SE', 'z', 'p', 'R2_marginal']]
    print("Real-trial reference (from canonical LMM summary):")
    print(real_compare.to_string(index=False))
    print()

                                                    
    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: data root not found: {DATA_ROOT}")
        return

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]

    rng = np.random.default_rng(PSEUDOTRIAL_SEED)
    all_pseudo = []
    exclusions = {}
    print(f"Processing {len(sub_ids)} subjects for pseudotrials...\n")
    for sid in sub_ids:
        print(f"  sub-{sid} ...", end=' ')
        _, df_p, err = process_subject_pseudotrials(sid, rng)
        if df_p is None:
            exclusions[f'sub-{sid}'] = err
            print(f"excluded ({err})")
        else:
            all_pseudo.append(df_p)
            print(f"ok ({len(df_p)} pseudotrials)")

    if not all_pseudo:
        print("\nNo pseudotrials generated. Aborting.")
        return

    df_p_all = pd.concat(all_pseudo, ignore_index=True)
    print(f"\nTotal pseudotrials: {len(df_p_all)} "
          f"from {df_p_all['subject'].nunique()} subjects.\n")

                      
    feat = ['mean_early_fz', 'sd_early_fz', 'rms_early_fz',
                           , 'sd_early_pz', 'rms_early_pz',
                          , 'rms_base_fz',
                          , 'rms_base_pz',
                      ]
    for c in feat:
        if c in df_p_all.columns:
            df_p_all[c + '_z'] = df_p_all.groupby('subject')[c].transform(
                robust_z_within_subject)

                                                              
    pseudo_rows = []

    r = fit_RI(df_p_all,
                                     , 'rms_early_fz_z')
    r['model'] = 'M1_RMS_Fz_0_150'; r['predictor'] = 'rms_early_fz_z'
    r['n_trials'] = len(df_p_all)
    pseudo_rows.append(r)

    r = fit_RI(df_p_all,
                                                      ,
                         )
    r['model'] = 'M4a_competitive_Fz_mean'; r['predictor'] = 'mean_early_fz_z'
    r['n_trials'] = len(df_p_all)
    pseudo_rows.append(r)

    r = fit_RI(df_p_all,
                                                      ,
                         )
    r['model'] = 'M9a_competitive_Pz_mean'; r['predictor'] = 'mean_early_pz_z'
    r['n_trials'] = len(df_p_all)
    pseudo_rows.append(r)

    r = fit_RI(df_p_all,
                                                       ,
                         )
    r['model'] = 'M12_Pz_mean_with_baseline_cov'; r['predictor'] = 'mean_early_pz_z'
    r['n_trials'] = len(df_p_all)
    pseudo_rows.append(r)

    pseudo_summary = pd.DataFrame(pseudo_rows)[
        ['model', 'predictor', 'beta', 'SE', 'z', 'p', 'R2_marginal', 'n_trials']
    ]

                                                                
    real_compare = real_compare.rename(columns={
              : 'beta_real', 'SE': 'SE_real',
           : 'z_real', 'p': 'p_real', 'R2_marginal': 'R2_real',
    })
    pseudo_compare = pseudo_summary.rename(columns={
              : 'beta_pseudo', 'SE': 'SE_pseudo',
           : 'z_pseudo', 'p': 'p_pseudo', 'R2_marginal': 'R2_pseudo',
    })[['model', 'beta_pseudo', 'SE_pseudo', 'z_pseudo', 'p_pseudo',
                   , 'n_trials']]
    comparison = real_compare.merge(pseudo_compare, on='model', how='inner')
    comparison['R2_real_minus_pseudo'] = (
        comparison['R2_real'] - comparison['R2_pseudo'])
    comparison['beta_ratio_pseudo_over_real'] = np.where(
        np.abs(comparison['beta_real']) > 1e-12,
        comparison['beta_pseudo'] / comparison['beta_real'],
        np.nan)

    out_csv = os.path.join(RESULTS_DIR, FILES['pseudotrial_summary'])
    comparison.to_csv(out_csv, index=False)
    print(f"\nPseudotrial comparison saved -> {out_csv}\n")

    print("Comparison: real-trial vs pseudotrial LMM fits")
    print("-" * 72)
    cols_show = ['model', 'beta_real', 'beta_pseudo', 'R2_real', 'R2_pseudo',
                                       ]
    print(comparison[cols_show].to_string(index=False))
    print()

    banner("INTERPRETATION GUIDE")
    print("For each model, the comparison answers a specific question:")
    print()
    print("M9a is the headline finding (same-electrode Pz->Pz coupling).")
    print("  If R2_real >> R2_pseudo: the M9a coupling is time-locked to")
    print("    real events; the headline survives autocorrelation control.")
    print("  If R2_real ~ R2_pseudo: the M9a coupling is autocorrelation in")
    print("    the slow parietal positivity; the headline does NOT survive.")
    print()
    print("M1/M4a are cross-electrode (Fz->Pz). These should be LESS")
    print("affected by within-trial autocorrelation. A clean differential")
    print("pattern (M9a drops in pseudotrials but M1/M4a do not) is the")
    print("cleanest possible outcome.")
    print()
    print("M12 controls for Pz baseline. If M12 R2_real >> R2_pseudo, the")
    print("Pz early-window signal carries information ABOVE baseline drift.")

if __name__ == '__main__':
    main()
