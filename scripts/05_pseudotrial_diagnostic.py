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
    BASELINE,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT,
    ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, P300_WINDOW, BASELINE_CONTROL_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
    MIN_TRIALS_REQUIRED,
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

                                                                            
def generate_pseudotrial_samples(n_pseudo, sfreq, n_continuous_samples,
                                  real_event_samples, min_gap_seconds,
                                  tmin, tmax, rng):
                                                                                
    min_gap_samples = int(min_gap_seconds * sfreq)
    epoch_samples = int((tmax - tmin) * sfreq)
    pre_buffer = int(abs(tmin) * sfreq) + 1
    post_buffer = int(tmax * sfreq) + 1

    valid_low = pre_buffer
    valid_high = n_continuous_samples - post_buffer
    if valid_high <= valid_low:
        return np.array([], dtype=int), 0, 0

    forbidden = np.zeros(n_continuous_samples, dtype=bool)
    for ev in real_event_samples:
        lo = max(0, int(ev) - min_gap_samples)
        hi = min(n_continuous_samples, int(ev) + min_gap_samples)
        forbidden[lo:hi] = True

    placed_samples = []
    placed_set_for_gap = np.zeros(n_continuous_samples, dtype=bool)
    n_attempts = max(n_pseudo * 20, 5000)
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
    n_continuous_free = int((~forbidden).sum())
    return (np.array(placed_samples, dtype=int),
            n_continuous_free, valid_high - valid_low)

                                                                            
def process_subject(sub_id, config_label, min_gap_s, reject_thresh, rng,
                    log_rows, extracted_rows):
\
                                              
    set_path = os.path.join(
        DATA_ROOT, f"sub-{sub_id}", "ses-P3", "eeg",
        f"sub-{sub_id}_ses-P3_task-P3_eeg.set",
    )
    if not os.path.exists(set_path):
        log_rows.append({'subject': f'sub-{sub_id}', 'config': config_label,
                                 : 'file_not_found'})
        return

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
        except Exception:
            pass

    sfreq = raw.info['sfreq']
    n_samples_total = raw.n_times
    events, _ = mne.events_from_annotations(raw, verbose=False)
    real_sample_indices = events[:, 0]

                                                     
    epochs_real = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=reject_thresh),
        preload=True, verbose=False,
    )
    real_codes_kept = epochs_real.events[:, 2] if len(epochs_real) > 0 else []
    valid_real_mask = np.isin(real_codes_kept,
                              STANDARD_CODES + TARGET_CODES)
    n_real_retained = int(valid_real_mask.sum())

                                        
    n_pseudo_target = max(n_real_retained, 1)
    pseudo_samples, n_free, n_valid_range = generate_pseudotrial_samples(
        n_pseudo=n_pseudo_target,
        sfreq=sfreq,
        n_continuous_samples=n_samples_total,
        real_event_samples=real_sample_indices,
        min_gap_seconds=min_gap_s,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        rng=rng,
    )
    n_pseudo_placed = len(pseudo_samples)

    if n_pseudo_placed < MIN_TRIALS_REQUIRED:
        log_rows.append({
                     : f'sub-{sub_id}', 'config': config_label,
                    : f'insufficient_placement_{n_pseudo_placed}',
                                  : n_samples_total,
                                      : n_free,
                                   : n_valid_range,
                             : n_pseudo_placed,
                               : 0,
                             : n_real_retained,
        })
        return

    pseudo_events = np.column_stack([
        pseudo_samples,
        np.zeros(len(pseudo_samples), dtype=int),
        np.full(len(pseudo_samples), 99, dtype=int),
    ])

                                                                           
                                                                     
                                    
    epochs_pseudo_all = mne.Epochs(
        raw, pseudo_events, event_id={'pseudo': 99},
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=None,
        preload=True, verbose=False,
    )
    epochs_pseudo = mne.Epochs(
        raw, pseudo_events, event_id={'pseudo': 99},
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=reject_thresh),
        preload=True, verbose=False,
    )
    n_pseudo_retained = len(epochs_pseudo)

                                                                       
    if len(epochs_pseudo_all) > 0:
        data_all = epochs_pseudo_all.get_data()
                                                                      
        ptp_per_epoch = (data_all.max(axis=2) - data_all.min(axis=2)).max(axis=1)
        ptp_median = float(np.median(ptp_per_epoch))
        ptp_95 = float(np.percentile(ptp_per_epoch, 95))
        ptp_max = float(ptp_per_epoch.max())
    else:
        ptp_median = ptp_95 = ptp_max = np.nan

    log_rows.append({
                 : f'sub-{sub_id}', 'config': config_label,
                : 'ok' if n_pseudo_retained >= MIN_TRIALS_REQUIRED else f'too_few_retained_{n_pseudo_retained}',
                              : n_samples_total,
                                  : n_free,
                               : n_valid_range,
                         : n_pseudo_placed,
                           : n_pseudo_retained,
                         : n_real_retained,
                       : ptp_median * 1e6,
                     : ptp_95 * 1e6,
                    : ptp_max * 1e6,
                               : n_pseudo_retained / max(n_pseudo_placed, 1),
                             : n_real_retained / max(len(real_sample_indices), 1),
    })

                                                                         
    if n_pseudo_retained < MIN_TRIALS_REQUIRED:
        return
    data_p = epochs_pseudo.get_data()
    times = epochs_pseudo.times
    ch_names = epochs_pseudo.ch_names
    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return

    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)
    m_basectrl = window_mask(times, BASELINE_CONTROL_WINDOW)

    for i in range(n_pseudo_retained):
        fz_tr = data_p[i, fz_idx, :]
        pz_tr = data_p[i, pz_idx, :]
        m_fz, sd_fz, rms_fz = feature_block(fz_tr[m_early])
        m_pz_e, sd_pz_e, rms_pz_e = feature_block(pz_tr[m_early])
        m_basef, _, rms_basef = feature_block(fz_tr[m_basectrl])
        m_basep, _, rms_basep = feature_block(pz_tr[m_basectrl])
        p300_amp = float(np.mean(pz_tr[m_p300]))
        extracted_rows.append({
                     : f'sub-{sub_id}', 'config': config_label,
                        : i,
                           : m_fz, 'sd_early_fz': sd_fz, 'rms_early_fz': rms_fz,
                           : m_pz_e, 'sd_early_pz': sd_pz_e, 'rms_early_pz': rms_pz_e,
                          : m_basef, 'rms_base_fz': rms_basef,
                          : m_basep, 'rms_base_pz': rms_basep,
                      : p300_amp,
        })

                                                                            
def fit_RI(df, formula, predictor):
    for kwargs in [dict(method='lbfgs', reml=True),
                   dict(method='nm', maxiter=2000, reml=True)]:
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
            return dict(beta=beta, SE=se, z=z, p=p, R2_marginal=r2m,
                        n_trials=len(df), n_subjects=df['subject'].nunique())
        except Exception:
            continue
    return dict(beta=np.nan, SE=np.nan, z=np.nan, p=np.nan,
                R2_marginal=np.nan, n_trials=len(df),
                n_subjects=df['subject'].nunique())

                                                                            
def main():
    banner("05_pseudotrial_diagnostic.py — diagnostic + sensitivity")

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]
    print(f"Processing {len(sub_ids)} subjects under 3 configurations:")
    print(f"  Config 1: min_gap=1.0 s, threshold=±100 µV  (original)")
    print(f"  Config 2: min_gap=0.5 s, threshold=±100 µV  (relaxed placement)")
    print(f"  Config 3: min_gap=1.0 s, threshold=±150 µV  (relaxed rejection)")
    print(f"  Config 4: min_gap=0.5 s, threshold=±150 µV  (both relaxed)")
    print()

    log_rows = []
    extracted_rows = []
    configs = [
        ('config1', 1.0, 100e-6),
        ('config2', 0.5, 100e-6),
        ('config3', 1.0, 150e-6),
        ('config4', 0.5, 150e-6),
    ]

    for cfg_label, min_gap, thresh in configs:
        banner(f"Configuration: {cfg_label}  min_gap={min_gap} s  threshold=±{thresh*1e6:.0f} µV")
        rng = np.random.default_rng(PSEUDOTRIAL_SEED)
        for sid in sub_ids:
            print(f"  sub-{sid} ...", end=' ')
            process_subject(sid, cfg_label, min_gap, thresh, rng,
                            log_rows, extracted_rows)
                                                            
            this_log = [r for r in log_rows
                        if r['subject'] == f'sub-{sid}' and r['config'] == cfg_label]
            if this_log:
                r = this_log[-1]
                if r.get('n_pseudo_retained', 0) >= MIN_TRIALS_REQUIRED:
                    print(f"ok ({r['n_pseudo_retained']} pseudo / {r['n_real_retained']} real, "
                          f"retention {r.get('pseudo_retention_rate', 0):.2f} vs real {r.get('real_retention_rate', 0):.2f})")
                else:
                    print(f"{r['status']}")

                         
    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(RESULTS_DIR, 'logs', 'pseudotrial_diagnostic.csv')
    log_df.to_csv(log_path, index=False)
    print(f"\nDiagnostic log -> {log_path}")

                                  
    banner("RETENTION SUMMARY BY CONFIGURATION")
    for cfg, _, _ in configs:
        sub = log_df[log_df['config'] == cfg]
        ok = sub[sub.get('n_pseudo_retained', pd.Series([0]*len(sub))).fillna(0) >= MIN_TRIALS_REQUIRED] if 'n_pseudo_retained' in sub.columns else pd.DataFrame()
        n_subjects_ok = len(ok)
        n_pseudo_total = int(ok['n_pseudo_retained'].sum()) if len(ok) else 0
        if len(ok):
            mean_retention = float(ok['pseudo_retention_rate'].mean())
            mean_real_retention = float(ok['real_retention_rate'].mean())
            mean_ptp = float(ok['ptp_median_uV'].mean()) if 'ptp_median_uV' in ok.columns else np.nan
        else:
            mean_retention = mean_real_retention = mean_ptp = np.nan
        print(f"  {cfg}: {n_subjects_ok}/{len(sub)} subjects, "
              f"{n_pseudo_total} pseudotrials total, "
              f"retention {mean_retention:.2f} (real: {mean_real_retention:.2f}), "
              f"median ptp = {mean_ptp:.0f} µV")
    print()

                                                                     
    if not extracted_rows:
        print("No pseudotrial features extracted under any configuration.")
        return

    df_extr = pd.DataFrame(extracted_rows)
    banner("M9a (and other headlines) on each configuration's pseudotrials")
    results = []
    for cfg, _, _ in configs:
        sub = df_extr[df_extr['config'] == cfg].copy()
        if len(sub) < MIN_TRIALS_REQUIRED or sub['subject'].nunique() < 3:
            print(f"\n  {cfg}: insufficient data ({len(sub)} trials, "
                  f"{sub['subject'].nunique()} subjects)")
            continue

        feat = ['mean_early_fz', 'sd_early_fz', 'rms_early_fz',
                               , 'sd_early_pz', 'rms_early_pz',
                              , 'rms_base_fz',
                              , 'rms_base_pz',
                          ]
        for c in feat:
            sub[c + '_z'] = sub.groupby('subject')[c].transform(
                robust_z_within_subject)

        print(f"\n  --- {cfg}: {len(sub)} pseudotrials from "
              f"{sub['subject'].nunique()} subjects ---")
        for model_name, formula, focal in [
            ('M1_RMS_Fz', 'p300_amp_z ~ rms_early_fz_z', 'rms_early_fz_z'),
            ('M4a_Fz_mean', 'p300_amp_z ~ mean_early_fz_z + sd_early_fz_z', 'mean_early_fz_z'),
            ('M9a_Pz_mean', 'p300_amp_z ~ mean_early_pz_z + sd_early_pz_z', 'mean_early_pz_z'),
            ('M12_Pz_mean_with_baseline', 'p300_amp_z ~ mean_early_pz_z + mean_base_pz_z', 'mean_early_pz_z'),
        ]:
            r = fit_RI(sub, formula, focal)
            r['model'] = model_name; r['config'] = cfg
            results.append(r)
            print(f"    {model_name:30s}  beta={r['beta']:+.3f}  "
                  f"R2={r['R2_marginal']:.3f}  n={r['n_trials']}")

    if results:
        res_df = pd.DataFrame(results)
        res_path = os.path.join(RESULTS_DIR, 'logs', 'pseudotrial_sensitivity.csv')
        res_df.to_csv(res_path, index=False)
        print(f"\nSensitivity results -> {res_path}")

                                    
    banner("M9a R² STABILITY ACROSS CONFIGURATIONS")
    print("The real-trial M9a R² is 0.312. Original (config1) pseudotrial R² was 0.348.")
    print()
    m9a_rows = [r for r in results if r['model'] == 'M9a_Pz_mean']
    print(f"  {'config':10s}  {'n_trials':10s}  {'n_subjects':12s}  {'beta':10s}  {'R2_pseudo':10s}")
    print("  " + "-" * 60)
    for r in m9a_rows:
        print(f"  {r['config']:10s}  {r['n_trials']:10d}  {r['n_subjects']:12d}  "
              f"{r['beta']:+8.3f}    {r['R2_marginal']:.3f}")
    print()
    print("Interpretation guide:")
    print("  If M9a R² is stable across configs and remains close to real R² (0.312),")
    print("  the original conclusion is robust: M9a is autocorrelation.")
    print("  If M9a R² varies substantially or drops sharply in better-sampled configs,")
    print("  the original conclusion was driven by sampling bias and the question")
    print("  is still open.")

if __name__ == '__main__':
    main()
