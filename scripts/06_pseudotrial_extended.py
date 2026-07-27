import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import mne

from config import (
    DATA_ROOT, RESULTS_DIR,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    BASELINE,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT,
    ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, EARLY_WINDOW_200, EARLY_WINDOW_250,
    P300_WINDOW, BASELINE_CONTROL_WINDOW,
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
        return np.array([], dtype=int)
    forbidden = np.zeros(n_continuous_samples, dtype=bool)
    for ev in real_event_samples:
        lo = max(0, int(ev) - min_gap_samples)
        hi = min(n_continuous_samples, int(ev) + min_gap_samples)
        forbidden[lo:hi] = True
    placed = []
    blocked = np.zeros(n_continuous_samples, dtype=bool)
    for _ in range(max(n_pseudo * 20, 5000)):
        if len(placed) >= n_pseudo:
            break
        cand = int(rng.integers(valid_low, valid_high))
        if forbidden[cand] or blocked[cand]:
            continue
        placed.append(cand)
        lo = max(0, cand - epoch_samples)
        hi = min(n_continuous_samples, cand + epoch_samples)
        blocked[lo:hi] = True
    placed.sort()
    return np.array(placed, dtype=int)

def process_subject(sub_id, config_label, min_gap_s, reject_thresh,
                     rng, extracted_rows):
    set_path = os.path.join(
        DATA_ROOT, f"sub-{sub_id}", "ses-P3", "eeg",
        f"sub-{sub_id}_ses-P3_task-P3_eeg.set",
    )
    if not os.path.exists(set_path):
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
    n_samples = raw.n_times
    events, _ = mne.events_from_annotations(raw, verbose=False)
    real_sample_indices = events[:, 0]

    epochs_real = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=reject_thresh),
        preload=True, verbose=False,
    )
    if len(epochs_real) < MIN_TRIALS_REQUIRED:
        return
    valid_real_mask = np.isin(epochs_real.events[:, 2],
                              STANDARD_CODES + TARGET_CODES)
    n_real_retained = int(valid_real_mask.sum())
    if n_real_retained < MIN_TRIALS_REQUIRED:
        return

    pseudo_samples = generate_pseudotrial_samples(
        n_pseudo=max(n_real_retained, 1),
        sfreq=sfreq,
        n_continuous_samples=n_samples,
        real_event_samples=real_sample_indices,
        min_gap_seconds=min_gap_s,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        rng=rng,
    )
    if len(pseudo_samples) < MIN_TRIALS_REQUIRED:
        return

    pseudo_events = np.column_stack([
        pseudo_samples,
        np.zeros(len(pseudo_samples), dtype=int),
        np.full(len(pseudo_samples), 99, dtype=int),
    ])

    epochs_pseudo = mne.Epochs(
        raw, pseudo_events, event_id={'pseudo': 99},
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=reject_thresh),
        preload=True, verbose=False,
    )
    if len(epochs_pseudo) < MIN_TRIALS_REQUIRED:
        return

    data_p = epochs_pseudo.get_data()
    times = epochs_pseudo.times
    ch_names = epochs_pseudo.ch_names
    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return

    m_early_150 = window_mask(times, EARLY_WINDOW)
    m_early_200 = window_mask(times, EARLY_WINDOW_200)
    m_early_250 = window_mask(times, EARLY_WINDOW_250)
    m_p300 = window_mask(times, P300_WINDOW)
    m_basectrl = window_mask(times, BASELINE_CONTROL_WINDOW)

    for i in range(len(epochs_pseudo)):
        fz_tr = data_p[i, fz_idx, :]
        pz_tr = data_p[i, pz_idx, :]

        m_fz_150, sd_fz_150, rms_fz_150 = feature_block(fz_tr[m_early_150])
        m_fz_200, sd_fz_200, rms_fz_200 = feature_block(fz_tr[m_early_200])
        m_fz_250, sd_fz_250, rms_fz_250 = feature_block(fz_tr[m_early_250])

        m_pz_e, sd_pz_e, rms_pz_e = feature_block(pz_tr[m_early_150])

        m_basef, _, rms_basef = feature_block(fz_tr[m_basectrl])
        m_basep, _, rms_basep = feature_block(pz_tr[m_basectrl])

        p300_amp = float(np.mean(pz_tr[m_p300]))

        extracted_rows.append({
            'subject': f'sub-{sub_id}', 'config': config_label,
            'pseudo_idx': i,
            'mean_early_fz': m_fz_150, 'sd_early_fz': sd_fz_150, 'rms_early_fz': rms_fz_150,
            'mean_early_fz_200': m_fz_200, 'rms_early_fz_200': rms_fz_200,
            'mean_early_fz_250': m_fz_250, 'rms_early_fz_250': rms_fz_250,
            'mean_early_pz': m_pz_e, 'sd_early_pz': sd_pz_e, 'rms_early_pz': rms_pz_e,
            'mean_base_fz': m_basef, 'rms_base_fz': rms_basef,
            'mean_base_pz': m_basep, 'rms_base_pz': rms_basep,
            'p300_amp': p300_amp,
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
    banner("06_pseudotrial_extended.py — pseudotrial test for M5m/M6m/M8/M11")
    print("Extending the pseudotrial control to additional models.\n")
    print("Config 4 only (the full 4-config sweep was done in script 05):")
    print("  Config 4: min_gap=0.5 s, threshold=±150 µV  (primary / cleanest)\n")

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]

    configs = [
        ('config4', 0.5, 150e-6),
    ]
    extracted_rows = []

    for cfg_label, min_gap, thresh in configs:
        banner(f"Configuration: {cfg_label}  min_gap={min_gap} s  threshold=±{thresh*1e6:.0f} µV")
        rng = np.random.default_rng(PSEUDOTRIAL_SEED)
        for sid in sub_ids:
            process_subject(sid, cfg_label, min_gap, thresh, rng, extracted_rows)
        n_this_cfg = len([r for r in extracted_rows if r['config'] == cfg_label])
        n_subj = len(set(r['subject'] for r in extracted_rows
                         if r['config'] == cfg_label))
        print(f"  -> {n_this_cfg} pseudotrials from {n_subj} subjects\n")

    if not extracted_rows:
        print("No pseudotrials generated. Aborting.")
        return

    df_extr = pd.DataFrame(extracted_rows)

    feat = ['mean_early_fz', 'sd_early_fz', 'rms_early_fz',
            'mean_early_fz_200', 'rms_early_fz_200',
            'mean_early_fz_250', 'rms_early_fz_250',
            'mean_early_pz', 'sd_early_pz', 'rms_early_pz',
            'mean_base_fz', 'rms_base_fz',
            'mean_base_pz', 'rms_base_pz',
            'p300_amp']

    models_to_test = [

        ('M1_RMS_Fz_0_150',         'p300_amp_z ~ rms_early_fz_z',                      'rms_early_fz_z'),
        ('M4a_Fz_mean_competitive', 'p300_amp_z ~ mean_early_fz_z + sd_early_fz_z',     'mean_early_fz_z'),
        ('M9a_Pz_mean_competitive', 'p300_amp_z ~ mean_early_pz_z + sd_early_pz_z',     'mean_early_pz_z'),
        ('M12_Pz_mean_with_basecov','p300_amp_z ~ mean_early_pz_z + mean_base_pz_z',    'mean_early_pz_z'),

        ('M5m_Fz_mean_0_200',       'p300_amp_z ~ mean_early_fz_200_z',                 'mean_early_fz_200_z'),
        ('M6m_Fz_mean_0_250',       'p300_amp_z ~ mean_early_fz_250_z',                 'mean_early_fz_250_z'),
        ('M8_Pz_RMS_0_150',         'p300_amp_z ~ rms_early_pz_z',                      'rms_early_pz_z'),
        ('M11_Pz_RMS_with_basecov', 'p300_amp_z ~ rms_early_pz_z + rms_base_pz_z',      'rms_early_pz_z'),
    ]

    results = []
    for cfg, _, _ in configs:
        sub = df_extr[df_extr['config'] == cfg].copy()
        if len(sub) < MIN_TRIALS_REQUIRED or sub['subject'].nunique() < 3:
            continue

        for c in feat:
            sub[c + '_z'] = sub.groupby('subject')[c].transform(
                robust_z_within_subject)

        for model_name, formula, focal in models_to_test:

            fit_df = sub.dropna(subset=[focal, 'p300_amp_z']).copy()
            if len(fit_df) < MIN_TRIALS_REQUIRED or fit_df['subject'].nunique() < 3:
                continue
            r = fit_RI(fit_df, formula, focal)
            r['model'] = model_name
            r['config'] = cfg
            results.append(r)

    res_df = pd.DataFrame(results)

    res_path = os.path.join(RESULTS_DIR, 'logs', 'pseudotrial_extended_results.csv')
    res_df.to_csv(res_path, index=False)
    print(f"Extended results saved -> {res_path}\n")

    real_path = os.path.join(RESULTS_DIR, 'lmm_summary_canonical.csv')
    real = pd.read_csv(real_path)

    name_map = {
        'M1_RMS_Fz_0_150': 'M1_RMS_Fz_0_150',
        'M4a_Fz_mean_competitive': 'M4a_competitive_Fz_mean',
        'M9a_Pz_mean_competitive': 'M9a_competitive_Pz_mean',
        'M12_Pz_mean_with_basecov': 'M12_Pz_mean_with_baseline_cov',
        'M5m_Fz_mean_0_200': 'M5m_mean_Fz_0_200',
        'M6m_Fz_mean_0_250': 'M6m_mean_Fz_0_250',
        'M8_Pz_RMS_0_150': 'M8_RMS_Pz_0_150',
        'M11_Pz_RMS_with_basecov': 'M11_Pz_RMS_with_baseline_cov',
    }

    banner("MODEL-BY-MODEL: real vs pseudotrial across configs")
    print()
    print(f"  {'model':30s}  {'config':10s}  {'n':6s}  {'beta':10s}  {'R²':8s}")
    print("  " + "-" * 70)
    for model_name, _, _ in models_to_test:

        canonical = name_map.get(model_name, model_name)
        real_row = real[real['model'] == canonical]
        if not real_row.empty:
            r_beta = float(real_row['beta'].iloc[0])
            r_r2 = float(real_row['R2_marginal'].iloc[0])
            r_n = int(real_row['n_trials'].iloc[0])
            print(f"  {model_name:30s}  {'REAL':10s}  {r_n:6d}  {r_beta:+8.3f}    {r_r2:.3f}")
        else:
            print(f"  {model_name:30s}  REAL: not found")

        for cfg, _, _ in configs:
            row = res_df[(res_df['model'] == model_name) &
                          (res_df['config'] == cfg)]
            if not row.empty:
                r = row.iloc[0]
                print(f"  {'':30s}  {cfg:10s}  {r['n_trials']:6d}  "
                      f"{r['beta']:+8.3f}    {r['R2_marginal']:.3f}")
        print()

    banner("CONFIG 4 SUMMARY: REAL vs PSEUDO (cleanest test)")
    print()
    print(f"  {'model':30s}  {'β_real':9s}  {'β_pseudo':10s}  "
          f"{'R²_real':9s}  {'R²_pseudo':10s}  {'ΔR² (real-pseudo)':9s}")
    print("  " + "-" * 85)
    for model_name, _, _ in models_to_test:
        canonical = name_map.get(model_name, model_name)
        real_row = real[real['model'] == canonical]
        cfg4_row = res_df[(res_df['model'] == model_name) &
                          (res_df['config'] == 'config4')]
        if real_row.empty or cfg4_row.empty:
            continue
        r_beta = float(real_row['beta'].iloc[0])
        r_r2 = float(real_row['R2_marginal'].iloc[0])
        p_beta = float(cfg4_row['beta'].iloc[0])
        p_r2 = float(cfg4_row['R2_marginal'].iloc[0])
        delta_r2 = r_r2 - p_r2
        print(f"  {model_name:30s}  {r_beta:+7.3f}    {p_beta:+8.3f}      "
              f"{r_r2:.3f}      {p_r2:.3f}       {delta_r2:+.3f}")

if __name__ == '__main__':
    main()
