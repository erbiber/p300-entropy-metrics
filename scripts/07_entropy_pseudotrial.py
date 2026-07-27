import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import mne
import antropy as ant

from config import (
    DATA_ROOT, RESULTS_DIR,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    BASELINE,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT,
    ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, P300_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
    MIN_TRIALS_REQUIRED,
    PSEUDOTRIAL_SEED,
    get_channel_index, banner,
)

warnings.filterwarnings('ignore', category=Warning)
mne.set_log_level('WARNING')

def hjorth_block(trace):
    try:
        n = len(trace)
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or n <= 2:
            return np.nan, np.nan
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        mob_x = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan
        ddx = np.diff(dx)
        var_ddx = float(np.var(ddx, ddof=0))
        mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan
        mob = float(mob_x) if np.isfinite(mob_x) else np.nan
        cplx = (mob_dx / mob_x) if (np.isfinite(mob_x) and mob_x > 0
                                     and np.isfinite(mob_dx)) else np.nan
        cplx = float(cplx) if np.isfinite(cplx) else np.nan
        return mob, cplx
    except Exception:
        return np.nan, np.nan

def entropy_block(trace):
    try:
        pe = ant.perm_entropy(trace, order=3, delay=1, normalize=True)
    except Exception:
        pe = np.nan
    try:
        se = ant.sample_entropy(trace)
    except Exception:
        se = np.nan
    try:
        sign_seq = np.sign(trace - np.median(trace)).astype(int)
        sign_seq[sign_seq == 0] = 1
        bin_str = ''.join('1' if v > 0 else '0' for v in sign_seq)
        if len(set(bin_str)) < 2:
            lz = 0.0
        else:
            lz = ant.lziv_complexity(bin_str, normalize=True)
    except Exception:
        lz = np.nan
    hjorth_mob, hjorth_cplx = hjorth_block(trace)
    return pe, se, lz, hjorth_mob, hjorth_cplx

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

def preprocess_subject(sub_id, reject_thresh):
    set_path = os.path.join(
        DATA_ROOT, f"sub-{sub_id}", "ses-P3", "eeg",
        f"sub-{sub_id}_ses-P3_task-P3_eeg.set",
    )
    if not os.path.exists(set_path):
        return None, None, None

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

    events, _ = mne.events_from_annotations(raw, verbose=False)
    return raw, events, raw.info['sfreq']

def extract_real_trial_entropy(sub_id, reject_thresh, rows):
    raw, events, sfreq = preprocess_subject(sub_id, reject_thresh)
    if raw is None:
        return

    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=reject_thresh),
        preload=True, verbose=False,
    )
    if len(epochs) < MIN_TRIALS_REQUIRED:
        return
    codes = epochs.events[:, 2]
    keep = np.isin(codes, STANDARD_CODES + TARGET_CODES)
    epochs = epochs[keep]
    codes = codes[keep]
    if len(epochs) < MIN_TRIALS_REQUIRED:
        return

    times = epochs.times
    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)
    data = epochs.get_data()

    ch_names = epochs.ch_names
    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return

    for i in range(len(epochs)):
        seg = data[i, fz_idx, m_early]
        pe, se, lz, hjmob, hjcplx = entropy_block(seg)
        p300 = float(np.mean(data[i, pz_idx, m_p300]))
        rows.append({
            'subject': f'sub-{sub_id}', 'kind': 'real',
            'config': 'real',
            'trial_idx': i,
            'early_pe': pe, 'early_sampen': se, 'early_lz': lz,
            'early_hjorth_mob': hjmob, 'early_hjorth_cplx': hjcplx,
            'p300_amp': p300,
        })

def extract_pseudo_entropy(sub_id, config_label, min_gap_s, reject_thresh,
                            rng, rows):
    raw, events, sfreq = preprocess_subject(sub_id, reject_thresh)
    if raw is None:
        return

    real_sample_indices = events[:, 0]
    n_samples = raw.n_times

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
    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)

    ch_names = epochs_pseudo.ch_names
    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return

    for i in range(len(epochs_pseudo)):
        seg = data_p[i, fz_idx, m_early]
        pe, se, lz, hjmob, hjcplx = entropy_block(seg)
        p300 = float(np.mean(data_p[i, pz_idx, m_p300]))
        rows.append({
            'subject': f'sub-{sub_id}', 'kind': 'pseudo',
            'config': config_label,
            'trial_idx': i,
            'early_pe': pe, 'early_sampen': se, 'early_lz': lz,
            'early_hjorth_mob': hjmob, 'early_hjorth_cplx': hjcplx,
            'p300_amp': p300,
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
    banner("07_entropy_pseudotrial.py — pseudotrial test for the entropy family")
    print("Three entropy measures from the rejected manuscript, tested against")
    print("the same pseudotrial control as 05/06.\n")

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]

    configs = [
        ('config4', 0.5, 150e-6),
    ]
    all_rows = []

    banner("Extracting REAL trials (threshold ±100 µV, matches original 03)")
    for sid in sub_ids:
        extract_real_trial_entropy(sid, 100e-6, all_rows)
    n_real = len([r for r in all_rows if r['kind'] == 'real'])
    n_real_subj = len(set(r['subject'] for r in all_rows if r['kind'] == 'real'))
    print(f"  -> {n_real} real trials from {n_real_subj} subjects\n")

    for cfg_label, min_gap, thresh in configs:
        banner(f"Configuration: {cfg_label}  min_gap={min_gap} s  threshold=±{thresh*1e6:.0f} µV")
        rng = np.random.default_rng(PSEUDOTRIAL_SEED)
        for sid in sub_ids:
            extract_pseudo_entropy(sid, cfg_label, min_gap, thresh, rng, all_rows)
        n_p = len([r for r in all_rows if r['config'] == cfg_label])
        n_s = len(set(r['subject'] for r in all_rows
                      if r['config'] == cfg_label))
        print(f"  -> {n_p} pseudotrials from {n_s} subjects\n")

    if not all_rows:
        print("No data extracted. Aborting.")
        return

    df_all = pd.DataFrame(all_rows)

    feat = ['early_pe', 'early_sampen', 'early_lz',
            'early_hjorth_mob', 'early_hjorth_cplx', 'p300_amp']
    for c in feat:
        df_all[c + '_z'] = (
            df_all
            .groupby(['kind', 'config', 'subject'])[c]
            .transform(robust_z_within_subject)
        )

    models = [
        ('M_PE_Fz_0_150',         'p300_amp_z ~ early_pe_z',            'early_pe_z'),
        ('M_SE_Fz_0_150',         'p300_amp_z ~ early_sampen_z',        'early_sampen_z'),
        ('M_LZ_Fz_0_150',         'p300_amp_z ~ early_lz_z',            'early_lz_z'),

        ('M_HJORTHMOB_Fz_0_150',  'p300_amp_z ~ early_hjorth_mob_z',    'early_hjorth_mob_z'),
        ('M_HJORTHCPLX_Fz_0_150', 'p300_amp_z ~ early_hjorth_cplx_z',   'early_hjorth_cplx_z'),
    ]

    results = []

    df_real = df_all[df_all['kind'] == 'real'].copy()
    for name, formula, pred in models:
        sub = df_real.dropna(subset=[pred, 'p300_amp_z']).copy()
        if len(sub) < MIN_TRIALS_REQUIRED or sub['subject'].nunique() < 3:
            continue
        r = fit_RI(sub, formula, pred)
        r.update(dict(model=name, kind='real', config='real',
                      formula=formula))
        results.append(r)

    for cfg, _, _ in configs:
        sub_cfg = df_all[(df_all['kind'] == 'pseudo')
                         & (df_all['config'] == cfg)].copy()
        if len(sub_cfg) < MIN_TRIALS_REQUIRED or sub_cfg['subject'].nunique() < 3:
            continue
        for name, formula, pred in models:
            sub = sub_cfg.dropna(subset=[pred, 'p300_amp_z']).copy()
            if len(sub) < MIN_TRIALS_REQUIRED:
                continue
            r = fit_RI(sub, formula, pred)
            r.update(dict(model=name, kind='pseudo', config=cfg,
                          formula=formula))
            results.append(r)

    df_res = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, 'logs',
                             'entropy_pseudotrial_results.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"\nEntropy pseudotrial results saved -> {out_path}\n")

    banner("MODEL-BY-MODEL: real vs pseudotrial across configs")
    print(f"  {'model':<24} {'kind/config':<12} {'n':>6} {'beta':>10} {'R²':>10}")
    print("  " + "-" * 66)
    for name, _, _ in models:
        rows = df_res[df_res['model'] == name].copy()

        for _, r in rows[rows['kind'] == 'real'].iterrows():
            print(f"  {name:<24} {'REAL':<12} {int(r['n_trials']):>6} "
                  f"{r['beta']:>+10.3f} {r['R2_marginal']:>10.3f}")
        for cfg_lbl in ['config1', 'config2', 'config3', 'config4']:
            sel = rows[(rows['kind'] == 'pseudo') & (rows['config'] == cfg_lbl)]
            for _, r in sel.iterrows():
                print(f"  {name:<24} {cfg_lbl:<12} {int(r['n_trials']):>6} "
                      f"{r['beta']:>+10.3f} {r['R2_marginal']:>10.3f}")
        print()

    banner("CONFIG 4 SUMMARY: REAL vs PSEUDO (cleanest test)")
    print(f"\n  {'model':<24} {'β_real':>10} {'β_pseudo':>10} "
          f"{'R²_real':>10} {'R²_pseudo':>10} {'ΔR² (real-pseudo)':>20}")
    print("  " + "-" * 90)
    for name, _, _ in models:
        rows = df_res[df_res['model'] == name]
        rr = rows[rows['kind'] == 'real']
        rp = rows[(rows['kind'] == 'pseudo') & (rows['config'] == 'config4')]
        if len(rr) and len(rp):
            b_r = float(rr['beta'].iloc[0])
            b_p = float(rp['beta'].iloc[0])
            r2_r = float(rr['R2_marginal'].iloc[0])
            r2_p = float(rp['R2_marginal'].iloc[0])
            dr2 = r2_r - r2_p
            print(f"  {name:<24} {b_r:>+10.3f} {b_p:>+10.3f} "
                  f"{r2_r:>10.3f} {r2_p:>10.3f} {dr2:>+20.3f}")

    print()

if __name__ == "__main__":
    main()
