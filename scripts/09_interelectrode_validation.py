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
    P300_CHANNEL, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
    MIN_TRIALS_REQUIRED,
    PSEUDOTRIAL_SEED,
    get_channel_index, banner,
)

warnings.filterwarnings('ignore', category=Warning)
mne.set_log_level('WARNING')

                                              
CFG_MIN_GAP = 0.5
CFG_THRESH = 150e-6
CFG_LABEL = 'config4'

                                                                            
def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def robust_z_within_subject(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

def perm_entropy_safe(trace):
    try:
        return ant.perm_entropy(trace, order=3, delay=1, normalize=True)
    except Exception:
        return np.nan

def hjorth_mobility_safe(trace):
    try:
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or len(trace) < 3:
            return np.nan
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        return float(np.sqrt(var_dx / var_x)) if var_x > 0 else np.nan
    except Exception:
        return np.nan

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
                n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
                random_state=ICA_RANDOM_STATE, max_iter='auto')
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

def get_epochs(raw, events, reject_thresh, restrict_codes=True):
    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE, reject=dict(eeg=reject_thresh),
        preload=True, verbose=False)
    if restrict_codes and len(epochs) > 0:
        codes = epochs.events[:, 2]
        keep = np.isin(codes, STANDARD_CODES + TARGET_CODES)
        epochs = epochs[keep]
    return epochs

def extract_features_all_electrodes(epochs):
\
\
\
\
\
\
\
       
    picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
    eeg_names = [epochs.ch_names[i] for i in picks]
    times = epochs.times
    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)
    data = epochs.get_data(picks=picks)                        
    n_tr, n_el, _ = data.shape

    early_mean = data[:, :, m_early].mean(axis=2)                        
    late_mean = data[:, :, m_p300].mean(axis=2)                          

    early_pe = np.full((n_tr, n_el), np.nan)
    early_mob = np.full((n_tr, n_el), np.nan)
    for i in range(n_tr):
        for j in range(n_el):
            seg = data[i, j, m_early]
            early_pe[i, j] = perm_entropy_safe(seg)
            early_mob[i, j] = hjorth_mobility_safe(seg)

                          
    pz_name, pz_local = get_channel_index(eeg_names, P300_CHANNEL, P300_FALLBACK)
    if pz_name is None:
        p300_pz = None
    else:
        p300_pz = late_mean[:, pz_local].copy()

    return eeg_names, early_mean, early_pe, early_mob, late_mean, p300_pz

                                                                            
def fit_RI(df, formula, predictor):
    for kwargs in [dict(method='lbfgs', reml=True),
                   dict(method='nm', maxiter=2000, reml=True)]:
        try:
            md = smf.mixedlm(formula, df, groups=df['subject'])
            fit = md.fit(**kwargs)
            if not np.isfinite(float(fit.llf)):
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
                var_r = float(np.trace(np.atleast_2d(np.asarray(fit.cov_re))))
            except Exception:
                var_r = 0.0
            denom = var_f + var_r + var_e
            r2m = var_f / denom if denom > 0 else np.nan
            return dict(beta=beta, SE=se, z=z, p=p, R2_marginal=r2m,
                        n_trials=len(df), n_subjects=df['subject'].nunique())
        except Exception:
            continue
    return dict(beta=np.nan, SE=np.nan, z=np.nan, p=np.nan, R2_marginal=np.nan,
                n_trials=len(df), n_subjects=df['subject'].nunique())

                                                                            
def main():
    banner("09_interelectrode_validation.py — three signatures across the montage")
    print("Config 4 only (min_gap=0.5 s, ±150 µV) — cleanest pseudotrial test.\n")

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]

                                                              
    rows_v1 = []                                                              
    rows_v2 = []                                                                    
    rows_v3 = []                                                                         

    rng = np.random.default_rng(PSEUDOTRIAL_SEED)

    for sid in sub_ids:
                        
        raw, events, sfreq = preprocess_subject(sid, CFG_THRESH)
        if raw is None:
            continue
        epochs_real = get_epochs(raw, events, CFG_THRESH, restrict_codes=True)
        if len(epochs_real) < MIN_TRIALS_REQUIRED:
            continue
        (names_r, em_r, pe_r, mob_r, lm_r, pz_r) =\
            extract_features_all_electrodes(epochs_real)
        if pz_r is None:
            continue

        for j, ename in enumerate(names_r):
            for i in range(len(epochs_real)):
                rows_v1.append({'subject': f'sub-{sid}', 'kind': 'real',
                                           : ename,
                                            : em_r[i, j], 'p300_pz': pz_r[i]})
                rows_v2.append({'subject': f'sub-{sid}', 'kind': 'real',
                                           : ename,
                                            : em_r[i, j], 'late_same': lm_r[i, j]})
                rows_v3.append({'subject': f'sub-{sid}', 'kind': 'real',
                                           : ename,
                                          : pe_r[i, j], 'early_mob': mob_r[i, j],
                                         : pz_r[i]})

                          
        n_real_retained = len(epochs_real)
        pseudo_samples = generate_pseudotrial_samples(
            n_pseudo=max(n_real_retained, 1), sfreq=sfreq,
            n_continuous_samples=raw.n_times,
            real_event_samples=events[:, 0],
            min_gap_seconds=CFG_MIN_GAP, tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
            rng=rng)
        if len(pseudo_samples) < MIN_TRIALS_REQUIRED:
            continue
        pseudo_events = np.column_stack([
            pseudo_samples, np.zeros(len(pseudo_samples), dtype=int),
            np.full(len(pseudo_samples), 99, dtype=int)])
        epochs_pseudo = mne.Epochs(
            raw, pseudo_events, event_id={'pseudo': 99},
            tmin=TMIN_EPOCH, tmax=TMAX_EPOCH, baseline=BASELINE,
            reject=dict(eeg=CFG_THRESH), preload=True, verbose=False)
        if len(epochs_pseudo) < MIN_TRIALS_REQUIRED:
            continue
        (names_p, em_p, pe_p, mob_p, lm_p, pz_p) =\
            extract_features_all_electrodes(epochs_pseudo)
        if pz_p is None:
            continue

        for j, ename in enumerate(names_p):
            for i in range(len(epochs_pseudo)):
                rows_v1.append({'subject': f'sub-{sid}', 'kind': 'pseudo',
                                           : ename,
                                            : em_p[i, j], 'p300_pz': pz_p[i]})
                rows_v2.append({'subject': f'sub-{sid}', 'kind': 'pseudo',
                                           : ename,
                                            : em_p[i, j], 'late_same': lm_p[i, j]})
                rows_v3.append({'subject': f'sub-{sid}', 'kind': 'pseudo',
                                           : ename,
                                          : pe_p[i, j], 'early_mob': mob_p[i, j],
                                         : pz_p[i]})

        print(f"  sub-{sid}: real={len(epochs_real)}  pseudo={len(epochs_pseudo)}")

    if not rows_v1:
        print("No data extracted. Aborting.")
        return

    df1 = pd.DataFrame(rows_v1)
    df2 = pd.DataFrame(rows_v2)
    df3 = pd.DataFrame(rows_v3)

                                                            
    def z_by(df, cols, group_extra):
        for c in cols:
            df[c + '_z'] = (df.groupby(['kind', 'electrode', 'subject'])[c]
                              .transform(robust_z_within_subject))
        return df

    df1 = z_by(df1, ['early_mean', 'p300_pz'], None)
    df2 = z_by(df2, ['early_mean', 'late_same'], None)
    df3 = z_by(df3, ['early_pe', 'early_mob', 'p300_pz'], None)

    out_dir = os.path.join(RESULTS_DIR, 'logs')
    os.makedirs(out_dir, exist_ok=True)

                                                 
    banner("VALIDATION 1: early MEAN at each electrode -> P300 at Pz")
    electrodes = sorted(df1['electrode'].unique())
    res1 = []
    for ename in electrodes:
        for kind in ['real', 'pseudo']:
            sub = df1[(df1['electrode'] == ename) & (df1['kind'] == kind)]
            sub = sub.dropna(subset=['early_mean_z', 'p300_pz_z'])
            if len(sub) < MIN_TRIALS_REQUIRED or sub['subject'].nunique() < 3:
                continue
            r = fit_RI(sub, 'p300_pz_z ~ early_mean_z', 'early_mean_z')
            r.update(dict(electrode=ename, kind=kind, measure='mean_cross_to_Pz'))
            res1.append(r)
    df_res1 = pd.DataFrame(res1)
    df_res1.to_csv(os.path.join(out_dir, 'interelectrode_val1_cross_to_Pz.csv'),
                   index=False)
    _print_val(df_res1, 'mean_cross_to_Pz')

                                                  
    banner("VALIDATION 2: early MEAN at each electrode -> P300 at SAME electrode")
    res2 = []
    for ename in electrodes:
        for kind in ['real', 'pseudo']:
            sub = df2[(df2['electrode'] == ename) & (df2['kind'] == kind)]
            sub = sub.dropna(subset=['early_mean_z', 'late_same_z'])
            if len(sub) < MIN_TRIALS_REQUIRED or sub['subject'].nunique() < 3:
                continue
            r = fit_RI(sub, 'late_same_z ~ early_mean_z', 'early_mean_z')
            r.update(dict(electrode=ename, kind=kind, measure='mean_same_channel'))
            res2.append(r)
    df_res2 = pd.DataFrame(res2)
    df_res2.to_csv(os.path.join(out_dir, 'interelectrode_val2_same_channel.csv'),
                   index=False)
    _print_val(df_res2, 'mean_same_channel')

                                                          
    banner("VALIDATION 3: early PERM-ENTROPY & HJORTH-MOBILITY -> P300 at Pz")
    res3 = []
    for ename in electrodes:
        for kind in ['real', 'pseudo']:
            sub = df3[(df3['electrode'] == ename) & (df3['kind'] == kind)]
                                 
            s_pe = sub.dropna(subset=['early_pe_z', 'p300_pz_z'])
            if len(s_pe) >= MIN_TRIALS_REQUIRED and s_pe['subject'].nunique() >= 3:
                r = fit_RI(s_pe, 'p300_pz_z ~ early_pe_z', 'early_pe_z')
                r.update(dict(electrode=ename, kind=kind, measure='perm_entropy_to_Pz'))
                res3.append(r)
                             
            s_mob = sub.dropna(subset=['early_mob_z', 'p300_pz_z'])
            if len(s_mob) >= MIN_TRIALS_REQUIRED and s_mob['subject'].nunique() >= 3:
                r = fit_RI(s_mob, 'p300_pz_z ~ early_mob_z', 'early_mob_z')
                r.update(dict(electrode=ename, kind=kind, measure='hjorth_mob_to_Pz'))
                res3.append(r)
    df_res3 = pd.DataFrame(res3)
    df_res3.to_csv(os.path.join(out_dir, 'interelectrode_val3_shape_to_Pz.csv'),
                   index=False)
    _print_val(df_res3, 'perm_entropy_to_Pz')
    _print_val(df_res3, 'hjorth_mob_to_Pz')

                                    
    combined = pd.concat([df_res1, df_res2, df_res3], ignore_index=True)
    combined.to_csv(os.path.join(out_dir, 'interelectrode_all.csv'), index=False)
    print(f"\nAll inter-electrode results saved -> {out_dir}\\interelectrode_all.csv\n")

                                                                           
    _signature_summary(df_res1, df_res2, df_res3)

                                                                            
def _print_val(df_res, measure):
    sub = df_res[df_res['measure'] == measure].copy()
    if sub.empty:
        print(f"  [no results for {measure}]")
        return
    print(f"\n  {measure}")
    print(f"  {'electrode':<10} {'β_real':>9} {'β_pseudo':>10} "
          f"{'|β|ratio':>9} {'R²_real':>9} {'R²_pseudo':>10} {'ΔR²':>8}")
    print("  " + "-" * 72)
    electrodes = sorted(sub['electrode'].unique())
    for e in electrodes:
        rr = sub[(sub['electrode'] == e) & (sub['kind'] == 'real')]
        rp = sub[(sub['electrode'] == e) & (sub['kind'] == 'pseudo')]
        if len(rr) and len(rp):
            br, bp = float(rr['beta'].iloc[0]), float(rp['beta'].iloc[0])
            r2r, r2p = float(rr['R2_marginal'].iloc[0]), float(rp['R2_marginal'].iloc[0])
            ratio = abs(bp) / abs(br) if br != 0 else float('nan')
            print(f"  {e:<10} {br:>+9.3f} {bp:>+10.3f} {ratio:>9.2f} "
                  f"{r2r:>9.3f} {r2p:>10.3f} {r2r - r2p:>+8.3f}")

def _signature_summary(df1, df2, df3):
    banner("SIGNATURE SUMMARY (classify each electrode by pseudotrial behaviour)")

    def classify(br, bp, r2r, r2p):
        if r2r < 0.005 and r2p < 0.005:
            return 'null'
        ratio = abs(bp) / abs(br) if br != 0 else float('nan')
        if not np.isfinite(ratio):
            return 'undef'
        if ratio > 1.3:
            return 'AUTOCORR (β grows)'
        if ratio < 0.85:
            return 'STIM-LOCKED (β shrinks)'
        return 'preserved'

    print("\n  VALIDATION 1 (mean cross-to-Pz) — expect AUTOCORR broadly:")
    _classify_block(df1, 'mean_cross_to_Pz', classify)

    print("\n  VALIDATION 2 (mean same-channel) — expect preserved-β; Pz largest R²:")
    _classify_block(df2, 'mean_same_channel', classify, show_r2_rank=True)

    print("\n  VALIDATION 3a (perm-entropy to Pz) — expect STIM-LOCKED centro-parietally:")
    _classify_block(df3, 'perm_entropy_to_Pz', classify)
    print("\n  VALIDATION 3b (hjorth-mobility to Pz) — expect STIM-LOCKED centro-parietally:")
    _classify_block(df3, 'hjorth_mob_to_Pz', classify)

def _classify_block(df_res, measure, classify, show_r2_rank=False):
    sub = df_res[df_res['measure'] == measure]
    if sub.empty:
        print("    [no results]")
        return
    rows = []
    for e in sorted(sub['electrode'].unique()):
        rr = sub[(sub['electrode'] == e) & (sub['kind'] == 'real')]
        rp = sub[(sub['electrode'] == e) & (sub['kind'] == 'pseudo')]
        if len(rr) and len(rp):
            br, bp = float(rr['beta'].iloc[0]), float(rp['beta'].iloc[0])
            r2r, r2p = float(rr['R2_marginal'].iloc[0]), float(rp['R2_marginal'].iloc[0])
            rows.append((e, br, bp, r2r, r2p, classify(br, bp, r2r, r2p)))
    if show_r2_rank:
        rows.sort(key=lambda x: -x[3])                         
    counts = {}
    for e, br, bp, r2r, r2p, sig in rows:
        counts[sig] = counts.get(sig, 0) + 1
           
    tally = ', '.join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"    tally across {len(rows)} electrodes -> {tally}")
    if show_r2_rank:
        print("    top 5 electrodes by R²_real:")
        for e, br, bp, r2r, r2p, sig in rows[:5]:
            print(f"      {e:<8} R²_real={r2r:.3f}  R²_pseudo={r2p:.3f}  "
                  f"β_real={br:+.3f}  β_pseudo={bp:+.3f}  [{sig}]")

if __name__ == "__main__":
    main()
