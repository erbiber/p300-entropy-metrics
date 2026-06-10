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

                                                                            
def endpoint_block(trace):
\
\
\
\
\
       
    n = len(trace)
    out = {}

                                                        
    out['median_amp'] = float(np.median(trace))
    out['mad_amp'] = float(1.4826 * np.median(np.abs(trace - np.median(trace))))
    out['p2p_amp'] = float(np.max(trace) - np.min(trace))

                          
    try:
        out['skew_amp'] = float(stats.skew(trace, bias=False))
    except Exception:
        out['skew_amp'] = np.nan
    try:
                                                       
        out['kurt_amp'] = float(stats.kurtosis(trace, fisher=True, bias=False))
    except Exception:
        out['kurt_amp'] = np.nan

                                
    try:
        x = np.arange(n, dtype=float)
                                                            
        slope, _ = np.polyfit(x, trace, 1)
        out['slope_amp'] = float(slope)
    except Exception:
        out['slope_amp'] = np.nan

                                     
    try:
        var_x = float(np.var(trace, ddof=0))
        if var_x > 0 and n > 2:
            dx = np.diff(trace)
            var_dx = float(np.var(dx, ddof=0))
            mob_x = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan

            ddx = np.diff(dx)
            var_ddx = float(np.var(ddx, ddof=0))
            mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan

            out['hjorth_mob'] = float(mob_x) if np.isfinite(mob_x) else np.nan
            cplx = (mob_dx / mob_x) if (np.isfinite(mob_x) and mob_x > 0
                                         and np.isfinite(mob_dx)) else np.nan
            out['hjorth_cplx'] = float(cplx) if np.isfinite(cplx) else np.nan
        else:
            out['hjorth_mob'] = np.nan
            out['hjorth_cplx'] = np.nan
    except Exception:
        out['hjorth_mob'] = np.nan
        out['hjorth_cplx'] = np.nan

    return out

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def feature_block(trace):
\
\
\
\
\
       
    mean_amp = float(np.mean(trace))
    sd_amp = float(np.std(trace - mean_amp, ddof=0))
    rms_amp = float(np.sqrt(np.mean(trace ** 2)))
    return mean_amp, sd_amp, rms_amp

def basic_amplitude_block(fz_tr, pz_tr, times):
\
\
\
\
\
\
\
       
    m_early_150 = window_mask(times, EARLY_WINDOW)
    m_early_200 = window_mask(times, EARLY_WINDOW_200)
    m_early_250 = window_mask(times, EARLY_WINDOW_250)
    m_basectrl = window_mask(times, BASELINE_CONTROL_WINDOW)

    m_fz_150, sd_fz_150, rms_fz_150 = feature_block(fz_tr[m_early_150])
    m_fz_200, _, rms_fz_200 = feature_block(fz_tr[m_early_200])
    m_fz_250, _, rms_fz_250 = feature_block(fz_tr[m_early_250])
    m_pz_150, sd_pz_150, rms_pz_150 = feature_block(pz_tr[m_early_150])
    m_basef, _, rms_basef = feature_block(fz_tr[m_basectrl])
    m_basep, _, rms_basep = feature_block(pz_tr[m_basectrl])

    return {
                          
                       : m_fz_150, 'sd_early_fz': sd_fz_150,
                      : rms_fz_150,
                           : m_fz_200, 'rms_early_fz_200': rms_fz_200,
                           : m_fz_250, 'rms_early_fz_250': rms_fz_250,
                                        
                       : m_pz_150, 'sd_early_pz': sd_pz_150,
                      : rms_pz_150,
                                    
                      : m_basef, 'rms_base_fz': rms_basef,
                      : m_basep, 'rms_base_pz': rms_basep,
    }

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

def extract_real_trial_features(sub_id, reject_thresh, rows):
                                                                   
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
        fz_full = data[i, fz_idx, :]
        pz_full = data[i, pz_idx, :]
        seg = fz_full[m_early]
        feats = endpoint_block(seg)
        feats.update(basic_amplitude_block(fz_full, pz_full, times))
        p300 = float(np.mean(data[i, pz_idx, m_p300]))
        row = {
                     : f'sub-{sub_id}', 'kind': 'real',
                    : 'real',
                       : i,
                      : p300,
        }
        row.update(feats)
        rows.append(row)

def extract_pseudo_features(sub_id, config_label, min_gap_s, reject_thresh,
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
        fz_full = data_p[i, fz_idx, :]
        pz_full = data_p[i, pz_idx, :]
        seg = fz_full[m_early]
        feats = endpoint_block(seg)
        feats.update(basic_amplitude_block(fz_full, pz_full, times))
        p300 = float(np.mean(data_p[i, pz_idx, m_p300]))
        row = {
                     : f'sub-{sub_id}', 'kind': 'pseudo',
                    : config_label,
                       : i,
                      : p300,
        }
        row.update(feats)
        rows.append(row)

                                                                            
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
    banner("08_extended_endpoint_pseudotrial.py — extended endpoint family")
    print("Eight new endpoint summaries at Fz, 0-150 ms, pseudotrial-tested")
    print("under Config 4 only (the 4-config sweep was done in script 05).\n")
    print("Framed as exploratory family-completeness checks. The original")
    print("eight models (covered by 05/06) remain the formal pseudotrial")
    print("tests of the manuscript's models.\n")

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]

                                                                      
    configs = [
        ('config4', 0.5, 150e-6),
    ]
    all_rows = []

                                                                         
    banner("Extracting REAL trials (threshold ±100 µV, manuscript default)")
    for sid in sub_ids:
        extract_real_trial_features(sid, 100e-6, all_rows)
    n_real = len([r for r in all_rows if r['kind'] == 'real'])
    n_real_subj = len(set(r['subject'] for r in all_rows if r['kind'] == 'real'))
    print(f"  -> {n_real} real trials from {n_real_subj} subjects\n")

                                                      
    for cfg_label, min_gap, thresh in configs:
        banner(f"Configuration: {cfg_label}  min_gap={min_gap} s  "
               f"threshold=±{thresh*1e6:.0f} µV")
        rng = np.random.default_rng(PSEUDOTRIAL_SEED)
        for sid in sub_ids:
            extract_pseudo_features(sid, cfg_label, min_gap, thresh, rng,
                                     all_rows)
        n_p = len([r for r in all_rows if r['config'] == cfg_label])
        n_s = len(set(r['subject'] for r in all_rows
                      if r['config'] == cfg_label))
        print(f"  -> {n_p} pseudotrials from {n_s} subjects\n")

    if not all_rows:
        print("No data extracted. Aborting.")
        return

    df_all = pd.DataFrame(all_rows)

                                                        
    feat_cols = ['median_amp', 'mad_amp', 'p2p_amp',
                           , 'kurt_amp', 'slope_amp',
                             , 'hjorth_cplx',
                                                                    
                                                         
                                , 'sd_early_fz', 'rms_early_fz',
                                    , 'rms_early_fz_200',
                                    , 'rms_early_fz_250',
                                                                     
                                , 'sd_early_pz', 'rms_early_pz',
                               , 'rms_base_fz',
                               , 'rms_base_pz',
                           ]
    for c in feat_cols:
        df_all[c + '_z'] = (
            df_all
            .groupby(['kind', 'config', 'subject'])[c]
            .transform(robust_z_within_subject)
        )

                                                             
    models = [
                              
        ('M_MED_Fz_0_150',       'p300_amp_z ~ median_amp_z',  'median_amp_z'),
        ('M_MAD_Fz_0_150',       'p300_amp_z ~ mad_amp_z',     'mad_amp_z'),
        ('M_P2P_Fz_0_150',       'p300_amp_z ~ p2p_amp_z',     'p2p_amp_z'),
                              
        ('M_SKEW_Fz_0_150',      'p300_amp_z ~ skew_amp_z',    'skew_amp_z'),
        ('M_KURT_Fz_0_150',      'p300_amp_z ~ kurt_amp_z',    'kurt_amp_z'),
                                    
        ('M_SLOPE_Fz_0_150',     'p300_amp_z ~ slope_amp_z',   'slope_amp_z'),
                               
        ('M_HJORTHMOB_Fz_0_150', 'p300_amp_z ~ hjorth_mob_z',  'hjorth_mob_z'),
        ('M_HJORTHCPLX_Fz_0_150','p300_amp_z ~ hjorth_cplx_z', 'hjorth_cplx_z'),

                                                                    
                                                                      
                                            
                                          
        ('M2_Fz_mean_0_150',     'p300_amp_z ~ mean_early_fz_z',  'mean_early_fz_z'),
        ('M3_Fz_sd_0_150',       'p300_amp_z ~ sd_early_fz_z',    'sd_early_fz_z'),
        ('M4b_Fz_sd_competitive','p300_amp_z ~ mean_early_fz_z + sd_early_fz_z', 'sd_early_fz_z'),
        ('M5_Fz_RMS_0_200',      'p300_amp_z ~ rms_early_fz_200_z', 'rms_early_fz_200_z'),
        ('M6_Fz_RMS_0_250',      'p300_amp_z ~ rms_early_fz_250_z', 'rms_early_fz_250_z'),
        ('M5m_Fz_mean_0_200',    'p300_amp_z ~ mean_early_fz_200_z', 'mean_early_fz_200_z'),
        ('M6m_Fz_mean_0_250',    'p300_amp_z ~ mean_early_fz_250_z', 'mean_early_fz_250_z'),
        ('M7_Fz_baseline_RMS',   'p300_amp_z ~ rms_base_fz_z',    'rms_base_fz_z'),
        ('M7m_Fz_baseline_mean', 'p300_amp_z ~ mean_base_fz_z',   'mean_base_fz_z'),
                                          
        ('M8_Pz_RMS_0_150',      'p300_amp_z ~ rms_early_pz_z',   'rms_early_pz_z'),
        ('M9b_Pz_sd_competitive','p300_amp_z ~ mean_early_pz_z + sd_early_pz_z', 'sd_early_pz_z'),
        ('M10_Pz_baseline_RMS',  'p300_amp_z ~ rms_base_pz_z',    'rms_base_pz_z'),
        ('M11_Pz_RMS_with_basecov', 'p300_amp_z ~ rms_early_pz_z + rms_base_pz_z', 'rms_early_pz_z'),
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
                                                                        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"\nExtended endpoint results saved -> {out_path}\n")

                                    
    banner("MODEL-BY-MODEL: real vs pseudotrial across configs")
    print(f"  {'model':<28} {'kind/config':<12} {'n':>6} {'beta':>10} {'R²':>10}")
    print("  " + "-" * 70)
    for name, _, _ in models:
        rows = df_res[df_res['model'] == name].copy()
        for _, r in rows[rows['kind'] == 'real'].iterrows():
            print(f"  {name:<28} {'REAL':<12} {int(r['n_trials']):>6} "
                  f"{r['beta']:>+10.3f} {r['R2_marginal']:>10.3f}")
        for cfg_lbl in ['config1', 'config2', 'config3', 'config4']:
            sel = rows[(rows['kind'] == 'pseudo') & (rows['config'] == cfg_lbl)]
            for _, r in sel.iterrows():
                print(f"  {name:<28} {cfg_lbl:<12} {int(r['n_trials']):>6} "
                      f"{r['beta']:>+10.3f} {r['R2_marginal']:>10.3f}")
        print()

                                
    banner("CONFIG 4 SUMMARY: REAL vs PSEUDO (cleanest test)")
    print(f"\n  {'model':<28} {'β_real':>10} {'β_pseudo':>10} "
          f"{'R²_real':>10} {'R²_pseudo':>10} {'ΔR² (real-pseudo)':>20}")
    print("  " + "-" * 92)
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
            print(f"  {name:<28} {b_r:>+10.3f} {b_p:>+10.3f} "
                  f"{r2_r:>10.3f} {r2_p:>10.3f} {dr2:>+20.3f}")

    print()

if __name__ == "__main__":
    main()
