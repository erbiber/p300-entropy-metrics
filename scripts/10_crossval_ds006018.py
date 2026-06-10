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

from config_ds006018 import (
    DATASET_ID, CACHE_DIR, TASK_FILTER,
    SFREQ_EXPECTED,
    TMIN_EPOCH, TMAX_EPOCH, BASELINE,
    EARLY_WINDOW, P300_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    EOG_CHANNELS, MASTOID_CHANNELS,
    TRIAL_CODES, TARGET_CODES, STANDARD_CODES,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT,
    ICA_METHOD, ICA_RANDOM_STATE,
    MIN_TRIALS_REQUIRED, PSEUDOTRIAL_SEED,
    PERM_ENTROPY_ORDER, PERM_ENTROPY_DELAY,
    RESULTS_DIR_DS,
    get_channel_index, banner,
)

warnings.filterwarnings('ignore', category=Warning)
mne.set_log_level('WARNING')

                                                                            
             
                                                                            
SUBSET_N = None                                                     
                                                          
CFG_MIN_GAP = 0.5
CFG_THRESH = 150e-6

                                                                            
                                                         
                                                                            
def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def robust_z_within_subject(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

def entropy_block(trace):
    try:
        pe = ant.perm_entropy(trace, order=PERM_ENTROPY_ORDER,
                              delay=PERM_ENTROPY_DELAY, normalize=True)
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
        lz = 0.0 if len(set(bin_str)) < 2 else ant.lziv_complexity(
            bin_str, normalize=True)
    except Exception:
        lz = np.nan
    return pe, se, lz

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

                                                                            
                                 
                                                                            
def load_visualoddball_recordings():
                                                                             
    from eegdash.dataset import DS006018
    print(f"Loading {DATASET_ID} via eegdash (cache: {CACHE_DIR}) ...")
    dataset = DS006018(cache_dir=CACHE_DIR)
    print(f"  total recordings: {len(dataset.datasets)}")
    recs = []
    for rec in dataset.datasets:
        try:
            desc = rec.description
            task = desc.get('task', None) if hasattr(
                desc, 'get') else desc['task']
            subj = desc.get('subject', None) if hasattr(
                desc, 'get') else desc['subject']
        except Exception:
            task, subj = None, None
        if task == TASK_FILTER:
            recs.append((str(subj), rec))
    print(f"  visualoddball recordings: {len(recs)}")
    return recs

def preprocess_recording(rec):
                                                                    
    raw = rec.raw
    if raw is None:
        return None

                                                                                   
    try:
        raw.load_data()
    except Exception:
        pass

                                                                        
    type_map = {}
    for ch in EOG_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = 'eog'
    for ch in MASTOID_CHANNELS:
        if ch in raw.ch_names:
            type_map[ch] = 'misc'
    if type_map:
        try:
            raw.set_channel_types(type_map)
        except Exception:
            pass

    raw.filter(FILTER_LOW, FILTER_HIGH,
               fir_design=FILTER_DESIGN, verbose=False)

    if DO_ICA:
        try:
            raw_for_ica = raw.copy().filter(
                ICA_HIGHPASS_FOR_FIT, None, fir_design=FILTER_DESIGN, verbose=False)
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

    return raw

def get_trial_epochs(raw, reject_thresh):
\
\
\
\
\
       
    events, event_id = mne.events_from_annotations(raw, verbose=False)
                                                                             
    trial_event_id = {k: v for k, v in event_id.items()
                      if v in TRIAL_CODES and v in events[:, 2]}
    if not trial_event_id:
        return None, events
    epochs = mne.Epochs(
        raw, events, event_id=trial_event_id,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE, reject=dict(eeg=reject_thresh),
        preload=True, verbose=False)
    return (epochs if len(epochs) > 0 else None), events

                                                                            
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
            var_f = float(np.var(X @ np.asarray(fit.fe_params)))
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

                                                                            
def extract_trial_rows(epochs, subject, kind, config, rows, trial_codes_set):
                                                            
    times = epochs.times
    m_early = window_mask(times, EARLY_WINDOW)
    m_p300 = window_mask(times, P300_WINDOW)
    data = epochs.get_data()
    ch = epochs.ch_names

    fz_name, fz_idx = get_channel_index(ch, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return

    codes = epochs.events[:, 2] if kind == 'real' else None
    for i in range(len(epochs)):
        fz_seg = data[i, fz_idx, m_early]
        pz_seg = data[i, pz_idx, m_early]
        pe, se, lz = entropy_block(fz_seg)
        is_target = (codes is not None and int(codes[i]) in TARGET_CODES)
        rows.append({
                     : subject, 'kind': kind, 'config': config,
                          : float(np.sqrt(np.mean(fz_seg ** 2))),
                           : float(np.mean(fz_seg)),
                           : float(np.mean(pz_seg)),
                          : float(np.sqrt(np.mean(pz_seg ** 2))),
                   : pe, 'fz_se': se, 'fz_lz': lz,
                     : float(np.mean(data[i, pz_idx, m_p300])),
                       : int(is_target) if codes is not None else -1,
        })

                                                                            
def main():
    banner("10_crossval_ds006018.py — cross-validation of Paper 1 signatures")
    trial_kind = ("TARGETS ONLY" if set(TRIAL_CODES) == set(TARGET_CODES)
                  else "ALL STIMULI (targets+standards)"
                  if set(TRIAL_CODES) == set(TARGET_CODES) | set(STANDARD_CODES)
                  else "CUSTOM")
    print(
        f"Trial definition (from config): {trial_kind}  ({len(TRIAL_CODES)} codes)")
    print(
        f"Subset: {'first ' + str(SUBSET_N) + ' subjects' if SUBSET_N else 'FULL ~127'}\n")

    recs = load_visualoddball_recordings()
    if SUBSET_N is not None:
        recs = recs[:SUBSET_N]

    all_rows = []
    per_subject_counts = []
    rng = np.random.default_rng(PSEUDOTRIAL_SEED)
    trial_codes_set = set(TRIAL_CODES)

    for subj, rec in recs:
        try:
            raw = preprocess_recording(rec)
        except Exception as e:
            print(f"  sub-{subj}: preprocess failed ({e})")
            continue
        if raw is None:
            print(f"  sub-{subj}: no raw")
            continue
        if abs(raw.info['sfreq'] - SFREQ_EXPECTED) > 1:
            print(f"  sub-{subj}: unexpected sfreq {raw.info['sfreq']}")

        epochs_real, events = get_trial_epochs(raw, CFG_THRESH)
        if epochs_real is None or len(epochs_real) < MIN_TRIALS_REQUIRED:
            print(f"  sub-{subj}: too few real trials")
            continue
        n_real = len(epochs_real)
        n_targ = int(np.isin(epochs_real.events[:, 2], TARGET_CODES).sum())
        per_subject_counts.append((subj, n_real, n_targ))

        extract_trial_rows(epochs_real, f'sub-{subj}', 'real', 'real',
                           all_rows, trial_codes_set)

                                
        n_pseudo_kept = 0
        pseudo_samples = generate_pseudotrial_samples(
            n_pseudo=max(n_real, 1), sfreq=raw.info['sfreq'],
            n_continuous_samples=raw.n_times,
            real_event_samples=events[:, 0],
            min_gap_seconds=CFG_MIN_GAP, tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
            rng=rng)
        if len(pseudo_samples) >= MIN_TRIALS_REQUIRED:
            pe_events = np.column_stack([
                pseudo_samples, np.zeros(len(pseudo_samples), dtype=int),
                np.full(len(pseudo_samples), 99, dtype=int)])
            epochs_pseudo = mne.Epochs(
                raw, pe_events, event_id={'pseudo': 99},
                tmin=TMIN_EPOCH, tmax=TMAX_EPOCH, baseline=BASELINE,
                reject=dict(eeg=CFG_THRESH), preload=True, verbose=False)
            if len(epochs_pseudo) >= MIN_TRIALS_REQUIRED:
                n_pseudo_kept = len(epochs_pseudo)
                extract_trial_rows(epochs_pseudo, f'sub-{subj}', 'pseudo',
                                            , all_rows, trial_codes_set)

        print(f"  sub-{subj}: real={n_real} (targets={n_targ})  "
              f"pseudo={n_pseudo_kept}")

                                                                                 
        try:
            del raw, epochs_real
        except NameError:
            pass
        try:
            del epochs_pseudo
        except NameError:
            pass

    if not all_rows:
        print("\nNo data extracted. Check trial definition / task filter.")
        return

    df = pd.DataFrame(all_rows)

                                                                       
    banner("PER-SUBJECT TRIAL COUNTS (reconciliation diagnostic)")
    counts = np.array([c[1] for c in per_subject_counts])
    targs = np.array([c[2] for c in per_subject_counts])
    print(f"  retained real trials/subject: mean={counts.mean():.1f} "
          f"median={np.median(counts):.0f} min={counts.min()} max={counts.max()}")
    print(f"  target trials/subject:        mean={targs.mean():.1f}")
    print(f"\n  Primary dataset was 40.1/subject.")
    print(f"  -> If this run's mean ≈ 40 and you used ALL stimuli, rejection is heavy.")
    print(f"  -> If this run's mean ≈ 40 and matches target count, primary likely TARGETS-ONLY.")
    print(f"  -> If this run's mean ≈ 140-160, primary likely used ALL stimuli.\n")

                                                         
    feats = ['fz_early_rms', 'fz_early_mean', 'pz_early_mean', 'pz_early_rms',
                    , 'fz_se', 'fz_lz', 'p300_pz']
    for c in feats:
        df[c + '_z'] = (df.groupby(['kind', 'config', 'subject'])[c]
                          .transform(robust_z_within_subject))

                                                    
    models = [
                                                                       
        ('M1_RMS_Fz_to_Pz',   'p300_pz_z ~ fz_early_rms_z',  'fz_early_rms_z'),
        ('M4a_mean_Fz_to_Pz', 'p300_pz_z ~ fz_early_mean_z', 'fz_early_mean_z'),
                                                               
        ('M9a_mean_Pz_to_Pz', 'p300_pz_z ~ pz_early_mean_z', 'pz_early_mean_z'),
        ('M8_RMS_Pz_to_Pz',   'p300_pz_z ~ pz_early_rms_z',  'pz_early_rms_z'),
                                                                     
        ('M_PE_Fz_to_Pz',     'p300_pz_z ~ fz_pe_z',         'fz_pe_z'),
        ('M_SE_Fz_to_Pz',     'p300_pz_z ~ fz_se_z',         'fz_se_z'),
        ('M_LZ_Fz_to_Pz',     'p300_pz_z ~ fz_lz_z',         'fz_lz_z'),
    ]

    results = []
    for kind, cfg in [('real', 'real'), ('pseudo', 'config4')]:
        sub_kc = df[(df['kind'] == kind) & (df['config'] == cfg)]
        for name, formula, pred in models:
            s = sub_kc.dropna(subset=[pred, 'p300_pz_z'])
            if len(s) < MIN_TRIALS_REQUIRED or s['subject'].nunique() < 3:
                continue
            r = fit_RI(s, formula, pred)
            r.update(dict(model=name, kind=kind, config=cfg))
            results.append(r)

    df_res = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR_DS, exist_ok=True)
    out = os.path.join(RESULTS_DIR_DS, 'crossval_ds006018_results.csv')
    df_res.to_csv(out, index=False)
    print(f"Cross-validation results saved -> {out}\n")

                                                 
    banner("CROSS-VALIDATION SUMMARY: real vs pseudo (config4)")
    print(f"\n  {'model':<22} {'β_real':>9} {'β_pseudo':>10} {'|β|ratio':>9} "
          f"{'R²_real':>9} {'R²_pseudo':>10} {'signature':<22}")
    print("  " + "-" * 95)
    for name, _, _ in models:
        rr = df_res[(df_res['model'] == name) & (df_res['kind'] == 'real')]
        rp = df_res[(df_res['model'] == name) & (df_res['kind'] == 'pseudo')]
        if len(rr) and len(rp):
            br, bp = float(rr['beta'].iloc[0]), float(rp['beta'].iloc[0])
            r2r, r2p = float(rr['R2_marginal'].iloc[0]), float(
                rp['R2_marginal'].iloc[0])
            ratio = abs(bp) / abs(br) if br != 0 else float('nan')
            if r2r < 0.005 and r2p < 0.005:
                sig = 'null'
            elif ratio > 1.3:
                sig = 'AUTOCORR (β grows)'
            elif ratio < 0.85:
                sig = 'STIM-LOCKED (β shrinks)'
            else:
                sig = 'preserved'
            print(f"  {name:<22} {br:>+9.3f} {bp:>+10.3f} {ratio:>9.2f} "
                  f"{r2r:>9.3f} {r2p:>10.3f} {sig:<22}")

    banner("EXPECTED (from primary ERP CORE P3)")
    print("  M1/M4a  (Fz->Pz amplitude)  : AUTOCORR  (β grows under pseudo)")
    print("  M9a     (Pz->Pz same-chan)  : preserved (large R², β unchanged)")
    print("  M8      (Pz->Pz RMS)        : AUTOCORR / mixed")
    print("  M_PE/SE/LZ (Fz entropy)     : STIM-LOCKED (β shrinks; small R²)")
    print("\n  Replication = same signature pattern in this independent dataset.\n")

if __name__ == "__main__":
    main()
