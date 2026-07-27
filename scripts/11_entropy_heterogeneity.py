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
    get_channel_index, banner,
)

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

PERM_ORDER = 3
PERM_DELAY = 1

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def robust_z_within_subject(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

def hjorth_mobility(trace):
    try:
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or len(trace) <= 2:
            return np.nan
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        mob = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan
        return float(mob) if np.isfinite(mob) else np.nan
    except Exception:
        return np.nan

def hjorth_complexity(trace):
    try:
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or len(trace) <= 3:
            return np.nan
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        if var_dx <= 0:
            return np.nan
        mob_x = np.sqrt(var_dx / var_x)
        ddx = np.diff(dx)
        var_ddx = float(np.var(ddx, ddof=0))
        mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan
        cplx = (mob_dx / mob_x) if (np.isfinite(mob_x) and mob_x > 0
                                     and np.isfinite(mob_dx)) else np.nan
        return float(cplx) if np.isfinite(cplx) else np.nan
    except Exception:
        return np.nan

def entropy_block(trace):
    try:
        pe = ant.perm_entropy(trace, order=PERM_ORDER,
                              delay=PERM_DELAY, normalize=True)
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
    hjmob = hjorth_mobility(trace)
    hjcplx = hjorth_complexity(trace)
    return pe, se, lz, hjmob, hjcplx

def extract_primary(reject_thresh=100e-6):
    banner("Extracting PRIMARY dataset (ERP CORE P3)")
    rows = []
    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))

    for d in sub_dirs:
        sid = d.split('-')[1]
        set_path = os.path.join(DATA_ROOT, f"sub-{sid}", "ses-P3", "eeg",
                                f"sub-{sid}_ses-P3_task-P3_eeg.set")
        if not os.path.exists(set_path):
            continue
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            raw.filter(FILTER_LOW, FILTER_HIGH,
                       fir_design=FILTER_DESIGN, verbose=False)
            if DO_ICA:
                try:
                    raw_ica = raw.copy().filter(
                        ICA_HIGHPASS_FOR_FIT, None,
                        fir_design=FILTER_DESIGN, verbose=False)
                    ica = mne.preprocessing.ICA(
                        n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
                        random_state=ICA_RANDOM_STATE, max_iter='auto')
                    ica.fit(raw_ica, verbose=False)
                    try:
                        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                        ica.exclude = eog_idx
                    except Exception:
                        ica.exclude = []
                    raw = ica.apply(raw, verbose=False)
                except Exception:
                    pass
            events, _ = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(
                raw, events, event_id=None,
                tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                baseline=BASELINE, reject=dict(eeg=reject_thresh),
                preload=True, verbose=False)
            if len(epochs) < MIN_TRIALS_REQUIRED:
                continue
            codes = epochs.events[:, 2]
            keep = np.isin(codes, STANDARD_CODES + TARGET_CODES)
            epochs = epochs[keep]
            if len(epochs) < MIN_TRIALS_REQUIRED:
                continue

            times = epochs.times
            m_e = window_mask(times, EARLY_WINDOW)
            m_p = window_mask(times, P300_WINDOW)
            data = epochs.get_data()
            ch = epochs.ch_names

            fz_name, fz_idx = get_channel_index(ch, EARLY_CHANNEL, EARLY_FALLBACK)
            pz_name, pz_idx = get_channel_index(ch, P300_CHANNEL, P300_FALLBACK)
            if fz_name is None or pz_name is None:
                continue

            for i in range(len(epochs)):
                seg = data[i, fz_idx, m_e]
                pe, se, lz, hjmob, hjcplx = entropy_block(seg)
                p300 = float(np.mean(data[i, pz_idx, m_p]))
                rows.append(dict(subject=f'sub-{sid}',
                                 pe=pe, se=se, lz=lz,
                                 hjorth_mob=hjmob, hjorth_cplx=hjcplx,
                                 p300=p300))
            print(f"  sub-{sid}: {len(epochs)} trials")
        except Exception as ex:
            print(f"  sub-{sid}: error ({ex})")

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df)} trials, {df['subject'].nunique()} subjects")
    return df

def extract_ds006018(reject_thresh=150e-6):
    banner("Extracting DS006018 (Isbell et al. 2025)")
    from eegdash.dataset import DS006018
    from config_ds006018 import (
        CACHE_DIR, TASK_FILTER, TRIAL_CODES, TARGET_CODES as TC_DS,
        EOG_CHANNELS, MASTOID_CHANNELS,
        EARLY_CHANNEL as EC_DS, P300_CHANNEL as PC_DS,
        EARLY_FALLBACK as EF_DS, P300_FALLBACK as PF_DS,
        PERM_ENTROPY_ORDER as PEO, PERM_ENTROPY_DELAY as PED,
        get_channel_index as gci_ds,
    )

    dataset = DS006018(cache_dir=CACHE_DIR)
    recs = []
    for rec in dataset.datasets:
        try:
            desc = rec.description
            task = desc.get('task', None) if hasattr(desc, 'get') else desc['task']
            subj = desc.get('subject', None) if hasattr(desc, 'get') else desc['subject']
        except Exception:
            task, subj = None, None
        if task == TASK_FILTER:
            recs.append((str(subj), rec))

    ckpt_path = os.path.join(RESULTS_DIR, 'logs',
                             'heterogeneity_ds006018_checkpoint.csv')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path)
        done_subjs = set(ckpt['subject'].unique())
        rows = ckpt.to_dict('records')
        print(f"  Resuming: {len(done_subjs)} subjects already in checkpoint")
    else:
        done_subjs = set()
        rows = []

    for subj, rec in recs:
        if f'sub-{subj}' in done_subjs:
            continue
        try:
            raw = rec.raw
            if raw is None:
                continue
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
                    raw_ica = raw.copy().filter(
                        ICA_HIGHPASS_FOR_FIT, None,
                        fir_design=FILTER_DESIGN, verbose=False)
                    ica = mne.preprocessing.ICA(
                        n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
                        random_state=ICA_RANDOM_STATE, max_iter='auto')
                    ica.fit(raw_ica, verbose=False)
                    try:
                        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                        ica.exclude = eog_idx
                    except Exception:
                        ica.exclude = []
                    raw = ica.apply(raw, verbose=False)
                except Exception:
                    pass

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            trial_event_id = {k: v for k, v in event_id.items()
                              if v in TRIAL_CODES and v in events[:, 2]}
            if not trial_event_id:
                continue
            epochs = mne.Epochs(
                raw, events, event_id=trial_event_id,
                tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                baseline=BASELINE, reject=dict(eeg=reject_thresh),
                preload=True, verbose=False)
            if len(epochs) < MIN_TRIALS_REQUIRED:
                continue

            times = epochs.times
            m_e = window_mask(times, EARLY_WINDOW)
            m_p = window_mask(times, P300_WINDOW)
            data = epochs.get_data()
            ch = epochs.ch_names

            fz_name, fz_idx = gci_ds(ch, EC_DS, EF_DS)
            pz_name, pz_idx = gci_ds(ch, PC_DS, PF_DS)
            if fz_name is None or pz_name is None:
                continue

            for i in range(len(epochs)):
                seg = data[i, fz_idx, m_e]
                pe, se, lz, hjmob, hjcplx = entropy_block(seg)
                p300 = float(np.mean(data[i, pz_idx, m_p]))
                rows.append(dict(subject=f'sub-{subj}',
                                 pe=pe, se=se, lz=lz,
                                 hjorth_mob=hjmob, hjorth_cplx=hjcplx,
                                 p300=p300))

            done_subjs.add(f'sub-{subj}')
            print(f"  sub-{subj}: {len(epochs)} trials")
            try:
                del raw, epochs
            except Exception:
                pass

            pd.DataFrame(rows).to_csv(ckpt_path, index=False)
        except Exception as ex:
            print(f"  sub-{subj}: error ({ex})")
            if rows:
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df)} trials, {df['subject'].nunique()} subjects")
    return df

def fit_per_subject_OLS(df, predictor):
    from scipy.stats import t as t_dist

    df = df.dropna(subset=[predictor, 'p300_z', 'subject']).copy()
    if df['subject'].nunique() < 5 or len(df) < 30:
        return None

    beta_pop = np.nan
    for kwargs in [dict(method='lbfgs', reml=True),
                   dict(method='nm', maxiter=3000, reml=True)]:
        try:
            md = smf.mixedlm(f'p300_z ~ {predictor}', df,
                             groups=df['subject'])
            fit = md.fit(**kwargs)
            if np.isfinite(float(fit.llf)):
                beta_pop = float(fit.params.get(predictor, np.nan))
                break
        except Exception:
            continue

    subj_slopes = []
    subj_ids = []
    subj_n = []
    for subj, grp in df.groupby('subject'):
        grp = grp.dropna(subset=[predictor, 'p300_z'])
        if len(grp) < 5:
            continue
        x = grp[predictor].values
        y = grp['p300_z'].values
        if np.std(x) < 1e-10:
            continue

        slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        subj_slopes.append(slope)
        subj_ids.append(subj)
        subj_n.append(len(grp))

    subj_slopes = np.array(subj_slopes)
    if len(subj_slopes) < 3:
        return None

    n_pos = int(np.sum(subj_slopes > 0))
    n_neg = int(np.sum(subj_slopes < 0))

    t_stat, t_p = stats.ttest_1samp(subj_slopes, 0)

    from scipy.stats import binomtest
    binom_p = float(binomtest(n_pos, len(subj_slopes), 0.5).pvalue)

    mean_s = float(np.mean(subj_slopes))
    sd_s = float(np.std(subj_slopes, ddof=1))
    hetero_ratio = sd_s / abs(mean_s) if abs(mean_s) > 1e-6 else np.inf

    return dict(
        beta_pop=beta_pop,
        n_subjects=len(subj_slopes),
        n_trials=len(df),
        subj_slopes=subj_slopes,
        subj_ids=subj_ids,
        mean_slope=mean_s,
        sd_slope=sd_s,
        hetero_ratio=hetero_ratio,
        n_pos=n_pos, n_neg=n_neg,
        t_stat=t_stat, t_p=t_p,
        binom_p=binom_p,
    )

def heterogeneity_report(df, dataset_name):
    banner(f"HETEROGENEITY TEST — {dataset_name}")

    for col in ['pe', 'se', 'lz', 'hjorth_mob', 'hjorth_cplx', 'p300']:
        if col in df.columns:
            df[col + '_z'] = (df.groupby('subject')[col]
                               .transform(robust_z_within_subject))

    measures = [
        ('pe_z',           'Permutation entropy'),
        ('se_z',           'Sample entropy'),
        ('lz_z',           'Lempel-Ziv complexity'),
        ('hjorth_mob_z',   'Hjorth mobility'),
        ('hjorth_cplx_z',  'Hjorth complexity'),
    ]

    results_out = []

    for pred, label in measures:
        print(f"\n--- {label} ---")
        if pred not in df.columns:

            print(f"  [skipped — {pred} not present; "
                  f"delete the ds006018 checkpoint and re-run to include it]")
            continue
        res = fit_per_subject_OLS(df, pred)
        if res is None:
            print("  [could not fit — insufficient data]")
            continue

        ss = res['subj_slopes']
        print(f"  N: {res['n_subjects']} subjects, {res['n_trials']} trials")
        print(f"  Population β (RI-LMM):     {res['beta_pop']:+.4f}")
        print(f"  Per-subject OLS slopes:")
        print(f"    mean = {res['mean_slope']:+.4f}")
        print(f"    SD   = {res['sd_slope']:.4f}")
        print(f"    range = [{np.min(ss):+.3f}, {np.max(ss):+.3f}]")
        print(f"    positive: {res['n_pos']}/{res['n_subjects']} subjects")
        print(f"    negative: {res['n_neg']}/{res['n_subjects']} subjects")
        print(f"  Heterogeneity ratio (SD/|mean|): {res['hetero_ratio']:.2f}  "
              f"(>2 = heterogeneous; <1 = homogeneous)")
        print(f"  1-sample t-test (mean ≠ 0): t={res['t_stat']:+.3f}, "
              f"p={res['t_p']:.4f}")
        print(f"  Sign test (proportion +): p={res['binom_p']:.4f}")

        sorted_slopes = sorted(zip(res['subj_ids'], ss), key=lambda x: x[1])
        print(f"  Individual slopes (sorted):")
        for sid, sl in sorted_slopes:
            bar = '█' * int(abs(sl) * 20)
            sign = '+' if sl >= 0 else '-'
            print(f"    {str(sid):<12} {sl:+.4f}  {sign}{bar}")

        print()
        results_out.append(dict(measure=label, **{k:v for k,v in res.items()
                                                   if k != 'subj_slopes'
                                                   and k != 'subj_ids'}))

    print(f"\n  KEY QUESTION for {dataset_name}:")
    print("  H_hetero: SD >> |mean|, substantial mix of + and - subjects")
    print("  H_noise:  SD ≈ |mean| or smaller, consistent direction")
    print()
    print("  INTERPRETATION GUIDE:")
    print("  Hetero ratio >> 2 AND ~50/50 pos/neg  -> heterogeneity confirmed")
    print("  Hetero ratio < 1 AND one-directional   -> noise / uniform small effect")
    print("  Hetero ratio 1-2 AND mixed direction   -> ambiguous, note for future work")

    return results_out

def main():
    banner("11_entropy_heterogeneity.py")
    print("Random-slope test of the heterogeneity hypothesis.")
    print("Does β → 0 at large N because the effect is absent (noise),")
    print("or because individual slopes are heterogeneous and cancel (person/state)?")

    df_primary = extract_primary()
    if len(df_primary) > 0:
        heterogeneity_report(df_primary.copy(), "PRIMARY (ERP CORE P3, N=27)")

    try:
        df_ds = extract_ds006018()
        if len(df_ds) > 0:
            heterogeneity_report(df_ds.copy(), "DS006018 (Isbell et al., N~98)")
    except Exception as e:
        print(f"\nDS006018 extraction failed: {e}")
        print("Run after eegdash cache is populated.")

    out_dir = os.path.join(RESULTS_DIR, 'logs')
    os.makedirs(out_dir, exist_ok=True)
    df_primary['dataset'] = 'primary'
    df_primary.to_csv(os.path.join(out_dir, 'heterogeneity_primary_trials.csv'),
                      index=False)
    print(f"\nTrial-level data saved -> {out_dir}")

if __name__ == '__main__':
    main()
