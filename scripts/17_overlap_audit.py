import os
import argparse
import numpy as np
import pandas as pd

from phase013_engine import clean_pseudo_mask, generate_pseudotrial_samples
from phase013_cache import build_or_load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')

CONFIGS = {'config1': ('config1', 1.0, 100e-6), 'config2': ('config2', 0.5, 100e-6),
           'config3': ('config3', 1.0, 150e-6), 'config4': ('config4', 0.5, 150e-6)}
EARLY = (0.0, 0.150)
P300 = (0.300, 0.600)
EVOKED_END = 0.8

def epoch_overlap_mask(pseudo_samples, stim_samples, fs, tmin=-0.2, tmax=0.8):
    P = np.asarray(pseudo_samples, float)[:, None]
    S = np.asarray(stim_samples, float)[None, :]
    p_lo, p_hi = P + tmin * fs, P + tmax * fs
    s_lo, s_hi = S + tmin * fs, S + tmax * fs
    return ((p_lo < s_hi) & (p_hi > s_lo)).any(axis=1)

def audit_subject(pseudo_samples, stim_samples, fs):
    n = len(pseudo_samples)
    if n == 0:
        return None
    epoch_hit = epoch_overlap_mask(pseudo_samples, stim_samples, fs)
    keep = clean_pseudo_mask(pseudo_samples, stim_samples, fs, EARLY, P300, EVOKED_END)
    window_hit = ~keep
    return dict(n_pseudo=n,
                n_epoch_overlap=int(epoch_hit.sum()),
                n_window_inside_evoked=int(window_hit.sum()),
                n_retained_clean=int(keep.sum()),
                pct_epoch_overlap=100.0 * epoch_hit.mean(),
                pct_window_inside_evoked=100.0 * window_hit.mean(),
                pct_retained_clean=100.0 * keep.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='erp_core', choices=['erp_core', 'ds006018'])
    ap.add_argument('--config', default='config4', choices=list(CONFIGS))
    ap.add_argument('--cache-dir', default=None)
    a = ap.parse_args()

    cname, min_gap, thresh = CONFIGS[a.config]
    cache = build_or_load_cache(a.dataset, thresh, cache_dir=a.cache_dir)

    rows = []
    for subj, rec in cache.items():
        fs = rec['sfreq']
        stim = rec['stim_onsets']
        ps = generate_pseudotrial_samples(rec, n_needed=rec['n_real'],
                                          min_gap_s=min_gap, fs=fs)
        r = audit_subject(ps, stim, fs)
        if r:
            r.update(subject=subj, dataset=a.dataset, config=cname,
                     min_gap_s=min_gap, reject_threshold=thresh)
            rows.append(r)

    df = pd.DataFrame(rows)
    out = os.path.join(RES, f'overlap_audit_{a.dataset}_{cname}.csv')
    df.to_csv(out, index=False)

    tot_p = df['n_pseudo'].sum()
    print(f'wrote {out}   ({len(df)} participants, {tot_p} pseudotrials)')
    print('\n=== pooled diagnostics (compare against Sections 2.8 / 3.2) ===')
    print(f"  epoch overlaps a real stimulus epoch : "
          f"{100.0 * df['n_epoch_overlap'].sum() / tot_p:.1f}%   (manuscript: 27.5%)")
    print(f"  measurement window inside evoked     : "
          f"{100.0 * df['n_window_inside_evoked'].sum() / tot_p:.1f}%   (manuscript: 19.8%)")
    print(f"  retained by evoked-clean filter      : "
          f"{100.0 * df['n_retained_clean'].sum() / tot_p:.1f}%   (manuscript: ~80%)")
    print(f"  mean clean surrogates per placement  : "
          f"{df['n_retained_clean'].mean():.1f}   (manuscript: ~118)")

if __name__ == '__main__':
    main()
