"""
17_overlap_audit.py — pseudotrial/evoked overlap diagnostics.

!! THIS SCRIPT DOES NOT RUN IN ITS CURRENT FORM. Verified 2026-11-04. !!

Three independent faults, each sufficient to stop it:

  1. `from phase013_cache import build_or_load_cache` — no such function exists.
     phase013_cache.py exports only cache_path(), load() and save().
  2. The call `generate_pseudotrial_samples(rec, n_needed=..., min_gap_s=..., fs=...)`
     matches no signature in phase013_engine.py, where the function is
     generate_pseudotrial_samples(n_pseudo, sfreq, n_continuous_samples,
     real_event_samples, min_gap_seconds, tmin, tmax, rng, ...).
  3. It expects per-subject records carrying 'sfreq', 'stim_onsets' and 'n_real'.
     The deposited cache pickles hold 'real_slope', 'pseudo_slopes', 'real_feats'
     and 'n_draws' only, so the required fields are not available without
     re-reading the raw recordings.

Consequently the deposited file results/overlap_audit_erp_core_config4.csv was NOT
produced by this script: its columns are subject, dataset, config, n_draws,
n_pseudo_total, pct_epoch_overlap, pct_window_inside_evoked, pct_retained_clean,
mean_clean_surrogates_per_draw, excluded_cached, whereas this script would write
n_pseudo, n_epoch_overlap, n_window_inside_evoked, n_retained_clean and three pct_
columns. The deposited result therefore has no runnable provenance in this repository.

The hard-coded comparison values printed at the end ("manuscript: 27.5%", "19.8%") are
also stale: the manuscript reports 46.9% of pseudotrials overlapping a neighbouring
evoked epoch and 7.8% with the measurement window inside a real evoked interval.

TO FIX, one of the following is required before submission:
  (a) write the missing cache builder that extracts sfreq, stimulus onsets and the
      retained real-trial count per participant from the raw recordings, correct the
      call signature, and align the output schema with the deposited CSV; or
  (b) deposit the code that actually produced the CSV; or
  (c) withdraw the CSV and the overlap figures that rest on it.

Until then this file is kept for provenance only and must not be presented as the
generator of the deposited audit.
"""

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
