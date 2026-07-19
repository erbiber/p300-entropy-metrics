import os, glob, argparse
import numpy as np, pandas as pd
from phase013_engine import generate_pseudotrial_samples

TARGET_CODES = {11, 22, 33, 44, 55}
FS_DEFAULT = 1024.0; TMIN, TMAX = -0.2, 0.8

def analyse_subject(events_path, min_gap=0.5, K=200, seed=12345):
    ev = pd.read_csv(events_path, sep='\t')
    fs = FS_DEFAULT
    if 'sample' in ev and len(ev) > 1:
        est = ev['sample'].iloc[-1] / ev['onset'].iloc[-1] if ev['onset'].iloc[-1] > 0 else FS_DEFAULT
        if 400 < est < 1200: fs = round(est)
    stim = ev[ev.trial_type == 'stimulus']['sample'].values.astype(int)
    all_samp = ev['sample'].values.astype(int)
    if len(stim) < 10: return None
    n_real = len(stim); n_cont = int(ev['sample'].max()) + int(2 * fs)
    dur = n_cont / fs

    s_sorted = np.sort(stim)
    real_iti = np.diff(s_sorted) / fs
    real_pos = s_sorted / n_cont

    rng = np.random.default_rng(seed)
    p_gap = []; p_pos = []
    for k in range(K):
        ps = generate_pseudotrial_samples(n_real, fs, n_cont, all_samp, min_gap, TMIN, TMAX, rng, 10, 1000)
        if len(ps) == 0: continue
        for x in ps:
            p_gap.append(np.min(np.abs(all_samp - x)) / fs)
        p_pos.extend((ps / n_cont).tolist())
    p_gap = np.array(p_gap); p_pos = np.array(p_pos)
    q = lambda a, x: float(np.quantile(a, x)) if len(a) else np.nan
    return dict(n_real=n_real, fs=fs, dur_s=dur,
                real_iti_median=float(np.median(real_iti)), real_iti_q10=q(real_iti,.1), real_iti_q90=q(real_iti,.9),
                pseudo_gap_median=float(np.median(p_gap)), pseudo_gap_q10=q(p_gap,.1), pseudo_gap_q90=q(p_gap,.9),
                real_pos_mean=float(np.mean(real_pos)), pseudo_pos_mean=float(np.mean(p_pos)),
                real_pos_sd=float(np.std(real_pos)), pseudo_pos_sd=float(np.std(p_pos)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=r'C:\Users\erkan\Documents\dof_validation\data\erp_core_P3')
    ap.add_argument('--min-gap', type=float, default=0.5)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data_root, 'sub-*', 'ses-P3', 'eeg',
                                          'sub-*_ses-P3_task-P3_events.tsv')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(a.data_root, '**', '*_task-P3_events.tsv'), recursive=True))
    print(f"found {len(paths)} events files")
    rows = []
    for p in paths:
        sid = os.path.basename(p).split('_')[0]
        try:
            r = analyse_subject(p, min_gap=a.min_gap)
            if r is None: print(f"  {sid}: too few trials"); continue
            r['subject'] = sid; rows.append(r)
            print(f"  {sid}: real ITI median={r['real_iti_median']:.2f}s  "
                  f"pseudo gap-to-event median={r['pseudo_gap_median']:.2f}s  "
                  f"pos mean real/pseudo={r['real_pos_mean']:.2f}/{r['pseudo_pos_mean']:.2f}")
        except Exception as e:
            print(f"  {sid}: FAILED {e}")
    df = pd.DataFrame(rows)
    op = os.path.join(a.out, 'distributional_comparability.csv'); df.to_csv(op, index=False)
    print(f"\nWrote {op}")
    if len(df):
        print("\n=== AGGREGATE (mean across subjects) ===")
        print(f"  real ITI median        : {df['real_iti_median'].mean():.3f} s "
              f"[q10 {df['real_iti_q10'].mean():.2f}, q90 {df['real_iti_q90'].mean():.2f}]")
        print(f"  pseudo gap-to-event    : {df['pseudo_gap_median'].mean():.3f} s "
              f"[q10 {df['pseudo_gap_q10'].mean():.2f}, q90 {df['pseudo_gap_q90'].mean():.2f}]")
        print(f"  sequence position mean : real {df['real_pos_mean'].mean():.3f} vs pseudo {df['pseudo_pos_mean'].mean():.3f} "
              f"(both near 0.5 = evenly spread across the recording)")

if __name__ == '__main__':
    main()
