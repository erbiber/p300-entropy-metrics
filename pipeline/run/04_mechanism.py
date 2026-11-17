import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

warnings.filterwarnings('ignore')

WINDOWS = [('rms_base_fz', 'pre-stimulus -150 to 0', -75.0),
           ('rms_early_fz', 'early 0 to 150', 75.0),
           ('rms_early_fz_200', 'early 0 to 200', 100.0),
           ('rms_early_fz_250', 'early 0 to 250', 125.0)]
CONTROLS = [('rms_base_fz', 'pre-stimulus Fz'), ('rms_base_pz', 'pre-stimulus Pz')]
NEEDED = ['sd_early_fz', 'rms_base_pz', 'p300_pz'] + [w[0] for w in WINDOWS]

def robust_z(v):
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad <= 0:
        return v - med
    return (v - med) / mad

def standardise(df, cols):
    for c in cols:
        z = np.full(len(df), np.nan)
        for sub, idx in df.groupby('subject').indices.items():
            z[idx] = robust_z(df[c].values[idx])
        df[c + '_z'] = z
    return df

def fit(df, formula, pred):
    m = smf.mixedlm(formula, df, groups=df['subject']).fit(method='lbfgs')
    return float(m.params[pred]), float(m.bse[pred]), float(m.pvalues[pred])

def within_r(df, a, b):
    vals = []
    for _, g in df.groupby('subject'):
        x, y = g[a].values, g[b].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        vals.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return np.array(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['erpcore', 'ds006018'], required=True)
    ap.add_argument('--features', metavar='CSV')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    tag = cfg.variant_tag()
    cfg.banner(f'step 4  mechanism  |  dataset={a.dataset}  variant={tag}')

    fp = a.features or os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    if not os.path.exists(fp):
        print(f'features file not found: {fp}')
        print(f'run step 1 with the same options first')
        return 2
    df = pd.read_csv(fp)
    missing = [c for c in NEEDED + ['subject', 'condition'] if c not in df.columns]
    if missing:
        print(f'features file lacks required columns: {missing}')
        return 2
    df = standardise(df, NEEDED)
    df = df.dropna(subset=[c + '_z' for c in NEEDED] + ['condition', 'subject']).copy()
    print(f'{len(df)} trials, {df.subject.nunique()} subjects\n')

    rows = []

    r_fz = within_r(df, 'sd_early_fz_z', 'rms_base_fz_z')
    r_pz = within_r(df, 'sd_early_fz_z', 'rms_base_pz_z')
    print('TEST 1  shared trial-level state')
    for lab, r in (('Fz early vs Fz pre-stimulus', r_fz), ('Fz early vs Pz pre-stimulus', r_pz)):
        print(f'  r({lab:30}) mean {r.mean():+.4f}  median {np.median(r):+.4f}  '
              f'positive {int((r > 0).sum())}/{len(r)}')
        rows.append(dict(test='1_shared_state', quantity=lab, value=r.mean(),
                         median=np.median(r), n=len(r), n_positive=int((r > 0).sum()),
                         lo=r.min(), hi=r.max(), note=''))

    print('\nTEST 2  window position')
    betas = []
    for col, lab, mid in WINDOWS:
        b, se, p = fit(df, f'p300_pz_z ~ {col}_z + condition', f'{col}_z')
        betas.append(b)
        print(f'  {lab:26} beta={b:+.4f}  se={se:.4f}  p={p:.3e}')
        rows.append(dict(test='2_window_position', quantity=lab, value=b, median=np.nan,
                         n=len(df), n_positive=np.nan, lo=b - 1.96 * se, hi=b + 1.96 * se,
                         note=f'midpoint_ms={mid:g} p={p:.3e}'))
    pre, post = betas[0], betas[1:]
    same = all(x > 0 for x in betas) or all(x < 0 for x in betas)
    ratio = abs(pre) / np.mean([abs(x) for x in post])
    mono = abs(post[0]) < abs(post[1]) < abs(post[2])
    print(f'  all four same sign: {same}   sign: {"positive" if betas[0] > 0 else "negative"}')
    print(f'  |pre| / mean|post| = {ratio:.3f}')
    print(f'  monotonic toward the P300: {mono}')
    rows.append(dict(test='2_summary', quantity='pre_over_post_ratio', value=ratio,
                     median=np.nan, n=len(df), n_positive=np.nan, lo=np.nan, hi=np.nan,
                     note=f'all_same_sign={same} sign={"pos" if betas[0] > 0 else "neg"} monotonic={mono}'))

    print('\nTEST 3  retention against the parameter-free expectation 1/(1+r)')
    b0, _, p0 = fit(df, 'p300_pz_z ~ sd_early_fz_z + condition', 'sd_early_fz_z')
    print(f'  early alone: beta={b0:+.4f}  p={p0:.3e}')
    for col, lab in CONTROLS:
        bc, _, _ = fit(df, f'p300_pz_z ~ sd_early_fz_z + {col}_z + condition', 'sd_early_fz_z')
        r = float(np.mean(within_r(df, 'sd_early_fz_z', f'{col}_z')))
        ret = bc / b0 if b0 != 0 else np.nan
        pred = 1.0 / (1.0 + r)
        gap = abs(ret - pred) * 100
        print(f'  control={lab:18} r={r:+.4f}  expected {100 * pred:5.1f}%  '
              f'observed {100 * ret:5.1f}%  gap {gap:4.1f} pp')
        rows.append(dict(test='3_retention', quantity=lab, value=ret, median=pred,
                         n=len(df), n_positive=np.nan, lo=np.nan, hi=np.nan,
                         note=f'r={r:+.4f} expected={pred:.4f} gap_pp={gap:.2f}'))

    tgt = df[df.is_target == 1]['p300_pz'].values if 'is_target' in df.columns else None
    if tgt is not None:
        dvals = []
        for sub, g in df.groupby('subject'):
            t = g[g.is_target == 1]['p300_pz'].values
            s = g[g.is_target == 0]['p300_pz'].values
            if len(t) < 5 or len(s) < 5:
                continue
            sd = np.sqrt(((len(t) - 1) * t.var(ddof=1) + (len(s) - 1) * s.var(ddof=1)) /
                         (len(t) + len(s) - 2))
            if sd > 0:
                dvals.append((t.mean() - s.mean()) / sd)
        dvals = np.array(dvals)
        print(f'\nPOSITIVE CONTROL  oddball effect at Pz')
        print(f'  Cohen d mean {dvals.mean():+.4f}  median {np.median(dvals):+.4f}  '
              f'positive {int((dvals > 0).sum())}/{len(dvals)}')
        rows.append(dict(test='0_positive_control', quantity='oddball_cohen_d',
                         value=dvals.mean(), median=np.median(dvals), n=len(dvals),
                         n_positive=int((dvals > 0).sum()), lo=dvals.min(), hi=dvals.max(),
                         note='target minus standard p300_pz'))

    out = pd.DataFrame(rows)
    out.insert(0, 'dataset', a.dataset)
    out.insert(1, 'variant', tag)
    p = os.path.join(cfg.RESULTS_DIR, f'mechanism_{a.dataset}_{tag}.csv')
    out.to_csv(p, index=False)
    print(f'\n{len(out)} rows -> {p}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
