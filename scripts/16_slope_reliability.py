
import os
import numpy as np
import pandas as pd
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')

MEASURES = [('pe', 'permutation_entropy'), ('se', 'sample_entropy'),
            ('lz', 'lempel_ziv'), ('hjorth_mob', 'hjorth_mobility'),
            ('hjorth_cplx', 'hjorth_complexity')]
DATASETS = [('heterogeneity_primary_trials.csv', 'primary'),
            ('heterogeneity_ds006018_checkpoint.csv', 'independent')]
MIN_TRIALS = 5


def robust_z(s):
    s = np.asarray(s, float)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med


def ols_slope(x, y):
    if len(x) < 3 or np.std(x) < 1e-10:
        return np.nan
    return np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def prep(path):
    df = pd.read_csv(path)
    for c, _ in MEASURES:
        df[c + '_z'] = df.groupby('subject')[c].transform(robust_z)
    df['p300_z'] = df.groupby('subject')['p300'].transform(robust_z)
    return df


def slopes(df, m, parity=None):
    out = []
    for _, g in df.dropna(subset=[m + '_z', 'p300_z']).groupby('subject'):
        if len(g) < MIN_TRIALS:
            continue
        x, y = g[m + '_z'].values, g['p300_z'].values
        out.append(ols_slope(x, y) if parity is None
                   else ols_slope(x[parity::2], y[parity::2]))
    return np.array(out)


def main():
    rel_rows, con_rows = [], []
    for fname, dsname in DATASETS:
        df = prep(os.path.join(RES, fname))
        for m, label in MEASURES:
            full = slopes(df, m)
            odd, even = slopes(df, m, 0), slopes(df, m, 1)
            raw = corr(odd, even)
            sb = 2 * raw / (1 + raw) if np.isfinite(raw) else np.nan
            rel_rows.append(dict(
                dataset=dsname, measure=label, n_subjects=int(np.isfinite(full).sum()),
                slope_mean=float(np.nanmean(full)), slope_sd=float(np.nanstd(full, ddof=1)),
                odd_even_r=raw, spearman_brown=sb,
                reportable=(sb if raw > 0 else 0.0),
                sb_defined=bool(raw > 0)))
        for (a, la), (b, lb) in combinations(MEASURES, 2):
            con_rows.append(dict(
                dataset=dsname, measure_a=la, measure_b=lb,
                r_same_trials=corr(slopes(df, a), slopes(df, b)),
                r_disjoint_a_odd=corr(slopes(df, a, 0), slopes(df, b, 1)),
                r_disjoint_b_odd=corr(slopes(df, b, 0), slopes(df, a, 1))))
    rel = pd.DataFrame(rel_rows)
    con = pd.DataFrame(con_rows)
    con['r_disjoint_mean'] = con[['r_disjoint_a_odd', 'r_disjoint_b_odd']].mean(axis=1)
    rel.to_csv(os.path.join(RES, 'slope_reliability.csv'), index=False)
    con.to_csv(os.path.join(RES, 'cross_measure_concordance.csv'), index=False)

    print('=== split-half slope reliability (Table 8 reliability column) ===')
    for _, r in rel.iterrows():
        flag = '' if r['sb_defined'] else '   (odd-even r < 0; reported as <= 0)'
        print(f"  {r['dataset']:12s} {r['measure']:20s} odd-even r={r['odd_even_r']:+.3f}  "
              f"SB={r['spearman_brown']:+.3f}  reported={r['reportable']:.2f}{flag}")
    print('\n=== cross-measure concordance (Figure 7C) ===')
    for _, r in con[con.measure_a.eq('sample_entropy') & con.measure_b.eq('hjorth_mobility')].iterrows():
        print(f"  {r['dataset']:12s} same={r['r_same_trials']:+.3f}  "
              f"disjoint dir1={r['r_disjoint_a_odd']:+.3f} dir2={r['r_disjoint_b_odd']:+.3f} "
              f"mean={r['r_disjoint_mean']:+.3f}")


if __name__ == '__main__':
    main()
