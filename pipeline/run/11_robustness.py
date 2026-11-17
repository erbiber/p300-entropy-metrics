import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats
import estimators as E

SAME = ['M8-D', 'M9a-D', 'M10-D', 'M24-D']
LOC = ['M4a-D', 'M18-D', 'M21-D']
PAIRS = {'distributed': ('distributed', 'distributed'), 'cz': ('cz', 'distributed'), 'mastoid': ('mastoid', 'mastoid'), 'average': ('average', 'average'), 'csd': ('csd', 'csd'), 'csd18': ('csd18', 'csd18')}
B = 2000
B_BCA = 4000
SEED = 12345
SESOI = 0.05


def boot_p(d):
    rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, len(d), (B_BCA, len(d)))].mean(axis=1)
    return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()) + 1.0 / B_BCA))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load(results):
    A, subs, models = {}, {}, None
    for ds in ('erpcore', 'ds006018'):
        p = pd.read_csv(os.path.join(results, f'reffree_persubject_{ds}.csv'))
        p = p[p.used]
        subs[ds] = sorted(p.subject.unique())
        for r in p.reference.unique():
            g = p[p.reference == r]
            for q in ('real_slope', 'pseudo_mean', 'd'):
                w = g.pivot_table(index='subject', columns='model_id', values=q, aggfunc='first')
                if models is None:
                    models = sorted(w.columns)
                A[(ds, r, q)] = w.reindex(index=subs[ds], columns=models).values
    return A, subs, models


def equivalence(A, subs, models):
    rows = []
    cross = [m not in SAME for m in models]
    for ds in subs:
        for r in sorted({k[1] for k in A if k[0] == ds}):
            X = A[(ds, r, 'real_slope')]
            block = []
            for j, m in enumerate(models):
                d = X[:, j][~np.isnan(X[:, j])]
                est, lo, hi = E.bca_ci(d, B=B_BCA, alpha=0.05, seed=SEED)
                _, lo9, hi9 = E.bca_ci(d, B=B_BCA, alpha=0.10, seed=SEED)
                block.append(dict(dataset=ds, reference=r, model_id=m, cross=cross[j], n=len(d), mean=est, se=float(d.std(ddof=1) / np.sqrt(len(d))), bca95_lo=lo, bca95_hi=hi, bca90_lo=lo9, bca90_hi=hi9, equivalence_bound=float(max(abs(lo9), abs(hi9))), p_boot=boot_p(d)))
            h21 = holm([b['p_boot'] for b in block])
            idx = [i for i, b in enumerate(block) if b['cross']]
            h17 = holm([block[i]['p_boot'] for i in idx])
            for i, b in enumerate(block):
                b['p_holm21'] = float(h21[i])
                b['p_holm17'] = np.nan
            for k, i in enumerate(idx):
                block[i]['p_holm17'] = float(h17[k])
            rows += block
    return pd.DataFrame(rows)


def summary(eq, A, subs, models):
    rows = []
    cross = np.array([m not in SAME for m in models])
    for (ds, r), g in eq[eq.cross].groupby(['dataset', 'reference']):
        X = A[(ds, r, 'real_slope')][:, cross]
        X = X[~np.isnan(X).any(axis=1)]
        lam = np.clip(np.linalg.eigvalsh(np.corrcoef(X.T)), 0, None)
        rows.append(dict(dataset=ds, reference=r, n=int(g.n.iloc[0]), mean_abs=float(g['mean'].abs().mean()), noise_floor=float((g.se * np.sqrt(2 / np.pi)).mean()), max_abs=float(g['mean'].abs().max()), max_bound90=float(g.equivalence_bound.max()), n_equivalent=int((g.equivalence_bound < SESOI).sum()), n_holm17=int((g.p_holm17 < 0.05).sum()), n_holm21=int((g.p_holm21 < 0.05).sum()), effective_n=float(lam.sum() ** 2 / (lam ** 2).sum())))
    return pd.DataFrame(rows)


def boot_idx(n, rng):
    return [rng.integers(0, n, n) for _ in range(B)]


def pairs_for(A):
    return {p: v for p, v in PAIRS.items() if ('erpcore', v[0], 'real_slope') in A and ('ds006018', v[1], 'real_slope') in A}


def absolute_interaction(A, subs, models):
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(subs['erpcore']), rng)
    idd = boot_idx(len(subs['ds006018']), rng)
    cross = np.array([m not in SAME for m in models])
    P = pairs_for(A)
    def D(p, q, a, b):
        e, d = P[p]
        return np.abs(np.nanmean(A[('erpcore', e, q)][a], 0) - np.nanmean(A[('ds006018', d, q)][b], 0))[cross].mean()
    fe, fd = np.arange(len(subs['erpcore'])), np.arange(len(subs['ds006018']))
    obs = {(p, q): D(p, q, fe, fd) for p in P for q in ('real_slope', 'pseudo_mean')}
    bt = {k: np.array([D(k[0], k[1], a, b) for a, b in zip(ie, idd)]) for k in obs}
    rows = []
    for p in P:
        if p == 'distributed':
            continue
        cr = bt[(p, 'real_slope')] - bt[('distributed', 'real_slope')]
        cp = bt[(p, 'pseudo_mean')] - bt[('distributed', 'pseudo_mean')]
        o_r = obs[(p, 'real_slope')] - obs[('distributed', 'real_slope')]
        o_p = obs[(p, 'pseudo_mean')] - obs[('distributed', 'pseudo_mean')]
        for lab, v, b in (('real_change', o_r, cr), ('pseudo_change', o_p, cp), ('interaction', o_r - o_p, cr - cp)):
            rows.append(dict(pair=p, quantity=lab, value=float(v), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    return pd.DataFrame(rows)


def bca_correlation(A, subs, models):
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(subs['erpcore']), rng)
    idd = boot_idx(len(subs['ds006018']), rng)
    cross = np.array([m not in SAME for m in models])
    P = pairs_for(A)
    def T(p, q, a, b):
        e, d = P[p]
        return np.corrcoef(np.nanmean(A[('erpcore', e, q)][a], 0)[cross], np.nanmean(A[('ds006018', d, q)][b], 0)[cross])[0, 1]
    ne, nd = len(subs['erpcore']), len(subs['ds006018'])
    fe, fd = np.arange(ne), np.arange(nd)
    def stat(p, q, a, b):
        if p.endswith('-distributed'):
            return T(p.split('-')[0], q, a, b) - T('distributed', q, a, b)
        return T(p, q, a, b)
    rows = []
    for q in ('real_slope', 'pseudo_mean', 'd'):
        for p in list(P) + [f'{x}-distributed' for x in P if x != 'distributed']:
            th = stat(p, q, fe, fd)
            bt = np.array([stat(p, q, a, b) for a, b in zip(ie, idd)])
            jk = np.array([stat(p, q, np.delete(fe, i), fd) for i in range(ne)] + [stat(p, q, fe, np.delete(fd, i)) for i in range(nd)])
            z0 = stats.norm.ppf(min(max(np.mean(bt < th), 1e-6), 1 - 1e-6))
            jm = jk.mean()
            den = 6 * np.sum((jm - jk) ** 2) ** 1.5
            acc = np.sum((jm - jk) ** 3) / den if den else 0.0
            adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
            lo, hi = np.percentile(bt, 100 * adj(stats.norm.ppf(0.025))), np.percentile(bt, 100 * adj(stats.norm.ppf(0.975)))
            rows.append(dict(contrast=p, quantity=q, T=float(th), pct_lo=float(np.percentile(bt, 2.5)), pct_hi=float(np.percentile(bt, 97.5)), bca_lo=float(lo), bca_hi=float(hi), z0=float(z0), acceleration=float(acc), boot_max_abs=float(np.abs(bt).max()) if not p.endswith('-distributed') else np.nan))
    return pd.DataFrame(rows)


LOCATION = {'mean': lambda x, axis=None: np.mean(x, axis=axis), 'trimmed10': lambda x, axis=None: stats.trim_mean(x, 0.1, axis=axis), 'median': lambda x, axis=None: np.median(x, axis=axis)}


def bca_stat(d, f, alpha):
    n = len(d)
    rng = np.random.default_rng(SEED)
    theta = f(d)
    boot = np.array([f(d[rng.integers(0, n, n)]) for _ in range(B_BCA)])
    z0 = stats.norm.ppf(min(max(np.mean(boot < theta), 1e-06), 1 - 1e-06))
    jack = np.array([f(np.delete(d, i)) for i in range(n)])
    jm = jack.mean()
    den = 6 * np.sum((jm - jack) ** 2) ** 1.5
    acc = np.sum((jm - jack) ** 3) / den if den != 0 else 0.0
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    return float(theta), float(np.percentile(boot, 100 * adj(stats.norm.ppf(alpha / 2)))), float(np.percentile(boot, 100 * adj(stats.norm.ppf(1 - alpha / 2))))


def test_variants(A, subs, models):
    rows = []
    cross = np.array([m not in SAME for m in models])
    for ds in subs:
        for r in sorted({k[1] for k in A if k[0] == ds}):
            for q in ('real_slope', 'd'):
                X = A[(ds, r, q)]
                pi, pc, est = [], [], []
                for j in range(len(models)):
                    d = X[:, j][~np.isnan(X[:, j])]
                    rng = np.random.default_rng(SEED)
                    m = d[rng.integers(0, len(d), (B_BCA, len(d)))].mean(axis=1)
                    o = d.mean()
                    est.append(o)
                    pi.append(float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()) + 1.0 / B_BCA)))
                    pc.append(float((1 + (np.abs(m - o) >= abs(o)).sum()) / (B_BCA + 1)))
                pi, pc = np.array(pi), np.array(pc)
                h21, c21 = holm(pi), holm(pc)
                h17, c17 = np.full(len(models), np.nan), np.full(len(models), np.nan)
                h17[cross], c17[cross] = holm(pi[cross]), holm(pc[cross])
                for j, m in enumerate(models):
                    rows.append(dict(dataset=ds, reference=r, quantity=q, model_id=m, cross=bool(cross[j]), mean=float(est[j]), p_inversion=pi[j], p_centred=pc[j], holm21_inversion=float(h21[j]), holm21_centred=float(c21[j]), holm17_inversion=float(h17[j]), holm17_centred=float(c17[j])))
    return pd.DataFrame(rows)


def distribution(A, subs, models):
    rows = []
    for ds in subs:
        for r in sorted({k[1] for k in A if k[0] == ds}):
            for q in ('real_slope', 'd'):
                X = A[(ds, r, q)]
                for j, m in enumerate(models):
                    if m in SAME:
                        continue
                    d = X[:, j][~np.isnan(X[:, j])]
                    row = dict(dataset=ds, reference=r, quantity=q, model_id=m, n=len(d), skewness=float(stats.skew(d)), excess_kurtosis=float(stats.kurtosis(d)), shapiro_p=float(stats.shapiro(d).pvalue))
                    row['heavy'] = bool(abs(row['skewness']) > 1 or row['excess_kurtosis'] > 3)
                    if row['heavy']:
                        for nm in ('mean', 'trimmed10'):
                            est, lo, hi = bca_stat(d, LOCATION[nm], 0.05)
                            _, lo9, hi9 = bca_stat(d, LOCATION[nm], 0.10)
                            row.update({f'{nm}': est, f'{nm}_lo': lo, f'{nm}_hi': hi, f'{nm}_excludes0': bool(lo > 0 or hi < 0), f'{nm}_equivalent': bool(max(abs(lo9), abs(hi9)) < SESOI)})
                    rows.append(row)
    return pd.DataFrame(rows)


def location_agreement(A, subs, models):
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(subs['erpcore']), rng)
    idd = boot_idx(len(subs['ds006018']), rng)
    cross = np.array([m not in SAME for m in models])
    P = pairs_for(A)
    rows = []
    for nm, f in LOCATION.items():
        def T(p, q, a, b):
            e, d = P[p]
            x, y = A[('erpcore', e, q)][a][:, cross], A[('ds006018', d, q)][b][:, cross]
            px = np.array([f(c[~np.isnan(c)]) for c in x.T])
            py = np.array([f(c[~np.isnan(c)]) for c in y.T])
            return np.corrcoef(px, py)[0, 1]
        fe, fd = np.arange(len(subs['erpcore'])), np.arange(len(subs['ds006018']))
        for q in ('real_slope', 'pseudo_mean', 'd'):
            for p in P:
                bt = np.array([T(p, q, a, b) for a, b in zip(ie, idd)])
                rows.append(dict(estimator=nm, contrast=p, quantity=q, T=float(T(p, q, fe, fd)), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5))))
    return pd.DataFrame(rows)


FEAT = {'M1-D': 'rms_early_fz', 'M3-D': 'sd_early_fz', 'M4a-D': 'mean_early_fz', 'M5-D': 'rms_early_fz_200', 'M6-D': 'rms_early_fz_250', 'M7-D': 'rms_base_fz', 'M13-D': 'pe_early_fz', 'M14-D': 'se_early_fz', 'M15-D': 'lz_early_fz', 'M16-D': 'skew_amp_fz', 'M17-D': 'kurt_amp_fz', 'M18-D': 'median_amp_fz', 'M19-D': 'mad_amp_fz', 'M20-D': 'p2p_amp_fz', 'M21-D': 'slope_amp_fz', 'M22-D': 'hjmob_early_fz', 'M23-D': 'hjcplx_early_fz'}
REFTAG = {'distributed': 'r150_g05_ic20', 'mastoid': 'r150_g05_mastoid_ic20', 'average': 'r150_g05_average_ic20'}


def robust_z(v):
    v = np.asarray(v, float)
    med = np.median(v)
    s = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / s if s > 0 else v - med


def neighbour_residual(v, k=5):
    n = len(v)
    out = np.empty(n)
    for i in range(n):
        idx = [j for j in range(max(0, i - k), min(n, i + k + 1)) if j != i]
        out[i] = v[i] - v[idx].mean()
    return out


def neighbour_slopes(path, adjust):
    d = pd.read_csv(path).sort_values(['subject', 'trial_idx'])
    rows = {}
    for s, g in d.groupby('subject'):
        y = robust_z(g.p300_pz.values)
        if adjust:
            y = neighbour_residual(y)
        r = {}
        for m, c in FEAT.items():
            x = robust_z(g[c].values)
            if adjust:
                x = neighbour_residual(x)
            r[m] = E.ols_slope(x, y)
        rows[s] = r
    return pd.DataFrame(rows).T


def neighbour_adjustment(results):
    paths = {(ds, r): os.path.join(results, f'features_{ds}_{tag}.csv') for ds in ('erpcore', 'ds006018') for r, tag in REFTAG.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None
    S = {(k, a): neighbour_slopes(p, a) for k, p in paths.items() for a in (False, True)}
    common = {ds: sorted(set.intersection(*[set(S[((ds, r), False)].index) for r in REFTAG])) for ds in ('erpcore', 'ds006018')}
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(common['erpcore']), rng)
    idd = boot_idx(len(common['ds006018']), rng)
    rows, share = [], []
    for a in (False, True):
        X = {r: S[(('erpcore', r), a)].loc[common['erpcore']].values for r in REFTAG}
        Y = {r: S[(('ds006018', r), a)].loc[common['ds006018']].values for r in REFTAG}
        T = lambda r, i, j: np.corrcoef(X[r][i].mean(0), Y[r][j].mean(0))[0, 1]
        fe, fd = np.arange(len(common['erpcore'])), np.arange(len(common['ds006018']))
        bt = {r: np.array([T(r, i, j) for i, j in zip(ie, idd)]) for r in REFTAG}
        for r in REFTAG:
            rows.append(dict(adjusted=a, contrast=r, value=T(r, fe, fd), lo=np.percentile(bt[r], 2.5), hi=np.percentile(bt[r], 97.5)))
        for r in ('mastoid', 'average'):
            dd = bt[r] - bt['distributed']
            rows.append(dict(adjusted=a, contrast=f'{r}-distributed', value=T(r, fe, fd) - T('distributed', fe, fd), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
    for ds in ('erpcore', 'ds006018'):
        for r in REFTAG:
            raw = S[((ds, r), False)].loc[common[ds]].mean().abs().mean()
            adj = S[((ds, r), True)].loc[common[ds]].mean().abs().mean()
            share.append(dict(dataset=ds, reference=r, n=len(common[ds]), retained_share=adj / raw))
    return pd.DataFrame(rows), pd.DataFrame(share)


def main():
    ap = argparse.ArgumentParser(description='Step 11: equivalence and sampling-noise floor of real-trial slopes under every derivation, effective number of statistics, cross-dataset interaction on the absolute scale, BCa intervals for profile agreement, Holm adjustment over 21 and 17 models with interval-inversion and null-centred bootstrap p values, distribution of participant-level slopes with robust location estimates, and the neighbouring-trial adjustment on participants common to all references.')
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--dataset', default='all')
    a = ap.parse_args()
    a.out = a.out or a.results
    os.makedirs(a.out, exist_ok=True)
    A, subs, models = load(a.results)
    eq = equivalence(A, subs, models)
    eq.to_csv(os.path.join(a.out, 'robustness_equivalence.csv'), index=False)
    s = summary(eq, A, subs, models)
    s.to_csv(os.path.join(a.out, 'robustness_summary.csv'), index=False)
    absolute_interaction(A, subs, models).to_csv(os.path.join(a.out, 'robustness_absolute_interaction.csv'), index=False)
    bca_correlation(A, subs, models).to_csv(os.path.join(a.out, 'robustness_bca_agreement.csv'), index=False)
    test_variants(A, subs, models).to_csv(os.path.join(a.out, 'robustness_tests.csv'), index=False)
    distribution(A, subs, models).to_csv(os.path.join(a.out, 'robustness_distribution.csv'), index=False)
    location_agreement(A, subs, models).to_csv(os.path.join(a.out, 'robustness_location_agreement.csv'), index=False)
    na, ns = neighbour_adjustment(a.results)
    if na is not None:
        na.to_csv(os.path.join(a.out, 'robustness_neighbour_adjustment.csv'), index=False)
        ns.to_csv(os.path.join(a.out, 'robustness_neighbour_share.csv'), index=False)
    for _, r in s.iterrows():
        print(f"{r.dataset:9s} {r.reference:12s} mean|b| {r.mean_abs:.4f} floor {r.noise_floor:.4f} equivalent {r.n_equivalent}/17 Holm17 {r.n_holm17} Holm21 {r.n_holm21} effective n {r.effective_n:.1f}")
    print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
