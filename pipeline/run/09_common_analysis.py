import argparse
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats
import estimators as E

SAME = ['M8-D', 'M9a-D', 'M10-D', 'M24-D']
FAM = {'location': ['M4a-D', 'M18-D', 'M21-D'], 'dispersion': ['M1-D', 'M3-D', 'M5-D', 'M6-D', 'M7-D', 'M19-D', 'M20-D'], 'shape': ['M16-D', 'M17-D'], 'complexity': ['M13-D', 'M14-D', 'M15-D', 'M22-D', 'M23-D'], 'rms_repeats': ['M5-D', 'M6-D', 'M7-D']}
PAIRS = {'distributed': ('distributed', 'distributed'), 'cz': ('cz', 'distributed'), 'mastoid': ('mastoid', 'mastoid'), 'average': ('average', 'average')}
QS = ['real_slope', 'pseudo_mean', 'd', 'real_slope_cadj', 'pseudo_mean_cadj', 'd_cadj']
B = 2000
SEED = 12345
B_BCA = 4000
SESOI = 0.05
ZC = 0.9999999
THRESHOLDS = (30, 35)


def fz(r):
    return np.arctanh(np.clip(r, -ZC, ZC))


def load(results, mode, min_targets=None):
    A = {}
    subs = {}
    models = None
    for ds in ('erpcore', 'ds006018'):
        p = pd.read_csv(os.path.join(results, f'common_persubject_{ds}_{mode}.csv'))
        p = p[p.used]
        if min_targets:
            c = pd.read_csv(os.path.join(results, f'common_subjects_{ds}.csv'))
            keep = set(c.subject[(c.status == 'ok') & (c.n_target >= min_targets)])
            p = p[p.subject.isin(keep)]
        subs[ds] = sorted(p.subject.unique())
        for r in p.reference.unique():
            g = p[p.reference == r]
            for q in QS:
                w = g.pivot_table(index='subject', columns='model_id', values=q, aggfunc='first')
                if models is None:
                    models = list(w.columns)
                A[(ds, r, q)] = w.loc[subs[ds], models].values
    return A, subs, models


def boot_idx(n, rng):
    return [rng.integers(0, n, n) for _ in range(B)]


def transport(A, subs, models, sel, qs):
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(subs['erpcore']), rng)
    idd = boot_idx(len(subs['ds006018']), rng)
    def T(p, q, a=None, b=None):
        e, d = PAIRS[p]
        x = A[('erpcore', e, q)] if a is None else A[('erpcore', e, q)][a]
        y = A[('ds006018', d, q)] if b is None else A[('ds006018', d, q)][b]
        return np.corrcoef(np.nanmean(x, 0)[sel], np.nanmean(y, 0)[sel])[0, 1]
    obs = {(p, q): T(p, q) for p in PAIRS for q in qs}
    bt = {k: np.array([T(k[0], k[1], a, b) for a, b in zip(ie, idd)]) for k in obs}
    rows = []
    for (p, q), v in obs.items():
        e, d = PAIRS[p]
        sp = stats.spearmanr(np.nanmean(A[('erpcore', e, q)], 0)[sel], np.nanmean(A[('ds006018', d, q)], 0)[sel])[0]
        z = np.arctanh(np.clip(bt[(p, q)], -ZC, ZC))
        rows.append(dict(contrast=p, quantity=q, T=v, lo=np.percentile(bt[(p, q)], 2.5), hi=np.percentile(bt[(p, q)], 97.5), spearman=sp, zT=float(np.arctanh(np.clip(v, -ZC, ZC))), zlo=np.percentile(z, 2.5), zhi=np.percentile(z, 97.5)))
    for p in ('cz', 'mastoid', 'average'):
        for q in qs:
            d = bt[(p, q)] - bt[('distributed', q)]
            zd = fz(bt[(p, q)]) - fz(bt[('distributed', q)])
            rows.append(dict(contrast=f'{p}-distributed', quantity=q, T=obs[(p, q)] - obs[('distributed', q)], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_le0=float(np.mean(d <= 0)), zT=float(fz(obs[(p, q)]) - fz(obs[('distributed', q)])), zlo=np.percentile(zd, 2.5), zhi=np.percentile(zd, 97.5)))
        for real, pseudo, lab in (('real_slope', 'pseudo_mean', 'interaction'), ('real_slope_cadj', 'pseudo_mean_cadj', 'interaction_cadj')):
            if real in qs and pseudo in qs:
                d = (bt[(p, real)] - bt[('distributed', real)]) - (bt[(p, pseudo)] - bt[('distributed', pseudo)])
                o = (obs[(p, real)] - obs[('distributed', real)]) - (obs[(p, pseudo)] - obs[('distributed', pseudo)])
                zd = (fz(bt[(p, real)]) - fz(bt[('distributed', real)])) - (fz(bt[(p, pseudo)]) - fz(bt[('distributed', pseudo)]))
                zo = (fz(obs[(p, real)]) - fz(obs[('distributed', real)])) - (fz(obs[(p, pseudo)]) - fz(obs[('distributed', pseudo)]))
                rows.append(dict(contrast=f'{p}-distributed', quantity=lab, T=o, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), zT=float(zo), zlo=np.percentile(zd, 2.5), zhi=np.percentile(zd, 97.5)))
    return pd.DataFrame(rows)


def absdis(A, subs, models, qs):
    rng = np.random.default_rng(SEED)
    ie = boot_idx(len(subs['erpcore']), rng)
    idd = boot_idx(len(subs['ds006018']), rng)
    loc = np.array([m in FAM['location'] for m in models])
    cross = np.array([m not in SAME for m in models])
    sets = {'cross17': cross, 'location3': cross & loc, 'other14': cross & ~loc}
    def D(p, q, sel, a, b):
        e, d = PAIRS[p]
        return np.abs(np.nanmean(A[('erpcore', e, q)][a], 0) - np.nanmean(A[('ds006018', d, q)][b], 0))[sel].mean()
    fe, fd = np.arange(len(subs['erpcore'])), np.arange(len(subs['ds006018']))
    rows = []
    for q in qs:
        for name, sel in sets.items():
            bt = {p: np.array([D(p, q, sel, a, b) for a, b in zip(ie, idd)]) for p in PAIRS}
            for p in PAIRS:
                rows.append(dict(quantity=q, set=name, pair=p, value=D(p, q, sel, fe, fd), lo=np.percentile(bt[p], 2.5), hi=np.percentile(bt[p], 97.5)))
            for p in ('cz', 'mastoid', 'average'):
                d = bt[p] - bt['distributed']
                rows.append(dict(quantity=q, set=name, pair=f'{p}-distributed', value=D(p, q, sel, fe, fd) - D('distributed', q, sel, fe, fd), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def location_change(A, subs, models, qs):
    loc = np.array([m in FAM['location'] for m in models])
    cross = np.array([m not in SAME for m in models])
    rows = []
    for ds in ('erpcore', 'ds006018'):
        refs = ('distributed', 'cz', 'mastoid', 'average') if ds == 'erpcore' else ('distributed', 'mastoid', 'average')
        rng = np.random.default_rng(SEED)
        idx = boot_idx(len(subs[ds]), rng)
        for q in qs:
            for i, a in enumerate(refs):
                for b in refs[i + 1:]:
                    def st(ix):
                        ch = np.abs(np.nanmean(A[(ds, b, q)][ix], 0) - np.nanmean(A[(ds, a, q)][ix], 0))
                        return ch[cross & loc].mean(), ch[cross & ~loc].mean()
                    o = st(np.arange(len(subs[ds])))
                    bt = np.array([st(ix) for ix in idx])
                    d = bt[:, 0] - bt[:, 1]
                    rows.append(dict(dataset=ds, quantity=q, ref_a=a, ref_b=b, n=len(subs[ds]), location=o[0], other=o[1], ratio=o[0] / o[1], diff=o[0] - o[1], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def robustness(A, subs, models):
    cross = np.array([m not in SAME for m in models])
    sets = {'all17': cross}
    for f, ms in FAM.items():
        sets[f'without_{f}'] = cross & np.array([m not in ms for m in models])
    sets['without_location_and_rms_repeats'] = cross & np.array([m not in FAM['location'] + FAM['rms_repeats'] for m in models])
    rows = []
    for name, sel in sets.items():
        t = transport(A, subs, models, sel, ['real_slope', 'pseudo_mean'])
        t['set'] = name
        t['n_models'] = int(sel.sum())
        rows.append(t)
    loo = []
    for k, m in enumerate(models):
        if m in SAME:
            continue
        sel = cross.copy()
        sel[k] = False
        for q in ('real_slope', 'pseudo_mean'):
            r = {}
            for p in PAIRS:
                e, d = PAIRS[p]
                r[p] = np.corrcoef(np.nanmean(A[('erpcore', e, q)], 0)[sel], np.nanmean(A[('ds006018', d, q)], 0)[sel])[0, 1]
            loo.append(dict(dropped=m, quantity=q, **r))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(loo)


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


def dbeta(results, mode, measures):
    rows = []
    for ds in ('erpcore', 'ds006018'):
        p = pd.read_csv(os.path.join(results, f'common_persubject_{ds}_{mode}.csv'))
        p = p[p.used]
        for r in p.reference.unique():
            for col in ('d', 'd_cadj'):
                block = []
                for m, g in p[p.reference == r].groupby('model_id'):
                    d = g[col].dropna().values
                    est, lo, hi = E.bca_ci(d, B=B_BCA, alpha=0.05, seed=SEED)
                    _, lo9, hi9 = E.bca_ci(d, B=B_BCA, alpha=0.10, seed=SEED)
                    block.append(dict(dataset=ds, reference=r, contrast=col, model_id=m, measure=measures.get(m, ''), n=len(d), beta_real=float(g['real_slope' if col == 'd' else 'real_slope_cadj'].mean()), beta_pseudo=float(g['pseudo_mean' if col == 'd' else 'pseudo_mean_cadj'].mean()), dbeta=est, bca95_lo=lo, bca95_hi=hi, bca90_lo=lo9, bca90_hi=hi9, equivalence_bound=float(max(abs(lo9), abs(hi9))), excludes_zero=bool(lo > 0 or hi < 0), p_boot=boot_p(d)))
                adj = holm([b['p_boot'] for b in block])
                for b, h in zip(block, adj):
                    b['p_holm'] = float(h)
                    b['equivalent_at_sesoi'] = bool(b['equivalence_bound'] < SESOI)
                rows += block
    return pd.DataFrame(rows)


def condition(results):
    rows = []
    for ds in ('erpcore', 'ds006018'):
        c = pd.read_csv(os.path.join(results, f'common_condition_{ds}.csv'))
        for (r, pr), g in c.groupby(['reference', 'predictor']):
            x = g.z_target_minus_standard.dropna().values
            t = stats.ttest_1samp(x, 0.0)
            rows.append(dict(dataset=ds, reference=r, predictor=pr, n=len(x), mean_z_diff=float(x.mean()), sd=float(x.std(ddof=1)), dz=float(x.mean() / x.std(ddof=1)), p=float(t.pvalue)))
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for (ds, r), g in out.groupby(['dataset', 'reference']):
        out.loc[g.index, 'p_holm'] = holm(g.p.values)
    return out


def main():
    ap = argparse.ArgumentParser(description='Step 9: transportability, absolute disagreement, location statistics, robustness, matched contrasts and condition differences from the step-8 common-cleaning results.')
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--measures', default=None)
    ap.add_argument('--dataset', default='all')
    a = ap.parse_args()
    a.out = a.out or a.results
    os.makedirs(a.out, exist_ok=True)
    measures = {}
    src = [a.measures] if a.measures else sorted(glob.glob(os.path.join(a.results, 'matched_erpcore_*.csv')))
    for f in src:
        if f and os.path.exists(f):
            m = pd.read_csv(f)
            if {'model_id', 'measure'} <= set(m.columns):
                measures = dict(zip(m.model_id, m.measure))
                break
    for mode in ('free', 'onset'):
        A, subs, models = load(a.results, mode)
        cross = np.array([m not in SAME for m in models])
        print(f'{mode}: ERP CORE n = {len(subs["erpcore"])}, ds006018 n = {len(subs["ds006018"])}')
        transport(A, subs, models, cross, QS).to_csv(os.path.join(a.out, f'common_transport_{mode}.csv'), index=False)
        absdis(A, subs, models, ['real_slope', 'pseudo_mean', 'real_slope_cadj']).to_csv(os.path.join(a.out, f'common_absdis_{mode}.csv'), index=False)
        location_change(A, subs, models, ['real_slope', 'pseudo_mean']).to_csv(os.path.join(a.out, f'common_location_{mode}.csv'), index=False)
        rb, loo = robustness(A, subs, models)
        rb.to_csv(os.path.join(a.out, f'common_robust_{mode}.csv'), index=False)
        loo.to_csv(os.path.join(a.out, f'common_loo_{mode}.csv'), index=False)
        th = []
        for t in THRESHOLDS:
            At, st, mt = load(a.results, mode, t)
            ct = np.array([m not in SAME for m in mt])
            x = transport(At, st, mt, ct, ['real_slope', 'pseudo_mean', 'd'])
            x.insert(0, 'n_ds006018', len(st['ds006018']))
            x.insert(0, 'n_erpcore', len(st['erpcore']))
            x.insert(0, 'min_targets', t)
            th.append(x)
        pd.concat(th).to_csv(os.path.join(a.out, f'common_threshold_{mode}.csv'), index=False)
        dbeta(a.results, mode, measures).to_csv(os.path.join(a.out, f'common_dbeta_{mode}.csv'), index=False)
        prof = []
        for (ds, r, q), v in A.items():
            for k, m in enumerate(models):
                prof.append(dict(dataset=ds, reference=r, quantity=q, model_id=m, mean=float(np.nanmean(v[:, k]))))
        pd.DataFrame(prof).to_csv(os.path.join(a.out, f'common_profiles_{mode}.csv'), index=False)
    condition(a.results).to_csv(os.path.join(a.out, 'common_condition_summary.csv'), index=False)
    print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
