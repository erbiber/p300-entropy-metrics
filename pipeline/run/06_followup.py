import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats
import config as cfg
import estimators as E

SEED = 12345
B = 4000
SESOI = 0.05
REFS = {'distributed': 'r150_g05_ic20', 'mastoid': 'r150_g05_mastoid_ic20', 'average': 'r150_g05_average_ic20'}
VALID = ('mastoid', 'average')
DATASETS = ('erpcore', 'ds006018')
COMPLEXITY = ('M13-D', 'M14-D', 'M15-D', 'M22-D', 'M23-D')
ODD_EARLY = ('P1-D', 'P2-D', 'P3-D', 'P4-D')
EVEN_EARLY = ('E1-D', 'E2-D', 'E3-D')
ODD_BASE = ('B1-D', 'B2-D', 'B3-D', 'B4-D')
BINS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 30.0), (30.0, 90.0)]
LAG_FEATURES = ('rms_base_fz', 'sd_early_fz', 'pe_early_fz', 'lz_early_fz')


def bca(d, level):
    _, lo, hi = E.bca_ci(d, B=B, alpha=1.0 - level, seed=SEED)
    return (float(lo), float(hi))


def boot_p(d):
    rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()) + 1.0 / B))


def holm(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def path(results, kind, ds, tag):
    return os.path.join(results, f'{kind}_{ds}_{tag}.csv')


def equivalence(results, notes):
    rows = []
    for fam, kind in (('battery', 'matched'), ('parity', 'matched_parity')):
        for ref, tag in REFS.items():
            if fam == 'parity' and ref not in VALID:
                continue
            for ds in DATASETS:
                ps = path(results, f'{kind}_persubject', ds, tag)
                pm = path(results, kind, ds, tag)
                if not (os.path.exists(ps) and os.path.exists(pm)):
                    notes.append(f'missing {fam} {ds} {ref}: {os.path.basename(pm)}')
                    continue
                P = pd.read_csv(ps)
                M = pd.read_csv(pm).set_index('model_id')
                block = []
                for mid in M.index:
                    d = P[(P.model_id == mid) & P.used].d.values.astype(float)
                    d = d[np.isfinite(d)]
                    if len(d) < 3:
                        continue
                    lo95, hi95 = bca(d, 0.95)
                    lo90, hi90 = bca(d, 0.90)
                    bound = max(abs(lo90), abs(hi90))
                    block.append(dict(family=fam, reference=ref, dataset=ds, model_id=mid, measure=M.loc[mid, 'measure'], n=len(d), dbeta=float(d.mean()), bca95_lo=lo95, bca95_hi=hi95, file_bca95_lo=float(M.loc[mid, 'bca_lo']), file_bca95_hi=float(M.loc[mid, 'bca_hi']), check_max_abs_diff=float(max(abs(lo95 - M.loc[mid, 'bca_lo']), abs(hi95 - M.loc[mid, 'bca_hi']), abs(d.mean() - M.loc[mid, 'dbeta']))), excludes_zero=bool(lo95 > 0 or hi95 < 0), bca90_lo=lo90, bca90_hi=hi90, equivalence_bound=bound, equivalent_at_sesoi=bool(bound < SESOI), p_boot=boot_p(d)))
                if block:
                    adj = holm([b['p_boot'] for b in block])
                    for b, h in zip(block, adj):
                        b['p_holm'] = float(h)
                    rows += block
    return pd.DataFrame(rows)


def agreement(eq):
    rows = []
    for (fam, ref), g in eq.groupby(['family', 'reference']):
        a = g[g.dataset == 'erpcore'].set_index('model_id')
        b = g[g.dataset == 'ds006018'].set_index('model_id')
        ids = a.index.intersection(b.index)
        if len(ids) == 0:
            continue
        same = int((np.sign(a.loc[ids, 'dbeta']) == np.sign(b.loc[ids, 'dbeta'])).sum())
        both = [i for i in ids if a.loc[i, 'excludes_zero'] and b.loc[i, 'excludes_zero']]
        conc = [i for i in both if np.sign(a.loc[i, 'dbeta']) == np.sign(b.loc[i, 'dbeta'])]
        rows.append(dict(family=fam, reference=ref, n_models=len(ids), sign_agree=same, binom_p=float(stats.binomtest(same, len(ids), 0.5).pvalue), both_exclude_zero=';'.join(both), concordant=';'.join(conc)))
    return pd.DataFrame(rows)


def parity_decision(eq):
    lines = []
    p = eq[eq.family == 'parity']
    if p.empty:
        return ['parity results not found; run steps 5 and 3 --family parity first']
    lines.append('pre-stated rule: odd early-window Fz (P1-P4) must exclude zero with the same sign in both datasets on both valid references; odd baseline (B1-B4) and even partners (E1-E3) must include zero')
    verdict = True
    complete = True
    for ref in VALID:
        g = p[p.reference == ref]
        if g.dataset.nunique() < 2:
            lines.append(f'{ref}: incomplete (both datasets needed)')
            complete = False
            continue
        a = g[g.dataset == 'erpcore'].set_index('model_id')
        b = g[g.dataset == 'ds006018'].set_index('model_id')
        hits = [m for m in ODD_EARLY if m in a.index and m in b.index and a.loc[m, 'excludes_zero'] and b.loc[m, 'excludes_zero'] and np.sign(a.loc[m, 'dbeta']) == np.sign(b.loc[m, 'dbeta'])]
        base = [m for m in ODD_BASE if (m in a.index and a.loc[m, 'excludes_zero']) or (m in b.index and b.loc[m, 'excludes_zero'])]
        even = [m for m in EVEN_EARLY if (m in a.index and a.loc[m, 'excludes_zero']) or (m in b.index and b.loc[m, 'excludes_zero'])]
        lines.append(f'{ref}: concordant odd early hits {hits or "none"}; baseline controls excluding zero {base or "none"}; even partners excluding zero {even or "none"}')
        for m in ODD_EARLY + EVEN_EARLY + ODD_BASE:
            if m in a.index and m in b.index:
                lines.append(f'    {m:5s} {a.loc[m, "measure"][:42]:42s} ERP CORE {a.loc[m, "dbeta"]:+.4f} [{a.loc[m, "bca95_lo"]:+.4f},{a.loc[m, "bca95_hi"]:+.4f}]  ds006018 {b.loc[m, "dbeta"]:+.4f} [{b.loc[m, "bca95_lo"]:+.4f},{b.loc[m, "bca95_hi"]:+.4f}]')
        if not hits or base:
            verdict = False
    if not complete:
        lines.append('VERDICT: INCOMPLETE (parity results missing for at least one dataset x reference cell); no decision')
    else:
        lines.append('VERDICT: ' + ('prediction P2 SUPPORTED' if verdict else 'prediction P2 NOT supported (a clean negative result under the stated rule)'))
    return lines


def reference_agreement(results, notes, nboot=2000):
    D = {}
    for ref, tag in REFS.items():
        for ds in DATASETS:
            ps = path(results, 'matched_persubject', ds, tag)
            if not os.path.exists(ps):
                notes.append(f'reference agreement bootstrap skipped: missing {os.path.basename(ps)}')
                return pd.DataFrame()
            P = pd.read_csv(ps)
            P = P[P.used]
            D[(ref, ds)] = P.pivot_table(index='subject', columns='model_id', values='d', aggfunc='first')
    models = sorted(set.intersection(*[set(v.columns) for v in D.values()]))
    rng = np.random.default_rng(SEED)
    subs = {ds: sorted(set().union(*[set(D[(r, ds)].index) for r in REFS])) for ds in DATASETS}
    def stat(idx):
        out = {}
        for ref in REFS:
            v = []
            for ds in DATASETS:
                M = D[(ref, ds)].reindex(idx[ds])[models]
                v.append(M.mean(axis=0, skipna=True).values)
            a, b = v
            ok = np.isfinite(a) & np.isfinite(b)
            out[ref] = (float(np.corrcoef(a[ok], b[ok])[0, 1]), int((np.sign(a[ok]) == np.sign(b[ok])).sum()))
        return out
    obs = stat({ds: subs[ds] for ds in DATASETS})
    boots = []
    for _ in range(nboot):
        idx = {ds: list(rng.choice(subs[ds], len(subs[ds]), replace=True)) for ds in DATASETS}
        boots.append(stat(idx))
    rows = []
    for ref in REFS:
        r = np.array([b[ref][0] for b in boots])
        k = np.array([b[ref][1] for b in boots])
        rows.append(dict(contrast=ref, statistic='r(dbeta ERP CORE, dbeta ds006018)', observed=obs[ref][0], lo95=float(np.percentile(r, 2.5)), hi95=float(np.percentile(r, 97.5)), p_le_0=float((r <= 0).mean())))
        rows.append(dict(contrast=ref, statistic='sign agreement count', observed=obs[ref][1], lo95=float(np.percentile(k, 2.5)), hi95=float(np.percentile(k, 97.5)), p_le_0=np.nan))
    for ref in VALID:
        dr = np.array([b[ref][0] - b['distributed'][0] for b in boots])
        rows.append(dict(contrast=f'{ref} minus distributed', statistic='difference in r', observed=obs[ref][0] - obs['distributed'][0], lo95=float(np.percentile(dr, 2.5)), hi95=float(np.percentile(dr, 97.5)), p_le_0=float((dr <= 0).mean())))
    out = pd.DataFrame(rows)
    out['n_models'] = len(models)
    out['n_boot'] = nboot
    return out


def rz(v):
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    m = np.nanmedian(v)
    s = 1.4826 * np.nanmedian(np.abs(v - m))
    return (v - m) / s if s > 0 else v - m


def lags(results, notes):
    rows = []
    soa = []
    for ref in VALID:
        tag = REFS[ref]
        for ds in DATASETS:
            pf = path(results, 'features', ds, tag)
            pp = path(results, 'features_parity', ds, tag)
            if not (os.path.exists(pf) and os.path.exists(pp)):
                notes.append(f'lag analysis skipped for {ds} {ref}: needs {os.path.basename(pf)} and {os.path.basename(pp)}')
                continue
            F = pd.read_csv(pf, usecols=['subject', 'trial_idx'] + list(LAG_FEATURES))
            O = pd.read_csv(pp, usecols=['subject', 'trial_idx', 'onset_s'])
            X = F.merge(O, on=['subject', 'trial_idx'], how='inner')
            per = {(k, b): [] for k in LAG_FEATURES for b in BINS}
            for sub, g in X.groupby('subject'):
                g = g.sort_values('onset_s')
                t = g.onset_s.values
                dt = np.diff(t)
                if len(dt):
                    soa.append(dict(reference=ref, dataset=ds, subject=sub, min_gap_s=float(dt.min()), median_gap_s=float(np.median(dt))))
                i, j = np.triu_indices(len(t), 1)
                sep = t[j] - t[i]
                for k in LAG_FEATURES:
                    z = rz(g[k].values)
                    for b in BINS:
                        sel = (sep >= b[0]) & (sep < b[1])
                        a_, c_ = z[i[sel]], z[j[sel]]
                        ok = np.isfinite(a_) & np.isfinite(c_)
                        if ok.sum() >= 10 and np.std(a_[ok]) > 0 and np.std(c_[ok]) > 0:
                            per[(k, b)].append(np.corrcoef(a_[ok], c_[ok])[0, 1])
            for (k, b), v in per.items():
                v = np.asarray(v)
                if len(v) < 3:
                    continue
                pos = int((v > 0).sum())
                rows.append(dict(reference=ref, dataset=ds, feature=k, bin_lo_s=b[0], bin_hi_s=b[1], n_subjects=len(v), mean_r=float(v.mean()), se_r=float(v.std(ddof=1) / np.sqrt(len(v))), n_positive=pos, sign_p=float(stats.binomtest(pos, len(v), 0.5).pvalue), t_p=float(stats.ttest_1samp(v, 0.0).pvalue)))
    return (pd.DataFrame(rows), pd.DataFrame(soa))


def bounds(results, soa):
    rows = []
    for ref in VALID:
        for ds in DATASETS:
            pm = path(results, 'mechanism', ds, REFS[ref])
            if not os.path.exists(pm) or soa.empty:
                continue
            M = pd.read_csv(pm)
            r = M[(M.test == '1_shared_state') & (M.quantity.str.contains('Fz early vs Fz'))]
            s = soa[(soa.reference == ref) & (soa.dataset == ds)]
            if r.empty or s.empty:
                continue
            rv = float(r.value.iloc[0])
            tau = 0.15 / np.log(1.0 / rv)
            gap = float(s.median_gap_s.median())
            rows.append(dict(reference=ref, dataset=ds, within_trial_r=rv, tau_fast_s=tau, median_soa_s=gap, min_soa_s=float(s.min_gap_s.min()), single_exponential_bound_at_soa=float(np.exp(-gap / tau))))
    return pd.DataFrame(rows)


def rate(results, notes):
    a = path(results, 'matched', 'erpcore', 'r150_g05_mastoid_ic20')
    b = path(results, 'matched', 'erpcore', 'r150_g05_rs500_mastoid_ic20')
    if not (os.path.exists(a) and os.path.exists(b)):
        notes.append('sampling-rate check on mastoid skipped: needs matched_erpcore_r150_g05_rs500_mastoid_ic20.csv')
        return pd.DataFrame()
    A = pd.read_csv(a).set_index('model_id')
    R = pd.read_csv(b).set_index('model_id')
    ids = A.index.intersection(R.index)
    d = (A.loc[ids, 'dbeta'] - R.loc[ids, 'dbeta'])
    return pd.DataFrame(dict(model_id=ids, measure=A.loc[ids, 'measure'].values, dbeta_1024=A.loc[ids, 'dbeta'].values, dbeta_500=R.loc[ids, 'dbeta'].values, diff=d.values, sign_change=(np.sign(A.loc[ids, 'dbeta']) != np.sign(R.loc[ids, 'dbeta'])).values, halfwidth_1024=((A.loc[ids, 'bca_hi'] - A.loc[ids, 'bca_lo']) / 2).values))


def main():
    ap = argparse.ArgumentParser(description='Step 6: equivalence, agreement, parity decision, lags in seconds, rate check.')
    ap.add_argument('--dataset', default='all', help='ignored; step 6 always reads both datasets')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    res = cfg.RESULTS_DIR
    notes = []
    out = []
    eq = equivalence(res, notes)
    if eq.empty:
        print('no matched results found')
        return 1
    eq.to_csv(os.path.join(res, 'followup_equivalence.csv'), index=False)
    worst = float(eq.check_max_abs_diff.max())
    out.append(f'BCa recomputation vs step-3 files: max abs difference {worst:.2e} over {len(eq)} model rows ' + ('(identical)' if worst < 1e-9 else '(DIFFERS: inspect check_max_abs_diff)'))
    ag = agreement(eq)
    ag.to_csv(os.path.join(res, 'followup_agreement.csv'), index=False)
    out.append('\ncross-dataset agreement (descriptive; models are correlated)')
    for _, r in ag.iterrows():
        out.append(f'  {r.family:8s} {r.reference:12s} sign agree {r.sign_agree}/{r.n_models} (binomial p {r.binom_p:.4f}); concordant exclusions: {r.concordant or "none"}')
    ra = reference_agreement(res, notes)
    if not ra.empty:
        ra.to_csv(os.path.join(res, 'followup_reference_agreement.csv'), index=False)
        out.append(f'\nreference effect on cross-dataset agreement (subject bootstrap, {int(ra.n_boot.iloc[0])} resamples, {int(ra.n_models.iloc[0])} matched models)')
        for _, r in ra.iterrows():
            out.append(f'  {r.contrast:24s} {r.statistic:36s} {r.observed:+.3f} [{r.lo95:+.3f},{r.hi95:+.3f}]' + (f' P(<=0) {r.p_le_0:.4f}' if np.isfinite(r.p_le_0) else ''))
    out.append(f'\ncomplexity equivalence (90% BCa inside +/-{SESOI}; bound = smallest margin at which equivalence holds)')
    c = eq[(eq.family == 'battery') & eq.model_id.isin(COMPLEXITY)]
    for _, r in c.sort_values(['reference', 'dataset', 'model_id']).iterrows():
        out.append(f'  {r.reference:12s} {r.dataset:9s} {r.model_id:6s} dbeta {r.dbeta:+.4f} 90% [{r.bca90_lo:+.4f},{r.bca90_hi:+.4f}] bound {r.equivalence_bound:.4f} {"EQUIVALENT" if r.equivalent_at_sesoi else "not shown"}')
    if not c.empty:
        out.append(f'  equivalent at +/-{SESOI}: {int(c.equivalent_at_sesoi.sum())} of {len(c)} cells; largest bound {c.equivalence_bound.max():.4f}')
    sig = eq[(eq.p_holm < 0.05)]
    out.append(f'\nmodels surviving Holm correction within each dataset x reference x family: {len(sig)}')
    for _, r in sig.iterrows():
        out.append(f'  {r.family:8s} {r.reference:12s} {r.dataset:9s} {r.model_id:6s} dbeta {r.dbeta:+.4f} p_holm {r.p_holm:.4f}')
    out.append('\nparity decision')
    out += ['  ' + x for x in parity_decision(eq)]
    lg, soa = lags(res, notes)
    if not lg.empty:
        lg.to_csv(os.path.join(res, 'followup_lags_seconds.csv'), index=False)
        soa.to_csv(os.path.join(res, 'followup_soa.csv'), index=False)
        bd = bounds(res, soa)
        if not bd.empty:
            bd.to_csv(os.path.join(res, 'followup_timescale_bounds.csv'), index=False)
            out.append('\nfast-timescale bound at the measured SOA')
            for _, r in bd.iterrows():
                out.append(f'  {r.reference:8s} {r.dataset:9s} r {r.within_trial_r:+.4f} tau {r.tau_fast_s:.3f}s median SOA {r.median_soa_s:.3f}s (min {r.min_soa_s:.3f}s) bound {r.single_exponential_bound_at_soa:.2e}')
        out.append('\nacross-trial correlation by separation in seconds (mean over subjects)')
        for _, r in lg.sort_values(['reference', 'dataset', 'feature', 'bin_lo_s']).iterrows():
            out.append(f'  {r.reference:8s} {r.dataset:9s} {r.feature:12s} {r.bin_lo_s:5.1f}-{r.bin_hi_s:5.1f}s r {r.mean_r:+.4f} (se {r.se_r:.4f}) {r.n_positive}/{r.n_subjects} positive, sign p {r.sign_p:.3g}')
    rt = rate(res, notes)
    if not rt.empty:
        rt.to_csv(os.path.join(res, 'followup_rate_mastoid.csv'), index=False)
        k = rt.drop(index=rt.index[rt.model_id == 'M15-D']) if (rt.model_id == 'M15-D').any() else rt
        out.append(f'\nsampling rate on mastoid: max |dbeta(1024)-dbeta(500)| {rt["diff"].abs().max():.4f} ({rt.loc[rt["diff"].abs().idxmax(), "model_id"]}), excluding M15 {k["diff"].abs().max():.4f}, sign changes {int(rt.sign_change.sum())}, mean half-width {rt.halfwidth_1024.mean():.4f}')
    if notes:
        out.append('\nnotes')
        out += ['  ' + n for n in notes]
    txt = '\n'.join(out)
    with open(os.path.join(res, 'followup_summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
