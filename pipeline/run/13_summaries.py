import argparse
import importlib.util
import json
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats
import config as cfg
import estimators as E

warnings.filterwarnings('ignore', category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 12345
B = 2000
SAME = ['M8-D', 'M9a-D', 'M10-D', 'M24-D']
COMPLEXITY = ['M13-D', 'M14-D', 'M15-D', 'M22-D', 'M23-D']
AMPLITUDE = ['M1-D', 'M3-D', 'M5-D', 'M6-D', 'M7-D', 'M19-D', 'M20-D']
QS = ['real_slope', 'pseudo_mean', 'd']
DS = ('erpcore', 'ds006018')
MATCHED = {'distributed': 'r150_g05_ic20', 'mastoid': 'r150_g05_mastoid_ic20', 'average': 'r150_g05_average_ic20'}
SHORT = {'dist': '', 'cz': '_cz', 'mast': '_mastoid', 'aver': '_average'}
LAG_FEATURES = ['rms_base_fz', 'sd_early_fz', 'pe_early_fz', 'lz_early_fz']
LAGS = [1, 2, 3, 5, 10, 20]


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def records(df):
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict('records')


def csv(results, name):
    return pd.read_csv(os.path.join(results, name))


def positive_control(results, ds, tag, window):
    e = csv(results, f'erp_{ds}_{tag}.csv')
    e = e[e.channel == 'Pz']
    cols = [c for c in e.columns[5:] if window[0] * 1000 <= float(c) <= window[1] * 1000]
    v = e.set_index(['subject', 'condition'])[cols].mean(axis=1).unstack()
    dd = (v['target'] - v['standard']) * 1e6
    return dict(mean_uv=round(float(dd.mean()), 2), sd_uv=round(float(dd.std()), 2), dz=round(float(dd.mean() / dd.std()), 2), pos=int((dd > 0).sum()), n=int(len(dd)), target_uv=round(float(v['target'].mean() * 1e6), 2), standard_uv=round(float(v['standard'].mean() * 1e6), 2), t_p=float(stats.ttest_1samp(dd, 0).pvalue))


def cells(results, window):
    V = {}
    for ref, tag in MATCHED.items():
        for ds in DS:
            L = csv(results, f'features_{ds}_{tag}_log.csv')
            ok = L[L.status == 'ok']
            V[f'n_{ds}_{ref}'] = dict(subjects=int(len(ok)), trials=int(ok.n_trials.sum()), targets=int(ok.n_target.sum()), standards=int(ok.n_standard.sum()), listed=int(len(L)), excluded={r.subject: r.status for _, r in L[L.status != 'ok'].iterrows()})
            V[f'mech_{ds}_{ref}'] = records(csv(results, f'mechanism_{ds}_{tag}.csv')[['test', 'quantity', 'value', 'n_positive', 'lo', 'hi']])
            V[f'matched_{ds}_{ref}'] = records(csv(results, f'matched_{ds}_{tag}.csv'))
            V[f'canon_{ds}_{ref}'] = records(csv(results, f'canonical_lmm_{ds}_{tag}.csv'))
            V[f'pc_{ds}_{ref}'] = positive_control(results, ds, tag, window)
    V['pc_erpcore_cz'] = positive_control(results, 'erpcore', 'r150_g05_cz_ic20', window)
    return V


def common_subjects(results):
    refs = {'erpcore': ['distributed', 'cz', 'mastoid', 'average'], 'ds006018': ['distributed', 'mastoid', 'average']}
    out = {}
    for ds in DS:
        c = csv(results, f'common_subjects_{ds}.csv')
        ok = c[c.status == 'ok']
        pf = csv(results, f'common_persubject_{ds}_free.csv')
        po = csv(results, f'common_persubject_{ds}_onset.csv')
        uf = sorted(pf[pf.used].subject.unique())
        uo = sorted(po[po.used].subject.unique())
        of = ok[ok.subject.isin(uf)]
        o = dict(listed=len(c), processed=len(ok), used_free=len(uf), used_onset=len(uo), trials=int(ok.n_real.sum()), targets=int(ok.n_target.sum()), standards=int(ok.n_standard.sum()),
                 ica_mean=float(ok.n_ica_excluded.mean()), ica_min=int(ok.n_ica_excluded.min()), ica_max=int(ok.n_ica_excluded.max()),
                 pseudo_free_median=float(of.pseudo_per_draw_median_free.median()), pseudo_onset_median=float(ok.pseudo_per_draw_median_onset.median()),
                 eligible_free_median=float(of.candidate_seconds_free.median()), eligible_free_min=float(of.candidate_seconds_free.min()),
                 share_event_free=float(ok.share_with_event_in_epoch_free.max()), share_event_onset_mean=float(ok.share_with_event_in_epoch_onset.mean()),
                 gap_start_min=float(ok.gap_epoch_start_after_event_min_free.min()), gap_end_min=float(ok.gap_epoch_end_before_event_min_free.min()),
                 rejected_median={r: float(ok[f'rejected_under_{r}'].median()) for r in refs[ds]},
                 draws_min_free=int(pf[pf.used].n_draws_valid.min()), draws_min_onset=int(po[po.used].n_draws_valid.min()),
                 excluded=[dict(subject=r.subject, status=r.status) for _, r in c[c.status != 'ok'].iterrows()],
                 free_unused=sorted(set(ok.subject) - set(uf)), trials_free=int(of.n_real.sum()))
        for nm, g in (('all', ok), ('free', of)):
            for col in ('n_target', 'n_standard'):
                o[f'{col}_{nm}_median'] = float(g[col].median())
                o[f'{col}_{nm}_min'] = int(g[col].min())
                o[f'{col}_{nm}_max'] = int(g[col].max())
        out[ds] = o
    return out


def profiles(results):
    p = csv(results, 'common_profiles_free.csv')
    return {f'{ds}_{r}': {m: dict(zip(h.quantity, h['mean'])) for m, h in g.groupby('model_id')} for (ds, r), g in p.groupby(['dataset', 'reference'])}


def all21(ca, results):
    A, subs, models = ca.load(results, 'free')
    t = ca.transport(A, subs, models, np.ones(len(models), bool), ['d'])
    return records(t[['contrast', 'quantity', 'T', 'lo', 'hi', 'spearman', 'p_le0']])


def sensitivity(results):
    tags = {'distributed': '', 'cz': '_cz', 'mastoid': '_mastoid', 'average': '_average'}
    rows = []
    for ds, refs in (('erpcore', ['distributed', 'cz', 'mastoid', 'average']), ('ds006018', ['distributed', 'mastoid', 'average'])):
        for r in refs:
            c = csv(results, f'common_persubject_{ds}_free.csv')
            c = c[c.used & (c.reference == r)]
            m = csv(results, f'matched_persubject_{ds}_r150_g05{tags[r]}_ic20.csv')
            m = m[m.used]
            a = c.groupby('model_id').real_slope.mean()
            b = m.groupby('model_id').real_slope.mean()
            k = [x for x in a.index if x not in SAME and x in b.index]
            rows.append(dict(dataset=ds, reference=r, r=float(np.corrcoef(a[k], b[k])[0, 1]), max_abs_diff=float((a[k] - b[k]).abs().max())))
    return rows


def specificity(results):
    D = {}
    for ref, tag in MATCHED.items():
        for ds in DS:
            P = csv(results, f'matched_persubject_{ds}_{tag}.csv')
            P = P[P.used]
            for q in QS:
                D[(ref, ds, q)] = P.pivot_table(index='subject', columns='model_id', values=q, aggfunc='first')
    models = sorted(set.intersection(*[set(v.columns) for v in D.values()]))
    subs = {ds: sorted(set().union(*[set(D[(r, ds, 'd')].index) for r in MATCHED])) for ds in DS}
    sets = {'all21': models, 'crosschannel17': [m for m in models if m not in SAME], 'crosschannel_noncomplexity12': [m for m in models if m not in SAME + COMPLEXITY], 'non-complexity16': [m for m in models if m not in COMPLEXITY], 'complexity5': COMPLEXITY}
    def prof(idx):
        return {(r, q): [D[(r, ds, q)].reindex(idx[ds])[models].mean(axis=0, skipna=True) for ds in DS] for r in MATCHED for q in QS}
    def stat(P):
        out = {}
        for s, ms in sets.items():
            for r in MATCHED:
                for q in QS:
                    a, b = P[(r, q)]
                    a, b = a[ms].values, b[ms].values
                    ok = np.isfinite(a) & np.isfinite(b)
                    out[(s, r, q)] = np.corrcoef(a[ok], b[ok])[0, 1]
        return out
    obs = stat(prof(subs))
    rng = np.random.default_rng(SEED)
    bt = []
    for _ in range(B):
        idx = {ds: list(rng.choice(subs[ds], len(subs[ds]), replace=True)) for ds in DS}
        bt.append(stat(prof(idx)))
    rows = []
    for s in sets:
        for q in QS:
            for r in MATCHED:
                v = np.array([b[(s, r, q)] for b in bt])
                rows.append(dict(set=s, quantity=q, contrast=r, T=float(obs[(s, r, q)]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p_le0=None))
            for r in ('mastoid', 'average'):
                v = np.array([b[(s, r, q)] - b[(s, 'distributed', q)] for b in bt])
                rows.append(dict(set=s, quantity=q, contrast=f'{r}-distributed', T=float(obs[(s, r, q)] - obs[(s, 'distributed', q)]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p_le0=float((v <= 0).mean())))
        for r in ('mastoid', 'average'):
            f = lambda b: (b[(s, r, 'real_slope')] - b[(s, 'distributed', 'real_slope')]) - (b[(s, r, 'pseudo_mean')] - b[(s, 'distributed', 'pseudo_mean')])
            v = np.array([f(b) for b in bt])
            rows.append(dict(set=s, quantity='interaction real-pseudo', contrast=f'{r}-distributed', T=float(f(obs)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p_le0=float((v <= 0).mean())))
    return rows


def refstruct(results):
    dtag = {'dist': '', 'cz': '', 'mast': '_mastoid', 'aver': '_average'}
    rows = []
    for base, suffix in (('baseline', ''), ('nobase', '_nobase')):
        A = {}
        for c in SHORT:
            for ds, tg in (('erpcore', SHORT), ('ds006018', dtag)):
                P = csv(results, f'matched_persubject_{ds}_r150_g05{tg[c]}{suffix}_ic20.csv')
                P = P[P.used]
                for q in QS:
                    A[(ds, c, q)] = P.pivot_table(index='subject', columns='model_id', values=q, aggfunc='first')
        models = sorted(A[('erpcore', 'dist', 'd')].columns)
        sel = np.array([m not in SAME for m in models])
        subs = {ds: sorted(set.intersection(*[set(A[(ds, c, 'd')].index) for c in SHORT])) for ds in DS}
        M = {k: v.loc[subs[k[0]], models].values for k, v in A.items()}
        rng = np.random.default_rng(SEED)
        ie = [rng.integers(0, len(subs['erpcore']), len(subs['erpcore'])) for _ in range(B)]
        idd = [rng.integers(0, len(subs['ds006018']), len(subs['ds006018'])) for _ in range(B)]
        T = lambda c, q, a, b: np.corrcoef(np.nanmean(M[('erpcore', c, q)][a], 0)[sel], np.nanmean(M[('ds006018', c, q)][b], 0)[sel])[0, 1]
        fe, fd = np.arange(len(subs['erpcore'])), np.arange(len(subs['ds006018']))
        obs = {(c, q): T(c, q, fe, fd) for c in SHORT for q in QS}
        bt = {k: np.array([T(k[0], k[1], a, b) for a, b in zip(ie, idd)]) for k in obs}
        n = dict(n_erp=len(subs['erpcore']), n_ds=len(subs['ds006018']))
        row = lambda c, q, t, d, p=None: dict(baseline=base, contrast=c, quantity=q, T=float(t), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), **n, p_le0=p)
        for c in SHORT:
            for q in QS:
                rows.append(row(c, q, obs[(c, q)], bt[(c, q)]))
        for c in ('cz', 'mast', 'aver'):
            for q in QS:
                d = bt[(c, q)] - bt[('dist', q)]
                rows.append(row(f'{c}-dist', q, obs[(c, q)] - obs[('dist', q)], d, float((d <= 0).mean())))
            d = (bt[(c, 'real_slope')] - bt[('dist', 'real_slope')]) - (bt[(c, 'pseudo_mean')] - bt[('dist', 'pseudo_mean')])
            o = (obs[(c, 'real_slope')] - obs[('dist', 'real_slope')]) - (obs[(c, 'pseudo_mean')] - obs[('dist', 'pseudo_mean')])
            rows.append(row(f'{c}-dist', 'interaction', o, d))
        for q in QS:
            rows.append(row('cz-mast', q, obs[('cz', q)] - obs[('mast', q)], bt[('cz', q)] - bt[('mast', q)]))
    return rows


def symmetry(results):
    rows = []
    for ds, refs in (('erpcore', ['dist', 'cz', 'mast', 'aver']), ('ds006018', ['dist', 'mast', 'aver'])):
        for ref in refs:
            sym = csv(results, f'symmetry_{ds}_r150_g05{SHORT[ref]}_ic20.csv').set_index('subject')
            m = csv(results, f'matched_persubject_{ds}_r150_g05{SHORT[ref]}_ic20.csv')
            m = m[m.used]
            v = m[m.model_id.isin(AMPLITUDE)].pivot_table(index='subject', columns='model_id', values='pseudo_mean').mean(axis=1)
            s = sym.reindex(v.index)
            for meas in ('bsi_fz', 'bsi_pz', 'skew_fz', 'skew_pz'):
                ok = s[meas].notna()
                r, p = stats.spearmanr(s[meas][ok], v[ok])
                rows.append(dict(dataset=ds, ref=ref, n=int(ok.sum()), measure=meas, rho=float(r), p=float(p), mean=float(s[meas][ok].mean()), n_pos=int((s[meas][ok] > 0).sum())))
    for r in rows:
        r['p_bonf'] = float(min(1.0, r['p'] * len(rows)))
    return rows


def reffree_magnitude(results):
    d = csv(results, 'reffree_dbeta.csv')
    d = d[~d.model_id.isin(SAME)]
    rows = []
    for (ds, r, q), g in d.groupby(['dataset', 'reference', 'quantity']):
        ex = (g.bca95_lo > 0) | (g.bca95_hi < 0)
        rows.append(dict(dataset=ds, reference=r, quantity=q, mean_abs=float(g['mean'].abs().mean()), max_abs=float(g['mean'].abs().max()), n_excl0=int(ex.sum()), n_holm=int((g.p_holm < 0.05).sum())))
    return rows


def lags_trials(results):
    out = {}
    soa = csv(results, 'followup_soa.csv')
    for ref in ('mastoid', 'average'):
        for ds in DS:
            d = csv(results, f'features_{ds}_{MATCHED[ref]}.csv')
            o = {}
            for c in LAG_FEATURES:
                o[c] = {}
                for k in LAGS:
                    rs = []
                    for _, g in d.groupby('subject'):
                        x = g.set_index('trial_idx')[c]
                        x = (x - x.median()) / (1.4826 * (x - x.median()).abs().median())
                        y = x.reindex(x.index + k)
                        m = ~y.isna().values
                        if m.sum() > 10:
                            rs.append(np.corrcoef(x.values[m], y.values[m])[0, 1])
                    rs = np.array(rs)
                    rs = rs[np.isfinite(rs)]
                    pos = int((rs > 0).sum())
                    o[c][str(k)] = dict(mean=round(float(rs.mean()), 3), pos=pos, n=len(rs), p=float(stats.binomtest(pos, len(rs), 0.5).pvalue))
            M = csv(results, f'mechanism_{ds}_{MATCHED[ref]}.csv')
            r = M[(M.test == '1_shared_state') & M.quantity.str.contains('Fz early vs Fz')].iloc[0]
            rv = float(r.value)
            tau = 0.15 / np.log(1.0 / rv)
            s = soa[(soa.reference == ref) & (soa.dataset == ds)]
            med = float(s.median_gap_s.median())
            b1 = float(np.exp(-1.0 / tau))
            bs = float(np.exp(-med / tau))
            l1 = o['rms_base_fz']['1']['mean']
            o.update(within_r=round(rv, 4), within_pos=float(r.n_positive), tau=round(tau, 3), bound_1s=b1, ratio_lag1=int(round(l1 / b1)), bound_soa=bs, median_soa=med, min_soa=float(s.min_gap_s.min()), ratio_soa=int(round(l1 / bs)))
            out[f'{ds}_{ref}'] = o
    return {k: out[k] for k in ('erpcore_mastoid', 'erpcore_average', 'ds006018_mastoid', 'ds006018_average')}


def slope_correlation(results):
    feats = {'M13-D': 'pe_early_fz', 'M14-D': 'se_early_fz', 'M15-D': 'lz_early_fz', 'M22-D': 'hjmob_early_fz', 'M23-D': 'hjcplx_early_fz'}
    rows = []
    for ds in DS:
        d = csv(results, f'features_{ds}_{MATCHED["mastoid"]}.csv')
        m = csv(results, f'matched_persubject_{ds}_{MATCHED["mastoid"]}.csv')
        used = set(m[m.used].subject)
        for mid, c in feats.items():
            sl, rr = [], []
            for s, g in d.groupby('subject'):
                if s not in used:
                    continue
                x, _ = E.robust_z_within_subject(g[c].values)
                y, _ = E.robust_z_within_subject(g.p300_pz.values)
                ok = np.isfinite(x) & np.isfinite(y)
                sl.append(E.ols_slope(x, y))
                rr.append(np.corrcoef(x[ok], y[ok])[0, 1])
            rows.append(dict(dataset=ds, reference='mastoid', model_id=mid, mean_slope=float(np.nanmean(sl)), mean_r=float(np.nanmean(rr)), abs_diff=float(abs(np.nanmean(sl) - np.nanmean(rr)))))
    return rows


def drift(results):
    na = csv(results, 'robustness_neighbour_adjustment.csv')
    ns = csv(results, 'robustness_neighbour_share.csv')
    v = lambda a, c: [float(x) for x in na[(na.adjusted == a) & (na.contrast == c)][['value', 'lo', 'hi']].iloc[0]]
    cross = {'raw': {r: v(False, r) for r in MATCHED}, 'adj': {r: v(True, r) for r in MATCHED},
             'raw_change': {r: v(False, f'{r}-distributed') for r in ('mastoid', 'average')}, 'adj_change': {r: v(True, f'{r}-distributed') for r in ('mastoid', 'average')}}
    n = [int(ns[ns.dataset == ds].n.iloc[0]) for ds in DS]
    share = {f'{r.dataset}_{r.reference}': float(r.retained_share) for _, r in ns.iterrows()}
    return dict(n=n, cross17=cross, retained_share=share)


def main():
    ap = argparse.ArgumentParser(description='Step 13: collect the values reported in the tables and figures into one file, deriving participant summaries, profile sensitivity, specificity, baseline structure, symmetry associations, reference-free magnitudes and persistence from the step 1 to 12 results.')
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--dataset', default='all')
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.results, 'paper_values.json')
    R = a.results
    ca = module('common_analysis', '09_common_analysis.py')
    fu = module('followup', '06_followup.py')
    notes = []
    agreement = fu.reference_agreement(R, notes)
    if agreement.empty:
        print('\n'.join(notes))
        return 1
    V = cells(R, cfg.P300_WINDOW)
    V['agreement'] = records(agreement)
    V['parity_erpcore_mastoid'] = records(csv(R, 'matched_parity_erpcore_r150_g05_mastoid_ic20.csv'))
    for k in ('absdis_free', 'dbeta_free', 'location_free', 'robust_free', 'threshold_free', 'transport_free', 'transport_onset'):
        V[f'common_{k}'] = records(csv(R, f'common_{k}.csv'))
    V['common_condition'] = records(csv(R, 'common_condition_summary.csv'))
    V['common_subjects'] = common_subjects(R)
    V['common_profiles_free'] = profiles(R)
    V['common_all21_free'] = all21(ca, R)
    V['common_sensitivity'] = sensitivity(R)
    V['lags_seconds'] = records(csv(R, 'followup_lags_seconds.csv'))
    V['lags_trials'] = lags_trials(R)
    V['drift'] = drift(R)
    V['slope_correlation'] = slope_correlation(R)
    V['specificity'] = specificity(R)
    V['refstruct'] = refstruct(R)
    V['symmetry_assoc'] = symmetry(R)
    V['reffree_agreement'] = records(csv(R, 'reffree_agreement.csv'))
    V['reffree_dbeta'] = records(csv(R, 'reffree_dbeta.csv'))
    V['reffree_poscontrol'] = records(csv(R, 'reffree_poscontrol_summary.csv'))
    V['reffree_magnitude'] = reffree_magnitude(R)
    V['robustness_summary'] = records(csv(R, 'robustness_summary.csv'))
    V['robustness_absint'] = records(csv(R, 'robustness_absolute_interaction.csv'))
    V['robustness_bca'] = records(csv(R, 'robustness_bca_agreement.csv'))
    V['robustness_distribution'] = records(csv(R, 'robustness_distribution.csv'))
    V['robustness_tests'] = records(csv(R, 'robustness_tests.csv'))
    V['robustness_location'] = records(csv(R, 'robustness_location_agreement.csv'))
    for k, f in (('sim_summary', 'simulation_summary'), ('sim_agreement', 'simulation_agreement'), ('sim_calibration', 'simulation_calibration')):
        V[k] = records(pd.read_csv(os.path.join(R, f'{f}.csv'), keep_default_na=False, na_values=['']))
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(V, fh, indent=1)
    print(f'{len(V)} entries -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
