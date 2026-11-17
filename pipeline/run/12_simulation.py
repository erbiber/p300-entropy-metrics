import argparse
import os
import sys
import zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats
import estimators as E

MODELS = ['M1-D', 'M3-D', 'M4a-D', 'M5-D', 'M6-D', 'M7-D', 'M13-D', 'M14-D', 'M15-D', 'M16-D', 'M17-D', 'M18-D', 'M19-D', 'M20-D', 'M21-D', 'M22-D', 'M23-D']
FEAT = {'M1-D': 'rms_early_fz', 'M3-D': 'sd_early_fz', 'M4a-D': 'mean_early_fz', 'M5-D': 'rms_early_fz_200', 'M6-D': 'rms_early_fz_250', 'M7-D': 'rms_base_fz', 'M13-D': 'pe_early_fz', 'M14-D': 'se_early_fz', 'M15-D': 'lz_early_fz', 'M16-D': 'skew_amp_fz', 'M17-D': 'kurt_amp_fz', 'M18-D': 'median_amp_fz', 'M19-D': 'mad_amp_fz', 'M20-D': 'p2p_amp_fz', 'M21-D': 'slope_amp_fz', 'M22-D': 'hjmob_early_fz', 'M23-D': 'hjcplx_early_fz'}
DATASETS = ('erpcore', 'ds006018')
K_DRAWS = 300
B = 4000
B_AGREE = 2000
SEED = 12345
DF_GRID = (5, 10, 20, 0)
SESOI = 0.05


def predictor_correlation(results, ds):
    d = pd.read_csv(os.path.join(results, f'features_{ds}_r150_g05_mastoid_ic20.csv'))
    Z = []
    for _, g in d.groupby('subject'):
        Z.append(np.column_stack([E.robust_z_within_subject(g[FEAT[m]].values)[0] for m in MODELS]))
    Z = np.vstack(Z)
    Z = Z[np.isfinite(Z).all(axis=1)]
    return np.corrcoef(Z.T)


def design(results):
    out = {}
    for ds in DATASETS:
        p = pd.read_csv(os.path.join(results, f'reffree_persubject_{ds}.csv'))
        p = p[p.used & (p.reference == 'mastoid')]
        n = p.groupby('subject').n_real.first()
        s = pd.read_csv(os.path.join(results, f'common_subjects_{ds}.csv')).set_index('subject')
        m = s.loc[n.index, 'pseudo_per_draw_median_free'].round().astype(int).values
        pool = np.maximum(m, s.loc[n.index, 'candidate_seconds_free'].round().astype(int).values)
        q = pd.read_csv(os.path.join(results, f'reffree_persubject_{ds}.csv'))
        q = q[q.used & (q.reference == 'csd') & q.model_id.isin(MODELS)]
        out[ds] = dict(n=n.values.astype(int), m=m, pool=pool, S=predictor_correlation(results, ds), tau=0.0, df=0,
                       sd_real=float(q.groupby('model_id').real_slope.std().mean()), sd_pseudo=float(q.groupby('model_id').pseudo_mean.std().mean()))
    return out


def calibrate(D):
    z = np.zeros(17)
    for ds in DATASETS:
        for df in DF_GRID:
            D[ds]['df'] = df
            rng = np.random.default_rng(SEED)
            r, p = simulate_dataset(D[ds], z, z, rng)
            sr, sp = r.std(0, ddof=1).mean(), p.std(0, ddof=1).mean()
            if sr <= 1.01 * D[ds]['sd_real']:
                break
        k = (sp / D[ds]['sd_pseudo']) ** 2
        D[ds]['pool'] = np.maximum(D[ds]['m'], np.round(D[ds]['pool'] * k).astype(int))
        D[ds]['tau'] = float(np.sqrt(max(0.0, D[ds]['sd_real'] ** 2 - sr ** 2)))
        rng = np.random.default_rng(SEED + 1)
        r2, p2 = simulate_dataset(D[ds], z, z, rng)
        D[ds]['calibration'] = dict(df=('normal' if D[ds]['df'] == 0 else D[ds]['df']), pool_factor=k, tau=D[ds]['tau'], sd_real_observed=D[ds]['sd_real'], sd_real_simulated=r2.std(0, ddof=1).mean(), sd_pseudo_observed=D[ds]['sd_pseudo'], sd_pseudo_simulated=p2.std(0, ddof=1).mean())
    return D


def observed_profile(results):
    p = pd.read_csv(os.path.join(results, 'reffree_persubject_erpcore.csv'))
    p = p[p.used & (p.reference == 'mastoid')]
    return p.groupby('model_id').real_slope.mean().reindex(MODELS).values


def scenarios(prof):
    z = np.zeros(17)
    u = np.full(17, 0.05)
    return {
        'null': dict(real=(z, z), pseudo=(z, z)),
        'shared_background': dict(real=(prof, prof), pseudo=(prof, prof)),
        'mirror_background': dict(real=(prof, -prof), pseudo=(prof, -prof)),
        'event_related': dict(real=(prof, prof), pseudo=(z, z)),
        'boundary_0.05': dict(real=(u, u), pseudo=(z, z)),
    }


def participant_profile(c, S, Sh, Sinv, rng, tau):
    for _ in range(1000):
        ci = c + tau * (Sh @ rng.standard_normal(17))
        if ci @ Sinv @ ci < 0.95:
            return ci
    return c


def draw(S, ci, n, rng, df):
    R = np.empty((18, 18))
    R[:17, :17] = S
    R[:17, 17] = ci
    R[17, :17] = ci
    R[17, 17] = 1.0
    L = np.linalg.cholesky(R)
    g = rng.standard_normal((n, 18)) @ L.T
    if df == 0:
        return g
    return g / np.sqrt(rng.chisquare(df, n) / df)[:, None]


def robust_z(A, axis):
    med = np.median(A, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(A - med), axis=axis, keepdims=True)
    return (A - med) / np.where(mad > 0, mad, 1.0)


def slopes(Z, axis):
    Z = robust_z(Z, axis)
    Zc = Z - Z.mean(axis=axis, keepdims=True)
    x, y = Zc[..., :17], Zc[..., 17:]
    return (x * y).sum(axis=axis) / (x ** 2).sum(axis=axis)


def simulate_dataset(D, c_real, c_pseudo, rng):
    S = D['S']
    w, V = np.linalg.eigh(S)
    Sh = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    Sinv = np.linalg.inv(S)
    real, pseudo = [], []
    for n, m, N in zip(D['n'], D['m'], D['pool']):
        real.append(slopes(draw(S, participant_profile(c_real, S, Sh, Sinv, rng, D['tau']), n, rng, D['df']), 0))
        pool = draw(S, participant_profile(c_pseudo, S, Sh, Sinv, rng, D['tau']), N, rng, D['df'])
        idx = np.argsort(rng.random((K_DRAWS, N)), axis=1)[:, :m]
        pseudo.append(slopes(pool[idx], 1).mean(axis=0))
    return np.array(real), np.array(pseudo)


def boot_stats(d):
    n = len(d)
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, n, (B, n))].mean(axis=1)
    th = d.mean()
    z0 = stats.norm.ppf(min(max(np.mean(boot < th), 1e-6), 1 - 1e-6))
    jack = (d.sum() - d) / (n - 1)
    jm = jack.mean()
    den = 6 * np.sum((jm - jack) ** 2) ** 1.5
    a = np.sum((jm - jack) ** 3) / den if den else 0.0
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    q = lambda al: (np.percentile(boot, 100 * adj(stats.norm.ppf(al / 2))), np.percentile(boot, 100 * adj(stats.norm.ppf(1 - al / 2))))
    lo95, hi95 = q(0.05)
    lo90, hi90 = q(0.10)
    p_inv = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()) + 1.0 / B)
    p_cen = (1 + np.sum(np.abs(boot - th) >= abs(th))) / (B + 1)
    return th, lo95, hi95, lo90, hi90, p_inv, p_cen, d.std(ddof=1) / np.sqrt(n)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def agreement(XE, XD):
    rng = np.random.default_rng(SEED)
    ie = rng.integers(0, len(XE), (B_AGREE, len(XE)))
    idd = rng.integers(0, len(XD), (B_AGREE, len(XD)))
    T = np.corrcoef(XE.mean(0), XD.mean(0))[0, 1]
    bt = np.array([np.corrcoef(XE[a].mean(0), XD[b].mean(0))[0, 1] for a, b in zip(ie, idd)])
    return T, np.percentile(bt, 2.5), np.percentile(bt, 97.5), np.abs(XE.mean(0) - XD.mean(0)).mean()


def one_rep(args):
    name, sc, D, rep = args
    rng = np.random.default_rng([SEED, rep, zlib.crc32(name.encode())])
    rows, agree = [], []
    sims = {}
    for k, ds in enumerate(DATASETS):
        r, p = simulate_dataset(D[ds], sc['real'][k], sc['pseudo'][k], rng)
        sims[ds] = dict(real_slope=r, pseudo_mean=p, d=r - p)
        truth = dict(real_slope=sc['real'][k], pseudo_mean=sc['pseudo'][k], d=sc['real'][k] - sc['pseudo'][k])
        for q in ('real_slope', 'd'):
            st = np.array([boot_stats(sims[ds][q][:, j]) for j in range(17)])
            hi = holm(st[:, 5]) < 0.05
            hc = holm(st[:, 6]) < 0.05
            t = truth[q]
            rows.append(dict(scenario=name, rep=rep, dataset=ds, quantity=q,
                             cover95=float(np.mean((st[:, 1] <= t) & (t <= st[:, 2]))),
                             reject95=float(np.mean((st[:, 1] > 0) | (st[:, 2] < 0))),
                             any_holm_inv=bool(hi.any()), any_holm_cen=bool(hc.any()),
                             n_holm_inv=int(hi.sum()), n_holm_cen=int(hc.sum()),
                             n_equivalent=int(np.sum(np.maximum(np.abs(st[:, 3]), np.abs(st[:, 4])) < SESOI)),
                             mean_abs=float(np.abs(st[:, 0]).mean()), noise_floor=float((st[:, 7] * np.sqrt(2 / np.pi)).mean()),
                             bias=float(np.mean(st[:, 0] - t)), mean_se=float(st[:, 7].mean())))
    for q in ('real_slope', 'pseudo_mean', 'd'):
        T, lo, hi, ad = agreement(sims['erpcore'][q], sims['ds006018'][q])
        tE, tD = (sc['real'] if q == 'real_slope' else sc['pseudo'] if q == 'pseudo_mean' else (sc['real'][0] - sc['pseudo'][0], sc['real'][1] - sc['pseudo'][1]))
        tr = np.corrcoef(tE, tD)[0, 1] if np.std(tE) > 0 and np.std(tD) > 0 else np.nan
        agree.append(dict(scenario=name, rep=rep, quantity=q, r=T, lo=lo, hi=hi, abs_disagreement=ad, true_r=tr))
    return rows, agree


def main():
    ap = argparse.ArgumentParser(description='Step 12: calibration of the estimators by simulation. Participants are simulated with the observed numbers of real trials, pseudotrials per draw and eligible pseudotrial time, trial-level data from a multivariate t distribution whose degrees of freedom (5, 10, 20 or normal) are the heaviest-tailed that does not exceed the observed spread of real-trial slopes with the observed correlation among the 17 predictors, participant-level heterogeneity of the true slopes and an effective pseudotrial pool calibrated so that the between-participant spread of real-trial and pseudotrial slopes matches that observed under the surface Laplacian, and known true slopes; the simulated data pass through robust standardisation, the OLS slope, 300 overlapping pseudotrial draws, BCa intervals, both bootstrap p-value procedures with Holm adjustment, equivalence at ±0.05 and profile agreement.')
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dataset', default='all')
    a = ap.parse_args()
    a.out = a.out or a.results
    os.makedirs(a.out, exist_ok=True)
    D = calibrate(design(a.results))
    cal = pd.DataFrame([dict(dataset=ds, **D[ds]['calibration']) for ds in DATASETS])
    cal.to_csv(os.path.join(a.out, 'simulation_calibration.csv'), index=False)
    print(cal.round(4).to_string())
    S = scenarios(observed_profile(a.results))
    jobs = [(name, sc, D, rep) for name, sc in S.items() for rep in range(a.reps)]
    with Pool(a.workers) as pool:
        res = pool.map(one_rep, jobs, chunksize=4)
    rows = pd.DataFrame([r for x in res for r in x[0]])
    agree = pd.DataFrame([r for x in res for r in x[1]])
    rows.to_csv(os.path.join(a.out, 'simulation_reps.csv'), index=False)
    agree.to_csv(os.path.join(a.out, 'simulation_agreement_reps.csv'), index=False)
    summ = rows.groupby(['scenario', 'dataset', 'quantity']).agg(reps=('rep', 'size'), coverage95=('cover95', 'mean'), reject95=('reject95', 'mean'), fwer_holm_inv=('any_holm_inv', 'mean'), fwer_holm_cen=('any_holm_cen', 'mean'), mean_n_holm_inv=('n_holm_inv', 'mean'), p_all17_equivalent=('n_equivalent', lambda v: np.mean(v == 17)), mean_n_equivalent=('n_equivalent', 'mean'), mean_abs=('mean_abs', 'mean'), noise_floor=('noise_floor', 'mean'), bias=('bias', 'mean'), mean_se=('mean_se', 'mean')).reset_index()
    summ.to_csv(os.path.join(a.out, 'simulation_summary.csv'), index=False)
    ag = agree.groupby(['scenario', 'quantity']).agg(reps=('rep', 'size'), true_r=('true_r', 'first'), mean_r=('r', 'mean'), r_q025=('r', lambda v: np.percentile(v, 2.5)), r_q975=('r', lambda v: np.percentile(v, 97.5)), p_ci_excludes_zero=('lo', lambda v: np.nan), mean_abs_disagreement=('abs_disagreement', 'mean')).reset_index()
    ag['p_ci_excludes_zero'] = [np.mean((g.lo > 0) | (g.hi < 0)) for _, g in agree.groupby(['scenario', 'quantity'])]
    ag['p_ci_covers_true'] = [np.mean((g.lo <= g.true_r) & (g.true_r <= g.hi)) if g.true_r.notna().all() else np.nan for _, g in agree.groupby(['scenario', 'quantity'])]
    ag.to_csv(os.path.join(a.out, 'simulation_agreement.csv'), index=False)
    print(summ.round(3).to_string())
    print(ag.round(3).to_string())
    print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
