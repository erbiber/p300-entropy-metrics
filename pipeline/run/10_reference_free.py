import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import config as cfg
import estimators as E
import placement as P
from lib import fastfeat

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load('common08', '08_common.py')
VERSION = 'f1'
COMMON18 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
DELAY_S = {'erpcore': 0.026, 'ds006018': 0.020}
PSEUDO_REFS = ('mastoid', 'csd', 'csd18')
R_SUB = 200
SUB_SEED = 777
B_BOOT = 2000
B_BCA = 4000
SEED = 12345
SAME = {'M8-D', 'M9a-D', 'M10-D', 'M24-D'}
LOCATION = {'M4a-D', 'M18-D', 'M21-D'}


def csd_pair(c, chans):
    import mne
    names = c['names']
    low = {n.lower(): i for i, n in enumerate(names)}
    miss = [x for x in chans if x.lower() not in low]
    if miss:
        return None, 'csd_missing_' + '_'.join(miss)
    ix = [low[x.lower()] for x in chans]
    info = mne.create_info([names[i] for i in ix], c['sfreq'], 'eeg')
    raw = mne.io.RawArray(c['V'][ix], info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), match_case=False, verbose=False)
    out = mne.preprocessing.compute_current_source_density(raw, verbose=False).get_data()
    sub = [names[i].lower() for i in ix]
    return (out[sub.index('fz')], out[sub.index('pz')]), None


def trial_set(c, dataset, ptp, shift):
    sf = c['sfreq']
    targets, standards = cfg.codes_for(dataset)
    keep = set(targets + standards)
    ev = c['events']
    codes = np.array([c['code_of'].get(int(v)) for v in ev[:, 2]], dtype=object)
    sel = np.array([x is not None and x in keep for x in codes])
    samples = ev[sel, 0].astype(np.int64) + int(round(shift * sf))
    codes = codes[sel]
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    inb = (samples + lo >= 0) & (samples + hi < c['n'])
    samples, codes = samples[inb], codes[inb]
    ok = ptp[samples] <= cfg.REJECT_THRESHOLD
    samples, codes = samples[ok], codes[ok]
    return samples, np.array([int(x in targets) for x in codes])


def subject(args):
    sub, dataset, K, cache_dir, state = args
    for k, v in state.items():
        setattr(cfg, k, v)
    models = C.battery()
    t0 = time.time()
    cp = os.path.join(cache_dir, f'reffree_{VERSION}_{dataset}_K{K}_{sub}.pkl')
    if os.path.exists(cp):
        return pd.read_pickle(cp), 'cached', 0.0
    c, err = C.clean_continuous(sub, dataset)
    if err:
        return dict(rows=None, info=dict(subject=sub, status=err)), 'excluded', time.time() - t0
    refs, scalp = C.reference_signals(c, dataset)
    if refs is None:
        return dict(rows=None, info=dict(subject=sub, status=scalp)), 'excluded', time.time() - t0
    ptp, _ = C.combined_ptp(c, refs, scalp)
    sf = c['sfreq']
    idx = {n: i for i, n in enumerate(c['names'])}
    sig = {r: (c['V'][idx['Fz']] - s, c['V'][idx['Pz']] - s) for r, s in refs.items()}
    full = [c['names'][i] for i in scalp]
    for name, chans in (('csd', full), ('csd18', COMMON18)):
        try:
            pair, err = csd_pair(c, chans)
        except Exception as ex:
            pair, err = None, f'csd_failed_{type(ex).__name__}'
        if err:
            return dict(rows=None, info=dict(subject=sub, status=err)), 'excluded', time.time() - t0
        sig[name] = pair
    samples, is_t = trial_set(c, dataset, ptp, 0.0)
    info = dict(subject=sub, n_real=len(samples), n_target=int(is_t.sum()), n_standard=int(len(is_t) - is_t.sum()), n_csd_channels=len(full))
    if is_t.sum() < cfg.MIN_TRIALS_PER_CONDITION or (len(is_t) - is_t.sum()) < cfg.MIN_TRIALS_PER_CONDITION or len(samples) < cfg.MIN_TRIALS_REQUIRED:
        info['status'] = 'below_minimum_trials'
        return dict(rows=None, info=info), 'excluded', time.time() - t0
    s_sh, t_sh = trial_set(c, dataset, ptp, DELAY_S[dataset])
    info['n_real_shifted'] = len(s_sh)
    outcome = models[0].outcome
    cols = sorted({m.predictor for m in models} | {m.outcome for m in models})
    feats, real, real_sh, pc = {}, {}, {}, []
    for r, (fzs, pzs) in sig.items():
        fz, pz, times = C.epochs(fzs, pzs, samples, sf)
        f = fastfeat.window_features(fz, pz, times, cfg, want_entropy=True)
        feats[r] = {col: np.asarray(f[col], float) for col in cols}
        real[r] = C.slopes(f, models, is_t)
        y = feats[r][outcome]
        fz2, pz2, times2 = C.epochs(fzs, pzs, s_sh, sf)
        f2 = fastfeat.window_features(fz2, pz2, times2, cfg, want_entropy=True)
        real_sh[r] = C.slopes(f2, models, t_sh)
        y2 = np.asarray(f2[outcome], float)
        pc.append(dict(subject=sub, reference=r, target_minus_standard=float(np.nanmean(y[is_t == 1]) - np.nanmean(y[is_t == 0])), target_minus_standard_shifted=float(np.nanmean(y2[t_sh == 1]) - np.nanmean(y2[t_sh == 0])), sd_within=float(np.nanstd(y, ddof=1))))
    all_events = c['events'][:, 0].astype(np.int64)
    share = is_t.mean()
    fb = C.forbidden(c['n'], all_events, sf, 'free')
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    rng = np.random.default_rng(cfg.PSEUDOTRIAL_SEED)
    scratch = np.zeros(c['n'], dtype=bool)
    acc = {(r, m.id): [] for r in PSEUDO_REFS for m in models}
    scl = {(r, col): [] for r in PSEUDO_REFS for col in cols}
    npd = []
    for k in range(K):
        s = P.generate_pseudotrial_samples(n_pseudo=len(samples), sfreq=sf, n_continuous_samples=c['n'], real_event_samples=all_events, min_gap_seconds=cfg.MIN_GAP_S, tmin=cfg.TMIN_EPOCH, tmax=cfg.TMAX_EPOCH, rng=rng, forbidden=fb, scratch=scratch)
        s = np.asarray(s, np.int64)
        if s.size:
            s = s[(s + lo >= 0) & (s + hi < c['n'])]
            s = s[ptp[s] <= cfg.REJECT_THRESHOLD]
        npd.append(int(s.size))
        if s.size < cfg.MIN_TRIALS_REQUIRED:
            for key in acc:
                acc[key].append(np.nan)
            continue
        lab = np.zeros(s.size, dtype=int)
        lab[rng.permutation(s.size)[:int(round(share * s.size))]] = 1
        for r in PSEUDO_REFS:
            fzs, pzs = sig[r]
            fz, pz, times = C.epochs(fzs, pzs, s, sf)
            f = fastfeat.window_features(fz, pz, times, cfg, want_entropy=True)
            sl = C.slopes(f, models, lab)
            for m in models:
                acc[(r, m.id)].append(sl[m.id][0])
            for col in cols:
                x = np.asarray(f[col], float)
                scl[(r, col)].append(float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))))
    valid = [x for x in npd if x >= cfg.MIN_TRIALS_REQUIRED]
    msub = int(np.median(valid)) if valid else 0
    srng = np.random.default_rng(SUB_SEED)
    subsets = [np.sort(srng.choice(len(samples), size=msub, replace=False)) for _ in range(R_SUB)] if cfg.MIN_TRIALS_REQUIRED <= msub < len(samples) else []
    rows = []
    for r in sig:
        sub_sl = {m.id: [] for m in models}
        for ix in subsets:
            sl = C.slopes({col: feats[r][col][ix] for col in cols}, models, is_t[ix])
            for m in models:
                sub_sl[m.id].append(sl[m.id][0])
        rmad = {col: 1.4826 * np.nanmedian(np.abs(feats[r][col] - np.nanmedian(feats[r][col]))) for col in cols}
        for m in models:
            v = np.array(acc[(r, m.id)], float) if r in PSEUDO_REFS else np.array([np.nan])
            nv = int(np.isfinite(v).sum())
            w = np.array(sub_sl[m.id], float)
            row = dict(subject=sub, reference=r, model_id=m.id, real_slope=real[r][m.id][0], real_slope_shifted=real_sh[r][m.id][0], real_slope_subsampled=float(np.nanmean(w)) if w.size else np.nan, subsample_size=msub, pseudo_mean=float(np.nanmean(v)) if nv else np.nan, pseudo_sd_draws=float(np.nanstd(v, ddof=1)) if nv > 1 else np.nan, n_draws_valid=nv, n_real=len(samples), mad_x_real=float(rmad[m.predictor]), mad_y_real=float(rmad[m.outcome]))
            if r in PSEUDO_REFS:
                row['mad_x_pseudo'] = float(np.nanmean(scl[(r, m.predictor)]))
                row['mad_y_pseudo'] = float(np.nanmean(scl[(r, m.outcome)]))
            rows.append(row)
    info['status'] = 'ok'
    info['pseudo_per_draw_median'] = float(np.median(npd))
    res = dict(rows=pd.DataFrame(rows), info=info, pc=pd.DataFrame(pc))
    pd.to_pickle(res, cp)
    return res, 'computed', time.time() - t0


def collect(a):
    cfg.ensure_dirs()
    fp = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{cfg.variant_tag()}_log.csv')
    if not os.path.exists(fp):
        print(f'subject list not found: {fp}')
        return 2
    subs = sorted(pd.read_csv(fp).subject.unique())
    if a.subjects:
        want = {x.strip().replace('sub-', '').zfill(3) for x in a.subjects.replace(';', ',').split(',') if x.strip()}
        subs = [s for s in subs if s.replace('sub-', '') in want]
    cache_dir = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    state = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper() and isinstance(getattr(cfg, k), (int, float, str, bool, tuple, list, dict, type(None)))}
    tasks = [(s, a.dataset, a.k, cache_dir, state) for s in subs]
    jobs = max(1, a.jobs)
    it = ProcessPoolExecutor(max_workers=jobs).map(subject, tasks) if jobs > 1 else map(subject, tasks)
    rows, infos, pcs = [], [], []
    for i, (res, what, dt) in enumerate(it, 1):
        infos.append(res['info'])
        if res['rows'] is not None:
            rows.append(res['rows'])
            pcs.append(res['pc'])
        print(f"  [{i}/{len(subs)}] {tasks[i - 1][0]}: {res['info'].get('status')} | {what} {dt:.0f}s")
    if not rows:
        print('no subject processed')
        return 1
    R = pd.concat(rows, ignore_index=True)
    mind = int(np.ceil(a.k * cfg.MIN_PSEUDO_DRAWS / 300.0))
    ok = R[R.reference.isin(PSEUDO_REFS)].groupby('subject').n_draws_valid.min() >= mind
    R['used'] = R.subject.map(ok).fillna(False).astype(bool)
    s8 = os.path.join(cfg.RESULTS_DIR, f'common_persubject_{a.dataset}_free.csv')
    if os.path.exists(s8):
        q8 = pd.read_csv(s8)[['subject', 'reference', 'model_id', 'pseudo_mean']].rename(columns={'pseudo_mean': 'pseudo_mean_s8'})
        R = R.merge(q8, on=['subject', 'reference', 'model_id'], how='left')
        fill = ~R.reference.isin(PSEUDO_REFS)
        R.loc[fill, 'pseudo_mean'] = R.loc[fill, 'pseudo_mean_s8']
        R['pseudo_source'] = np.where(fill, 'step8', 'step10')
        R = R.drop(columns='pseudo_mean_s8')
    R['d'] = R.real_slope - R.pseudo_mean
    R['d_shifted'] = R.real_slope_shifted - R.pseudo_mean
    R['d_subsampled'] = R.real_slope_subsampled - R.pseudo_mean
    R.to_csv(os.path.join(cfg.RESULTS_DIR, f'reffree_persubject_{a.dataset}.csv'), index=False)
    pd.DataFrame(infos).to_csv(os.path.join(cfg.RESULTS_DIR, f'reffree_subjects_{a.dataset}.csv'), index=False)
    pd.concat(pcs, ignore_index=True).to_csv(os.path.join(cfg.RESULTS_DIR, f'reffree_poscontrol_{a.dataset}.csv'), index=False)
    v = validate(cfg.RESULTS_DIR, a.dataset, R)
    v.to_csv(os.path.join(cfg.RESULTS_DIR, f'reffree_validation_{a.dataset}.csv'), index=False)
    print(v.to_string(index=False))
    return 0


def validate(results, dataset, R):
    p = os.path.join(results, f'common_persubject_{dataset}_free.csv')
    if not os.path.exists(p):
        return pd.DataFrame([dict(dataset=dataset, check='step8_file_missing', n_pairs=0, max_abs_diff=np.nan)])
    a = pd.read_csv(p)
    m = a.merge(R.drop(columns=[c for c in ('pseudo_source',) if c in R]), on=['subject', 'reference', 'model_id'], suffixes=('_s8', '_s10'))
    out = [dict(dataset=dataset, check='real_slope_all_references', n_pairs=len(m), max_abs_diff=float((m.real_slope_s8 - m.real_slope_s10).abs().max()))]
    mm = m[m.reference == 'mastoid']
    out.append(dict(dataset=dataset, check='pseudo_mean_mastoid', n_pairs=len(mm), max_abs_diff=float((mm.pseudo_mean_s8 - mm.pseudo_mean_s10).abs().max())))
    return pd.DataFrame(out)


def profiles(p, subs, q, models):
    w = p.pivot_table(index='subject', columns='model_id', values=q, aggfunc='first')
    return w.reindex(index=subs, columns=models).values


def analyse(results):
    S, ok = {}, {}
    for ds in ('erpcore', 'ds006018'):
        p = os.path.join(results, f'reffree_persubject_{ds}.csv')
        if not os.path.exists(p):
            print(f'missing {p}; run step 10 for both datasets first')
            return 2
        S[ds] = pd.read_csv(p)
        S[ds] = S[ds][S[ds].used]
    models = sorted(m for m in S['erpcore'].model_id.unique() if m not in SAME)
    subs = {ds: sorted(S[ds].subject.unique()) for ds in S}
    refs = {ds: sorted(S[ds].reference.unique()) for ds in S}
    qs = ['real_slope', 'pseudo_mean', 'd', 'real_slope_shifted', 'd_shifted', 'real_slope_subsampled', 'd_subsampled']
    A = {}
    for ds in S:
        for r in refs[ds]:
            g = S[ds][S[ds].reference == r]
            for q in qs:
                A[(ds, r, q)] = profiles(g, subs[ds], q, models)
    rng = np.random.default_rng(SEED)
    ix = {ds: [rng.integers(0, len(subs[ds]), len(subs[ds])) for _ in range(B_BOOT)] for ds in S}
    def mean_prof(key, i=None):
        X = A[key]
        return np.nanmean(X if i is None else X[i], axis=0)
    def r_(a, b):
        ok_ = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok_], b[ok_])[0, 1]) if ok_.sum() > 2 and np.std(a[ok_]) > 0 and np.std(b[ok_]) > 0 else np.nan
    pairs_x = {'distributed': ('distributed', 'distributed'), 'cz': ('cz', 'distributed'), 'mastoid': ('mastoid', 'mastoid'), 'average': ('average', 'average'), 'csd': ('csd', 'csd'), 'csd18': ('csd18', 'csd18')}
    rows = []
    for q in qs:
        if not all((('erpcore', e, q) in A and ('ds006018', d, q) in A) for e, d in pairs_x.values()):
            continue
        obs = {p: r_(mean_prof(('erpcore', e, q)), mean_prof(('ds006018', d, q))) for p, (e, d) in pairs_x.items()}
        bt = {p: np.array([r_(mean_prof(('erpcore', e, q), ie), mean_prof(('ds006018', d, q), idd)) for ie, idd in zip(ix['erpcore'], ix['ds006018'])]) for p, (e, d) in pairs_x.items()}
        for p in pairs_x:
            rows.append(dict(analysis='cross_dataset', quantity=q, contrast=p, value=obs[p], lo=np.nanpercentile(bt[p], 2.5), hi=np.nanpercentile(bt[p], 97.5)))
        for p in ('csd', 'csd18', 'mastoid', 'average'):
            dd = bt[p] - bt['distributed']
            rows.append(dict(analysis='cross_dataset_change', quantity=q, contrast=f'{p}-distributed', value=obs[p] - obs['distributed'], lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5)))
        ab_obs = float(np.nanmean(np.abs(mean_prof(('erpcore', 'csd', q)) - mean_prof(('ds006018', 'csd', q)))))
        ab_bt = np.array([np.nanmean(np.abs(mean_prof(('erpcore', 'csd', q), ie) - mean_prof(('ds006018', 'csd', q), idd))) for ie, idd in zip(ix['erpcore'], ix['ds006018'])])
        rows.append(dict(analysis='cross_dataset_absdis', quantity=q, contrast='csd', value=ab_obs, lo=np.nanpercentile(ab_bt, 2.5), hi=np.nanpercentile(ab_bt, 97.5)))
    def rx(e, d, q, ie=None, idd=None):
        return r_(mean_prof(('erpcore', e, q), ie), mean_prof(('ds006018', d, q), idd))
    for p in ('csd', 'csd18', 'mastoid', 'average'):
        e, d = pairs_x[p]
        obs_i = (rx(e, d, 'real_slope') - rx('distributed', 'distributed', 'real_slope')) - (rx(e, d, 'pseudo_mean') - rx('distributed', 'distributed', 'pseudo_mean'))
        di = np.array([(rx(e, d, 'real_slope', ie, idd) - rx('distributed', 'distributed', 'real_slope', ie, idd)) - (rx(e, d, 'pseudo_mean', ie, idd) - rx('distributed', 'distributed', 'pseudo_mean', ie, idd)) for ie, idd in zip(ix['erpcore'], ix['ds006018'])])
        rows.append(dict(analysis='cross_dataset_interaction', quantity='real-pseudo', contrast=f'{p}-distributed', value=obs_i, lo=np.nanpercentile(di, 2.5), hi=np.nanpercentile(di, 97.5)))
    for ds in S:
        for q in ('real_slope', 'pseudo_mean', 'd'):
            others = [r for r in refs[ds] if r not in ('csd', 'csd18')]
            for r in others + ['csd18']:
                obs = r_(mean_prof((ds, r, q)), mean_prof((ds, 'csd', q)))
                bt = np.array([r_(mean_prof((ds, r, q), i), mean_prof((ds, 'csd', q), i)) for i in ix[ds]])
                rows.append(dict(analysis=f'within_{ds}_vs_csd', quantity=q, contrast=r, value=obs, lo=np.nanpercentile(bt, 2.5), hi=np.nanpercentile(bt, 97.5)))
            for a_, b_ in (('mastoid', 'average'), ('distributed', 'mastoid'), ('distributed', 'average')):
                obs = r_(mean_prof((ds, a_, q)), mean_prof((ds, b_, q)))
                bt = np.array([r_(mean_prof((ds, a_, q), i), mean_prof((ds, b_, q), i)) for i in ix[ds]])
                rows.append(dict(analysis=f'within_{ds}_between_references', quantity=q, contrast=f'{a_}~{b_}', value=obs, lo=np.nanpercentile(bt, 2.5), hi=np.nanpercentile(bt, 97.5)))
        for r in [x for x in refs[ds] if x not in ('csd', 'csd18')]:
            obs = r_(mean_prof((ds, r, 'd')), mean_prof((ds, 'csd', 'd'))) - r_(mean_prof((ds, r, 'real_slope')), mean_prof((ds, 'csd', 'real_slope')))
            bt = np.array([r_(mean_prof((ds, r, 'd'), i), mean_prof((ds, 'csd', 'd'), i)) - r_(mean_prof((ds, r, 'real_slope'), i), mean_prof((ds, 'csd', 'real_slope'), i)) for i in ix[ds]])
            rows.append(dict(analysis=f'within_{ds}_residual_minus_raw_vs_csd', quantity='d-real', contrast=r, value=obs, lo=np.nanpercentile(bt, 2.5), hi=np.nanpercentile(bt, 97.5)))
    pd.DataFrame(rows).to_csv(os.path.join(results, 'reffree_agreement.csv'), index=False)
    db = []
    for ds in S:
        for r in refs[ds]:
            g = S[ds][S[ds].reference == r]
            for mid, h in g.groupby('model_id'):
                for q in ('real_slope', 'pseudo_mean', 'd', 'd_shifted', 'd_subsampled'):
                    x = h[q].values
                    x = x[np.isfinite(x)]
                    if len(x) < 3:
                        continue
                    est, lo, hi = E.bca_ci(x, B=B_BCA, seed=SEED)
                    bm = np.array([x[i].mean() for i in np.random.default_rng(SEED).integers(0, len(x), (B_BCA, len(x)))])
                    pb = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)) + 1.0 / B_BCA)
                    db.append(dict(dataset=ds, reference=r, model_id=mid, quantity=q, n=len(x), mean=float(est), bca95_lo=float(lo), bca95_hi=float(hi), p_boot=pb, scale_ratio_y=float((h.mad_y_real / h.mad_y_pseudo).median()) if 'mad_y_pseudo' in h and h.mad_y_pseudo.notna().any() else np.nan, scale_ratio_x=float((h.mad_x_real / h.mad_x_pseudo).median()) if 'mad_x_pseudo' in h and h.mad_x_pseudo.notna().any() else np.nan))
    D = pd.DataFrame(db)
    if not D.empty:
        D['p_holm'] = np.nan
        for (ds, r, q), g in D.groupby(['dataset', 'reference', 'quantity']):
            D.loc[g.index, 'p_holm'] = holm(g.p_boot.values)
    D.to_csv(os.path.join(results, 'reffree_dbeta.csv'), index=False)
    pc = []
    for ds in S:
        p = pd.read_csv(os.path.join(results, f'reffree_poscontrol_{ds}.csv'))
        p = p[p.subject.isin(subs[ds])]
        for r, g in p.groupby('reference'):
            for col in ('target_minus_standard', 'target_minus_standard_shifted'):
                x = g[col].values
                f = 1.0 if r.startswith('csd') else 1e6
                pc.append(dict(dataset=ds, reference=r, quantity=col, unit='V/m^2' if r.startswith('csd') else 'uV', n=len(x), mean=float(np.mean(x) * f), sd=float(np.std(x, ddof=1) * f), dz=float(np.mean(x) / np.std(x, ddof=1)), n_positive=int((x > 0).sum())))
    pd.DataFrame(pc).to_csv(os.path.join(results, 'reffree_poscontrol_summary.csv'), index=False)
    print(f'-> {results}')
    return 0


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description='Step 10: reference-free (surface Laplacian, CSD) coupling on the step-8 common cleaning, with identical trials and pseudotrial placements; monitor-delay-corrected real trials; real trials subsampled to the pseudotrial count; positive control under every derivation.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018', 'all'])
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--k', type=int, default=300)
    ap.add_argument('--jobs', type=int, default=1)
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    if cfg.RESAMPLE_HZ or cfg.REREF != 'distributed' or not cfg.APPLY_BASELINE:
        print('step 10 derives every reference itself; run it without --reref, --resample or --no-baseline')
        return 2
    if a.dataset == 'all':
        return analyse(cfg.RESULTS_DIR)
    cfg.banner(f'step 10  reference-free  |  dataset={a.dataset}  K={a.k}')
    return collect(a)


if __name__ == '__main__':
    sys.exit(main())
