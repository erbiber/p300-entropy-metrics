import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import config as cfg
import estimators as E
import placement as P
from lib import io_data as IO, preprocess as PP, fastfeat

MASTOID = {'erpcore': ('P9', 'P10'), 'ds006018': ('LM', 'RM')}
NON_SCALP = {'erpcore': (), 'ds006018': ('LM', 'RM')}
ONLINE = {'ds006018': 'Cz'}
REFS = {'erpcore': ('distributed', 'cz', 'mastoid', 'average'), 'ds006018': ('distributed', 'mastoid', 'average')}
MODES = {'free': (0.8, 1.2), 'onset': None}
VERSION = 'c1'


def load_raw(sid, dataset):
    if dataset == 'erpcore':
        return IO.load_erpcore(sid, cfg)
    return IO.load_ds006018(sid, cfg)


def battery():
    import spec
    return list(spec.MATCHED)


def clean_continuous(sub, dataset):
    import mne
    raw, err = load_raw(sub.replace('sub-', ''), dataset)
    if err:
        return (None, err)
    if int(raw.first_samp) != 0:
        return (None, f'first_samp_{raw.first_samp}_nonzero')
    raw, _ = IO.prepare_channel_types(raw)
    mast = [c for c in MASTOID[dataset] if c in raw.ch_names]
    if len(mast) != 2:
        return (None, 'mastoid_channels_missing')
    raw.set_channel_types({c: 'eeg' for c in mast}, verbose=False)
    raw = PP.bandpass(raw, cfg)
    raw, n_comp, n_exc = PP.run_ica(raw, cfg)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_of = {}
    for desc, val in event_id.items():
        m = re.search('(\\d+)\\s*$', str(desc))
        code_of[int(val)] = int(m.group(1)) if m else None
    eeg = mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
    names = [raw.ch_names[i] for i in eeg]
    V = raw.get_data()[eeg]
    online = ONLINE.get(dataset)
    if online and online not in names:
        V = np.vstack([V, np.zeros((1, V.shape[1]))])
        names = names + [online]
    for c in ('Fz', 'Pz'):
        if c not in names:
            return (None, f'missing_{c}')
    return (dict(V=V, names=names, sfreq=float(raw.info['sfreq']), events=events, code_of=code_of, n=V.shape[1], n_ica=int(n_comp), n_excluded=int(n_exc), mast=mast), None)


def reference_signals(c, dataset):
    V, names = c['V'], c['names']
    idx = {n: i for i, n in enumerate(names)}
    scalp = [i for i, n in enumerate(names) if n not in NON_SCALP[dataset]]
    out = {}
    for r in REFS[dataset]:
        if r == 'distributed':
            out[r] = np.zeros(c['n'])
        elif r == 'cz':
            if 'Cz' not in idx:
                return (None, 'cz_missing')
            out[r] = V[idx['Cz']].copy()
        elif r == 'mastoid':
            out[r] = V[[idx[m] for m in MASTOID[dataset]]].mean(axis=0)
        elif r == 'average':
            out[r] = V[scalp].mean(axis=0)
    return (out, scalp)


def combined_ptp(c, refs, scalp):
    sf = c['sfreq']
    lo = int(round(cfg.TMIN_EPOCH * sf))
    nw = int(round(cfg.TMAX_EPOCH * sf)) - lo + 1
    worst = None
    per = {}
    for r, s in refs.items():
        X = c['V'][scalp] - s[None, :]
        p = PP.rolling_ptp_max(X, nw, lo)
        per[r] = p
        worst = p if worst is None else np.maximum(worst, p)
    return (worst, per)


def epochs(sig_fz, sig_pz, samples, sf):
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    idx = samples[:, None] + np.arange(lo, hi + 1)[None, :]
    fz = sig_fz[idx]
    pz = sig_pz[idx]
    times = np.arange(lo, hi + 1) / sf
    if cfg.APPLY_BASELINE:
        bm = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[1])
        fz = fz - fz[:, bm].mean(axis=1, keepdims=True)
        pz = pz - pz[:, bm].mean(axis=1, keepdims=True)
    return (fz, pz, times)


def slopes(f, models, is_t):
    z = {}
    out = {}
    cols = sorted({m.predictor for m in models} | {m.outcome for m in models})
    for col in cols:
        z[col] = E.robust_z_within_subject(f[col])[0]
    g = np.asarray(is_t).astype(bool)
    for m in models:
        x, y = z[m.predictor], z[m.outcome]
        xa = x.copy()
        ya = y.copy()
        for flag in (True, False):
            k = g == flag
            if k.sum() > 0:
                xa[k] = x[k] - np.nanmean(x[k])
                ya[k] = y[k] - np.nanmean(y[k])
        out[m.id] = (E.ols_slope(x, y), E.ols_slope(xa, ya), E.ols_slope(x[g], y[g]) if g.sum() >= 3 else np.nan, E.ols_slope(x[~g], y[~g]) if (~g).sum() >= 3 else np.nan)
    return out


def forbidden(n, events, sf, mode):
    if MODES[mode] is None:
        return P.forbidden_mask(n, events, cfg.MIN_GAP_S, sf)
    before, after = MODES[mode]
    fb = np.zeros(n, dtype=bool)
    b = int(round(before * sf))
    a = int(round(after * sf))
    for ev in events:
        fb[max(0, int(ev) - b):min(n, int(ev) + a + 1)] = True
    return fb


def free_seconds(fb, sf):
    lo = int(round(-cfg.TMIN_EPOCH * sf)) + 1
    hi = int(round(cfg.TMAX_EPOCH * sf)) + 1
    ok = ~fb.copy()
    ok[:lo] = False
    ok[len(ok) - hi:] = False
    return float(ok.sum() / sf)


def subject(args):
    sub, dataset, K, modes, cache_dir, state = args
    for k, v in state.items():
        setattr(cfg, k, v)
    models = battery()
    t0 = time.time()
    cpath = {m: os.path.join(cache_dir, f'common_{VERSION}_{dataset}_{m}_K{K[m]}_{sub}.pkl') for m in modes}
    if all(os.path.exists(p) for p in cpath.values()):
        return {m: pd.read_pickle(cpath[m]) for m in modes}, None, 'cached', 0.0
    c, err = clean_continuous(sub, dataset)
    if err:
        return (None, dict(subject=sub, status=err), '', time.time() - t0)
    refs, scalp = reference_signals(c, dataset)
    if refs is None:
        return (None, dict(subject=sub, status=scalp), '', time.time() - t0)
    ptp, ptp_per = combined_ptp(c, refs, scalp)
    sf = c['sfreq']
    idx = {n: i for i, n in enumerate(c['names'])}
    sig = {r: (c['V'][idx['Fz']] - s, c['V'][idx['Pz']] - s) for r, s in refs.items()}
    targets, standards = cfg.codes_for(dataset)
    keep = set(targets + standards)
    ev = c['events']
    codes = np.array([c['code_of'].get(int(v)) for v in ev[:, 2]], dtype=object)
    sel = np.array([x is not None and x in keep for x in codes])
    samples = ev[sel, 0].astype(np.int64)
    codes = codes[sel]
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    inb = (samples + lo >= 0) & (samples + hi < c['n'])
    samples, codes = samples[inb], codes[inb]
    n_stim = len(samples)
    rej_by = {r: int((p[samples] > cfg.REJECT_THRESHOLD).sum()) for r, p in ptp_per.items()}
    ok = ptp[samples] <= cfg.REJECT_THRESHOLD
    samples, codes = samples[ok], codes[ok]
    is_t = np.array([int(x in targets) for x in codes])
    info = dict(subject=sub, n_ica=c['n_ica'], n_ica_excluded=c['n_excluded'], n_stimuli=n_stim, n_real=len(samples), n_target=int(is_t.sum()), n_standard=int(len(is_t) - is_t.sum()), **{f'rejected_under_{r}': v for r, v in rej_by.items()})
    if is_t.sum() < cfg.MIN_TRIALS_PER_CONDITION or (len(is_t) - is_t.sum()) < cfg.MIN_TRIALS_PER_CONDITION or len(samples) < cfg.MIN_TRIALS_REQUIRED:
        info['status'] = 'below_minimum_trials'
        return (None, info, '', time.time() - t0)
    real = {}
    condiff = []
    for r, (fzs, pzs) in sig.items():
        fz, pz, times = epochs(fzs, pzs, samples, sf)
        f = fastfeat.window_features(fz, pz, times, cfg, want_entropy=True)
        real[r] = slopes(f, models, is_t)
        for col in sorted({m.predictor for m in models}):
            zc = E.robust_z_within_subject(f[col])[0]
            raw_ = np.asarray(f[col], float)
            condiff.append(dict(subject=sub, reference=r, predictor=col, z_target_minus_standard=float(np.nanmean(zc[is_t == 1]) - np.nanmean(zc[is_t == 0])), raw_target=float(np.nanmean(raw_[is_t == 1])), raw_standard=float(np.nanmean(raw_[is_t == 0]))))
    all_events = ev[:, 0].astype(np.int64)
    share = is_t.mean()
    results = {}
    for mode in modes:
        fb = forbidden(c['n'], all_events, sf, mode)
        info[f'candidate_seconds_{mode}'] = free_seconds(fb, sf)
        rng = np.random.default_rng(cfg.PSEUDOTRIAL_SEED)
        scratch = np.zeros(c['n'], dtype=bool)
        acc = {(r, m.id): [] for r in sig for m in models}
        npd = []
        for k in range(K[mode]):
            s = P.generate_pseudotrial_samples(n_pseudo=len(samples), sfreq=sf, n_continuous_samples=c['n'], real_event_samples=all_events, min_gap_seconds=cfg.MIN_GAP_S, tmin=cfg.TMIN_EPOCH, tmax=cfg.TMAX_EPOCH, rng=rng, forbidden=fb, scratch=scratch)
            s = np.asarray(s, np.int64)
            if s.size:
                s = s[(s + lo >= 0) & (s + hi < c['n'])]
                s = s[ptp[s] <= cfg.REJECT_THRESHOLD]
            npd.append(int(s.size))
            if k == 0 and s.size:
                srt = np.sort(all_events)
                ip = np.searchsorted(srt, s, side='right') - 1
                gp = np.where(ip >= 0, (s + lo) - srt[np.clip(ip, 0, None)], np.nan) / sf
                inx = np.searchsorted(srt, s, side='left')
                gn = np.where(inx < srt.size, srt[np.clip(inx, None, srt.size - 1)] - (s + hi), np.nan) / sf
                info[f'gap_epoch_start_after_event_median_{mode}'] = float(np.nanmedian(gp)) if np.isfinite(gp).any() else np.nan
                info[f'gap_epoch_start_after_event_min_{mode}'] = float(np.nanmin(gp)) if np.isfinite(gp).any() else np.nan
                info[f'gap_epoch_end_before_event_min_{mode}'] = float(np.nanmin(gn)) if np.isfinite(gn).any() else np.nan
                info[f'share_with_event_in_epoch_{mode}'] = float(np.mean([np.any((srt >= x + lo) & (srt <= x + hi)) for x in s]))
            if s.size < cfg.MIN_TRIALS_REQUIRED:
                for key in acc:
                    acc[key].append((np.nan, np.nan))
                continue
            lab = np.zeros(s.size, dtype=int)
            lab[rng.permutation(s.size)[:int(round(share * s.size))]] = 1
            for r, (fzs, pzs) in sig.items():
                fz, pz, times = epochs(fzs, pzs, s, sf)
                f = fastfeat.window_features(fz, pz, times, cfg, want_entropy=True)
                sl = slopes(f, models, lab)
                for m in models:
                    acc[(r, m.id)].append((sl[m.id][0], sl[m.id][1]))
        rows = []
        for r in sig:
            for m in models:
                v = np.array(acc[(r, m.id)], float)
                nv = int(np.isfinite(v[:, 0]).sum())
                rr = real[r][m.id]
                rows.append(dict(subject=sub, reference=r, model_id=m.id, real_slope=rr[0], real_slope_cadj=rr[1], real_slope_target=rr[2], real_slope_standard=rr[3], pseudo_mean=float(np.nanmean(v[:, 0])) if nv else np.nan, pseudo_mean_cadj=float(np.nanmean(v[:, 1])) if nv else np.nan, n_draws_valid=nv, pseudo_per_draw_median=float(np.median(npd)), pseudo_per_draw_min=int(min(npd)), n_real=len(samples)))
        res = dict(rows=pd.DataFrame(rows), info=info, condiff=pd.DataFrame(condiff))
        pd.to_pickle(res, cpath[mode])
        results[mode] = res
    return (results, None, 'computed', time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description='Step 8: one ICA cleaning, references derived after cleaning, one real-trial set and one pseudotrial set for every reference; event-free pseudotrials; condition-adjusted slopes.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018'])
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--modes', default='free,onset')
    ap.add_argument('--k-free', type=int, default=300)
    ap.add_argument('--k-onset', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=1)
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    if cfg.RESAMPLE_HZ or cfg.REREF != 'distributed' or not cfg.APPLY_BASELINE:
        print('step 8 derives every reference itself; run it without --reref, --resample or --no-baseline')
        return 2
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f'unknown mode {m}')
            return 2
    K = {'free': a.k_free, 'onset': a.k_onset}
    tag = cfg.variant_tag()
    cfg.banner(f'step 8  common cleaning  |  dataset={a.dataset}  variant={tag}  modes={modes}  K={[K[m] for m in modes]}')
    fp = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}_log.csv')
    if os.path.exists(fp):
        subs = sorted(pd.read_csv(fp).subject.unique())
    else:
        print(f'subject list not found: {fp}\nrun step 1 for this dataset first')
        return 2
    if a.subjects:
        want = {x.strip().replace('sub-', '').zfill(3) for x in a.subjects.replace(';', ',').split(',') if x.strip()}
        subs = [s for s in subs if s.replace('sub-', '') in want]
    cache_dir = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    state = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper() and isinstance(getattr(cfg, k), (int, float, str, bool, tuple, list, dict, type(None)))}
    tasks = [(s, a.dataset, K, modes, cache_dir, state) for s in subs]
    got = {m: [] for m in modes}
    infos = []
    t0 = time.time()
    jobs = max(1, a.jobs)
    it = ProcessPoolExecutor(max_workers=jobs).map(subject, tasks) if jobs > 1 else map(subject, tasks)
    for i, (res, bad, what, dt) in enumerate(it, 1):
        sub = tasks[i - 1][0]
        if res is None:
            infos.append(bad)
            print(f'  [{i}/{len(subs)}] {sub}: EXCLUDED {bad.get("status")}')
            continue
        for m in modes:
            got[m].append(res[m])
        inf = dict(res[modes[0]]['info'])
        inf['status'] = 'ok'
        for m in modes:
            inf[f'candidate_seconds_{m}'] = res[m]['info'].get(f'candidate_seconds_{m}')
            inf[f'pseudo_per_draw_median_{m}'] = float(res[m]['rows'].pseudo_per_draw_median.iloc[0])
            for kk, vv in res[m]['info'].items():
                if kk.startswith(('gap_epoch', 'share_with_event')):
                    inf[kk] = vv
        infos.append(inf)
        print(f'  [{i}/{len(subs)}] {sub}: {inf["n_real"]} real trials, ICA removed {inf["n_ica_excluded"]} | ' + ' | '.join(f'{m}: {inf[f"pseudo_per_draw_median_{m}"]:.0f} pseudotrials/draw, {inf[f"candidate_seconds_{m}"]:.0f} s eligible' for m in modes) + f' | {what} {dt:.0f}s')
    if not any(got.values()):
        print('no subject processed')
        return 1
    out = []
    for m in modes:
        rows = pd.concat([g['rows'] for g in got[m]], ignore_index=True)
        mind = int(np.ceil(K[m] * cfg.MIN_PSEUDO_DRAWS / 300.0))
        good = (rows.n_draws_valid >= mind) & np.isfinite(rows.real_slope)
        okr = good.groupby(rows.subject).all()
        rows['used'] = rows.subject.map(okr)
        rows['d'] = rows.real_slope - rows.pseudo_mean
        rows['d_cadj'] = rows.real_slope_cadj - rows.pseudo_mean_cadj
        rows['dataset'] = a.dataset
        rows['mode'] = m
        p = os.path.join(cfg.RESULTS_DIR, f'common_persubject_{a.dataset}_{m}.csv')
        rows.to_csv(p, index=False)
        out.append(p)
        print(f'{m}: {int(okr.sum())} of {len(okr)} subjects used under every reference (>= {mind} valid draws)')
    cd = pd.concat([g['condiff'] for g in got[modes[0]]], ignore_index=True)
    p = os.path.join(cfg.RESULTS_DIR, f'common_condition_{a.dataset}.csv')
    cd.to_csv(p, index=False)
    out.append(p)
    p = os.path.join(cfg.RESULTS_DIR, f'common_subjects_{a.dataset}.csv')
    pd.DataFrame(infos).to_csv(p, index=False)
    out.append(p)
    print(f'\n{time.time() - t0:.0f}s')
    for p in out:
        print(f'-> {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
