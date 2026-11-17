import argparse
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
import spec
from lib import fastfeat, io_data as IO, preprocess as PP
from lib.features import window_mask

def epoch_at(raw_data, samples, sfreq, cfg, fz_i, pz_i):
    lo = int(round(cfg.TMIN_EPOCH * sfreq))
    hi = int(round(cfg.TMAX_EPOCH * sfreq))
    n = hi - lo + 1
    keep = [s for s in samples if s + lo >= 0 and s + hi < raw_data.shape[1]]
    if not keep:
        return (None, None, None)
    idx = np.array([np.arange(s + lo, s + lo + n) for s in keep])
    fz = raw_data[fz_i][idx]
    pz = raw_data[pz_i][idx]
    times = (np.arange(n) + lo) / sfreq
    if cfg.APPLY_BASELINE:
        bm = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[1])
        if bm.any():
            fz = fz - fz[:, bm].mean(axis=1, keepdims=True)
            pz = pz - pz[:, bm].mean(axis=1, keepdims=True)
    return (fz, pz, times)

def resample_windows(fz, pz, times, cfg, sfreq):
    if not cfg.RESAMPLE_HZ or abs(sfreq - cfg.RESAMPLE_HZ) < 1e-09:
        return (fz, pz, times)
    import mne
    fz = mne.filter.resample(fz, up=cfg.RESAMPLE_HZ, down=sfreq, axis=-1, npad='auto', window='auto', pad='edge', method='fft', verbose=False)
    pz = mne.filter.resample(pz, up=cfg.RESAMPLE_HZ, down=sfreq, axis=-1, npad='auto', window='auto', pad='edge', method='fft', verbose=False)
    times = np.arange(fz.shape[-1]) / cfg.RESAMPLE_HZ + times[0]
    return (fz, pz, times)

def reject_mask(ptp_at_samples, thresh):
    return ptp_at_samples <= thresh

def subject_pseudo_slopes(sub, cont, real_feats, models, cfg, rng, K, profile=False):
    sfreq = cont['sfreq']
    ncont = cont['n']
    real_samples = cont['events']
    data2 = np.vstack([cont['fz'], cont['pz']])
    n_real = len(real_feats)
    need_cols = sorted({m.predictor for m in models} | {m.outcome for m in models})
    ptp_all = cont['ptp']
    fb = P.forbidden_mask(ncont, real_samples, cfg.MIN_GAP_S, sfreq)
    scratch = np.zeros(ncont, dtype=bool)
    out = {m.id: [] for m in models}
    need_ent = any(c in need_cols for c in ('pe_early_fz', 'se_early_fz', 'lz_early_fz'))
    need_par = any(c.startswith('par_') for c in need_cols)
    meta_rows = []
    prof = dict(place=0.0, epoch=0.0, reject=0.0, feat_amp=0.0, feat_ent=0.0, fit=0.0)
    for k in range(K):
        t = time.perf_counter()
        samples = P.generate_pseudotrial_samples(n_pseudo=n_real, sfreq=sfreq, n_continuous_samples=ncont, real_event_samples=real_samples, min_gap_seconds=cfg.MIN_GAP_S, tmin=cfg.TMIN_EPOCH, tmax=cfg.TMAX_EPOCH, rng=rng, forbidden=fb, scratch=scratch)
        prof['place'] += time.perf_counter() - t
        if samples.size == 0:
            for m in models:
                out[m.id].append(np.nan)
            continue
        t = time.perf_counter()
        fz, pz, times = epoch_at(data2, samples, sfreq, cfg, 0, 1)
        prof['epoch'] += time.perf_counter() - t
        if fz is None:
            for m in models:
                out[m.id].append(np.nan)
            continue
        t = time.perf_counter()
        keep = reject_mask(ptp_all[samples], cfg.REJECT_THRESHOLD)
        prof['reject'] += time.perf_counter() - t
        if keep.sum() < cfg.MIN_TRIALS_REQUIRED:
            for m in models:
                out[m.id].append(np.nan)
            continue
        if k == 0:
            for r in P.placement_metadata(samples[keep], real_samples, sfreq, cfg.TMIN_EPOCH, cfg.TMAX_EPOCH):
                r['subject'] = sub
                meta_rows.append(r)
        if profile:
            t = time.perf_counter()
            fzk, pzk, tk = resample_windows(fz[keep], pz[keep], times, cfg, sfreq)
            f = fastfeat.window_features(fzk, pzk, tk, cfg, want_entropy=False, want_parity=need_par)
            prof['feat_amp'] += time.perf_counter() - t
            t = time.perf_counter()
            pe, se, lz = fastfeat.entropies(fzk[:, fastfeat.window_mask_early(tk, cfg)])
            prof['feat_ent'] += time.perf_counter() - t
            f['pe_early_fz'], f['se_early_fz'], f['lz_early_fz'] = (pe, se, lz)
        else:
            t = time.perf_counter()
            fzk, pzk, tk = resample_windows(fz[keep], pz[keep], times, cfg, sfreq)
            f = fastfeat.window_features(fzk, pzk, tk, cfg, want_entropy=need_ent, want_parity=need_par)
            prof['feat_amp'] += time.perf_counter() - t
        t = time.perf_counter()
        zc = {c: E.robust_z_within_subject(f[c])[0] for c in need_cols}
        for m in models:
            out[m.id].append(E.ols_slope(zc[m.predictor], zc[m.outcome]))
        prof['fit'] += time.perf_counter() - t
    if profile:
        tot = sum(prof.values())
        print(f'      profile over {K} draws, {n_real} windows each:')
        for key, v in sorted(prof.items(), key=lambda r: -r[1]):
            if v > 0:
                print(f'        {key:10s} {1000 * v / K:8.1f} ms/draw  {100 * v / tot:5.1f}%')
        print(f"        {'TOTAL':10s} {1000 * tot / K:8.1f} ms/draw")
    return ({k: np.asarray(v, float) for k, v in out.items()}, meta_rows, None)
METHOD_VERSION = 'm4'
_PLAIN = (int, float, str, bool, tuple, list, dict, type(None))

def config_state(mod):
    return {k: getattr(mod, k) for k in dir(mod)
            if k.isupper() and not k.startswith('_') and isinstance(getattr(mod, k), _PLAIN)}

def family_models(sp, family):
    return list(sp.PARITY) if family == 'parity' else list(sp.MATCHED)

def family_suffix(family):
    return '_parity' if family == 'parity' else ''

def merge_parity(feats, path):
    if not os.path.exists(path):
        return (None, f'parity features file not found: {path}\nrun step 5 with the same options first')
    par = pd.read_csv(path)
    if 'status' in par.columns:
        par = par[par.status == 'ok']
    keys = ['subject', 'trial_idx']
    cols = [c for c in par.columns if c.startswith('par_')]
    m = feats.merge(par[keys + cols], on=keys, how='left', validate='one_to_one')
    lost = m[cols[0]].isna() & feats.subject.isin(par.subject.unique()).values
    if lost.any():
        return (None, f'{int(lost.sum())} real trials have no parity row although their subject is in the parity file')
    m = m[m.subject.isin(par.subject.unique())].reset_index(drop=True)
    return (m, None)

def load_raw(sub, cfg, dataset):
    sid = sub.replace('sub-', '')
    if dataset == 'erpcore':
        return IO.load_erpcore(sid, cfg)
    if dataset == 'ds006018':
        return IO.load_ds006018(sid, cfg)
    return (None, f'unknown_dataset_{dataset}')

def _worker(args):
    sub, state, dataset, K, tag, pre_tag, cache_dir, feat_rows, family = args
    import config as c
    for k, v in state.items():
        setattr(c, k, v)
    import spec as sp
    models = family_models(sp, family)
    g = pd.DataFrame(feat_rows)
    cpath = os.path.join(cache_dir, f'{dataset}_{tag}_K{K}_{METHOD_VERSION}{family_suffix(family)}_{sub}.npz')
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return (sub, {k: z[k] for k in z.files}, [], 'cached', 0.0, 0.0)
    ppath = os.path.join(cache_dir, f'cont_{dataset}_{pre_tag}_{sub}.npz')
    t_pre = time.time()
    if os.path.exists(ppath):
        z = np.load(ppath)
        cont = dict(fz=z['fz'].astype(np.float64), pz=z['pz'].astype(np.float64), sfreq=float(z['sfreq']), events=z['events'], n=int(z['n']), ptp=z['ptp'], n_scalp=int(z['n_scalp']))
        what = 'preproc cached'
    else:
        raw, err = load_raw(sub, c, dataset)
        if err:
            return (sub, None, [], err, 0.0, 0.0)
        raw, _ = IO.prepare_channel_types(raw)
        raw = PP.bandpass(raw, c)
        raw, _ = PP.apply_reference(raw, dataset, c.REREF)
        raw, _, _ = PP.run_ica(raw, c)
        import mne
        ev, _ = mne.events_from_annotations(raw, verbose=False)
        ch = raw.ch_names
        _, fi = PP.get_channel_index(ch, c.EARLY_CHANNEL, c.EARLY_FALLBACK)
        _, pi = PP.get_channel_index(ch, c.P300_CHANNEL, c.P300_FALLBACK)
        if fi is None or pi is None:
            return (sub, None, [], 'missing_channels', 0.0, 0.0)
        d = raw.get_data()
        sf_ = float(raw.info['sfreq'])
        lo_ = int(round(c.TMIN_EPOCH * sf_))
        nw_ = int(round(c.TMAX_EPOCH * sf_)) - lo_ + 1
        import mne as _mne
        scalp = _mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
        ptp = PP.rolling_ptp_max(d[scalp], nw_, lo_)
        cont = dict(fz=d[fi], pz=d[pi], sfreq=float(raw.info['sfreq']), events=ev[:, 0], n=d.shape[1], ptp=ptp, n_scalp=len(scalp))
        np.savez_compressed(ppath, fz=cont['fz'], pz=cont['pz'], sfreq=cont['sfreq'], events=cont['events'], n=cont['n'], ptp=cont['ptp'], n_scalp=cont['n_scalp'])
        what = 'load+filter+ICA'
        del raw, d
    pre_s = time.time() - t_pre
    rng = np.random.default_rng(c.PSEUDOTRIAL_SEED)
    t_dr = time.time()
    ps, meta, err = subject_pseudo_slopes(sub, cont, g, models, c, rng, K)
    dr_s = time.time() - t_dr
    if err:
        return (sub, None, [], err, pre_s, dr_s)
    np.savez_compressed(cpath, **ps)
    return (sub, ps, meta, what, pre_s, dr_s)

def main():
    ap = argparse.ArgumentParser(description='Step 3: matched pseudotrial contrast.')
    ap.add_argument('--dataset', default='erpcore', choices=['erpcore', 'ds006018'])
    ap.add_argument('--k', type=int, default=cfg.N_DRAWS, help='pseudotrial draws')
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--profile', action='store_true', help='report where the time goes inside the draw loop')
    ap.add_argument('--jobs', type=int, default=1, help='subjects to process in parallel (default 1)')
    ap.add_argument('--family', default='battery', choices=['battery', 'parity'], help='model family: the 21-model battery or the time-reversal parity family')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    tag = cfg.variant_tag()
    cfg.banner(f'step 3  matched contrast  |  dataset={a.dataset}  variant={tag}  K={a.k}  family={a.family}')
    fp = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    if not os.path.exists(fp):
        print(f'features file not found: {fp}\nrun step 1 with the same options first')
        return 2
    feats = pd.read_csv(fp)
    if a.family == 'parity':
        feats, perr = merge_parity(feats, os.path.join(cfg.RESULTS_DIR, f'features_parity_{a.dataset}_{tag}.csv'))
        if perr:
            print(perr)
            return 2
    models = family_models(spec, a.family)
    fam = family_suffix(a.family)
    need = sorted({m.predictor for m in models} | {'p300_pz'})
    missing = [c for c in need if c not in feats.columns]
    if missing:
        print(f'features file lacks columns: {missing}')
        return 2
    print(f'{len(feats)} real trials, {feats.subject.nunique()} subjects')
    print(f'{len(models)} models, {a.k} draws, min {cfg.MIN_PSEUDO_DRAWS} valid draws to include a participant\n')
    cache_dir = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    pre_tag = f"v4_hp{cfg.FILTER_LOW:g}-{cfg.FILTER_HIGH:g}_{cfg.ica_tag()}_{cfg.REREF}"
    subs = sorted(feats.subject.unique())
    if a.subjects:
        want = set()
        for x in a.subjects.replace(';', ',').split(','):
            x = x.strip().replace('sub-', '')
            if x:
                want.add(x.zfill(3) if x.isdigit() and len(x) < 3 else x)
        subs = [s for s in subs if s.replace('sub-', '') in want]
        if not subs:
            print(f'none of the requested subjects {sorted(want)} are in the features file')
            return 2
    if a.limit:
        subs = subs[:a.limit]
    rows = []
    meta_all = []
    t0 = time.time()
    if a.jobs > 1:
        state = config_state(cfg)
        tasks = [(s_, state, a.dataset, a.k, tag, pre_tag, cache_dir, feats[feats.subject == s_].to_dict('records'), a.family) for s_ in subs]
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for sub, ps, meta, what, pre_s, dr_s in ex.map(_worker, tasks):
                done += 1
                if ps is None:
                    print(f'  [{done}/{len(subs)}] {sub}: SKIPPED {what}')
                    continue
                meta_all += meta
                g = feats[feats.subject == sub]
                if what == 'cached':
                    print(f'  [{done}/{len(subs)}] {sub}: cached')
                else:
                    print(f'  [{done}/{len(subs)}] {sub}: {len(g)} real trials | preproc {pre_s:.0f}s ({what}) | {a.k} draws {dr_s:.1f}s ({1000 * dr_s / a.k:.0f} ms/draw)')
                for m in models:
                    v = ps[m.id]
                    nv = int(np.isfinite(v).sum())
                    x, sx = E.robust_z_within_subject(g[m.predictor].values)
                    y, sy = E.robust_z_within_subject(g[m.outcome].values)
                    real = E.ols_slope(x, y)
                    rows.append(dict(dataset=a.dataset, variant=tag, model_id=m.id, subject=sub, real_slope=real, pseudo_mean=float(np.nanmean(v)) if nv else np.nan, n_pseudo_draws=nv, used=bool(nv >= cfg.MIN_PSEUDO_DRAWS and np.isfinite(real)), predictor_scaled=sx, outcome_scaled=sy))
        subs = []
    for i, sub in enumerate(subs, 1):
        g = feats[feats.subject == sub]
        cpath = os.path.join(cache_dir, f'{a.dataset}_{tag}_K{a.k}_{METHOD_VERSION}{fam}_{sub}.npz')
        if os.path.exists(cpath) and (not a.no_cache):
            z = np.load(cpath, allow_pickle=True)
            ps = {k: z[k] for k in z.files}
            print(f'  [{i}/{len(subs)}] {sub}: cached')
        else:
            ppath = os.path.join(cache_dir, f'cont_{a.dataset}_{pre_tag}_{sub}.npz')
            t_pre = time.time()
            if os.path.exists(ppath):
                z = np.load(ppath)
                cont = dict(fz=z['fz'].astype(np.float64), pz=z['pz'].astype(np.float64), sfreq=float(z['sfreq']), events=z['events'], n=int(z['n']), ptp=z['ptp'], n_scalp=int(z['n_scalp']))
                pre_what = 'preproc cached'
            else:
                raw, err = load_raw(sub, cfg, a.dataset)
                if err:
                    print(f'  [{i}/{len(subs)}] {sub}: SKIPPED {err}')
                    continue
                raw, _ = IO.prepare_channel_types(raw)
                raw = PP.bandpass(raw, cfg)
                raw, _ = PP.apply_reference(raw, a.dataset, cfg.REREF)
                raw, _, _ = PP.run_ica(raw, cfg)
                import mne
                ev, _ = mne.events_from_annotations(raw, verbose=False)
                ch = raw.ch_names
                _, fi = PP.get_channel_index(ch, cfg.EARLY_CHANNEL, cfg.EARLY_FALLBACK)
                _, pi = PP.get_channel_index(ch, cfg.P300_CHANNEL, cfg.P300_FALLBACK)
                if fi is None or pi is None:
                    print(f'  [{i}/{len(subs)}] {sub}: SKIPPED missing_channels')
                    continue
                d = raw.get_data()
                sf_ = float(raw.info['sfreq'])
                lo_ = int(round(cfg.TMIN_EPOCH * sf_))
                nw_ = int(round(cfg.TMAX_EPOCH * sf_)) - lo_ + 1
                import mne as _mne
                scalp = _mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
                ptp = PP.rolling_ptp_max(d[scalp], nw_, lo_)
                cont = dict(fz=d[fi], pz=d[pi], sfreq=sf_, events=ev[:, 0], n=d.shape[1], ptp=ptp, n_scalp=len(scalp))
                np.savez_compressed(ppath, fz=cont['fz'], pz=cont['pz'], sfreq=cont['sfreq'], events=cont['events'], n=cont['n'], ptp=cont['ptp'], n_scalp=cont['n_scalp'])
                pre_what = 'load+filter+ICA'
                del raw, d
            pre_s = time.time() - t_pre
            rng = np.random.default_rng(cfg.PSEUDOTRIAL_SEED)
            t_dr = time.time()
            ps, meta, err = subject_pseudo_slopes(sub, cont, g, models, cfg, rng, a.k, profile=a.profile)
            dr_s = time.time() - t_dr
            if err:
                print(f'  [{i}/{len(subs)}] {sub}: SKIPPED {err}')
                continue
            np.savez_compressed(cpath, **ps)
            meta_all += meta
            print(f'  [{i}/{len(subs)}] {sub}: {len(g)} real trials | preproc {pre_s:.0f}s ({pre_what}) | {a.k} draws {dr_s:.1f}s ({1000 * dr_s / a.k:.0f} ms/draw) | full K={cfg.N_DRAWS} would be {(pre_s + dr_s * cfg.N_DRAWS / a.k) / 60:.1f} min')
        for m in models:
            v = ps[m.id]
            nv = int(np.isfinite(v).sum())
            x, sx = E.robust_z_within_subject(g[m.predictor].values)
            y, sy = E.robust_z_within_subject(g[m.outcome].values)
            real = E.ols_slope(x, y)
            rows.append(dict(dataset=a.dataset, variant=tag, model_id=m.id, subject=sub, real_slope=real, pseudo_mean=float(np.nanmean(v)) if nv else np.nan, n_pseudo_draws=nv, used=bool(nv >= cfg.MIN_PSEUDO_DRAWS and np.isfinite(real)), predictor_scaled=sx, outcome_scaled=sy))
    if not rows:
        print('\nno participant produced slopes')
        return 1
    ps_df = pd.DataFrame(rows)
    ps_df['d'] = ps_df.real_slope - ps_df.pseudo_mean
    out = []
    for m in models:
        g = ps_df[(ps_df.model_id == m.id) & ps_df.used]
        if len(g) < 3:
            out.append(dict(model_id=m.id, n_used=len(g), status='too_few'))
            continue
        dbeta, lo, hi = E.bca_ci(g.d.values)
        br = float(np.nanmean(g.real_slope))
        bp = g.pseudo_mean.dropna().values
        tail = (1 + np.sum(bp >= br)) / (len(bp) + 1) if br >= 0 else (1 + np.sum(bp <= br)) / (len(bp) + 1)
        out.append(dict(dataset=a.dataset, variant=tag, model_id=m.id, paper_label=m.paper or '', measure=m.measure, predictor=m.predictor, channel=m.channel, beta_real=br, beta_pseudo_mean=float(np.nanmean(bp)), beta_pseudo_sd=float(np.nanstd(bp)), dbeta=dbeta, bca_lo=lo, bca_hi=hi, tail_fraction=float(tail), n_used=int(len(g)), n_total=int((ps_df.model_id == m.id).sum()), draws_min=int(g.n_pseudo_draws.min()), draws_median=int(g.n_pseudo_draws.median()), n_unscaled=int((~g.predictor_scaled).sum()), status='ok'))
    res = pd.DataFrame(out)
    p1 = os.path.join(cfg.RESULTS_DIR, f'matched{fam}_{a.dataset}_{tag}.csv')
    p2 = os.path.join(cfg.RESULTS_DIR, f'matched{fam}_persubject_{a.dataset}_{tag}.csv')
    res.to_csv(p1, index=False)
    ps_df.to_csv(p2, index=False)
    print(f"\n{'model':8s} {'measure':26s} {'beta_real':>10s} {'dbeta':>9s} {'95% BCa':>22s} {'tail':>7s} {'N':>4s}")
    for _, r in res.iterrows():
        if r.get('status') != 'ok':
            print(f"{r.model_id:8s} {'-':26s} {'':>10s} {'':>9s} {'':>22s} {'':>7s} {r.n_used:4d}  [{r.status}]")
            continue
        print(f'{r.model_id:8s} {r.measure:26s} {r.beta_real:+10.4f} {r.dbeta:+9.4f} [{r.bca_lo:+9.4f},{r.bca_hi:+9.4f}] {r.tail_fraction:7.4f} {r.n_used:4d}')
    print(f'\n-> {p1}\n-> {p2}')
    if meta_all:
        p3 = os.path.join(cfg.RESULTS_DIR, f'placement_meta{fam}_{a.dataset}_{tag}.csv')
        pd.DataFrame(meta_all).to_csv(p3, index=False)
        print(f'-> {p3}  ({len(meta_all)} placements, draw 0 of each participant)')
    nu = res.get('n_unscaled')
    if nu is not None and nu.fillna(0).sum() > 0:
        print(f'\nparticipants whose predictor had MAD = 0 and was left unscaled: {int(nu.fillna(0).sum())} model-participant cases')
    print(f'\ntotal {time.time() - t0:.0f}s')
    return 0
if __name__ == '__main__':
    sys.exit(main())
