import argparse
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats, signal
import config as cfg
from lib import io_data as IO, preprocess as PP, fastfeat
from lib.features import window_mask

META = ['dataset', 'variant', 'subject', 'trial_idx', 'event_code', 'condition', 'is_target']
ALPHA = (8.0, 13.0)
SLOW_HZ = 3.0
FEATURE_RTOL = 1e-6
ENTROPY = ('pe_early_fz', 'se_early_fz', 'lz_early_fz')


def load_raw(sid, dataset):
    if dataset == 'erpcore':
        return IO.load_erpcore(sid, cfg)
    return IO.load_ds006018(sid, cfg)


def build_cache(raw, dataset, ppath):
    import mne
    raw, _ = IO.prepare_channel_types(raw)
    raw = PP.bandpass(raw, cfg)
    raw, _ = PP.apply_reference(raw, dataset, cfg.REREF)
    raw, _, _ = PP.run_ica(raw, cfg)
    ev, _ = mne.events_from_annotations(raw, verbose=False)
    ch = raw.ch_names
    _, fi = PP.get_channel_index(ch, cfg.EARLY_CHANNEL, cfg.EARLY_FALLBACK)
    _, pi = PP.get_channel_index(ch, cfg.P300_CHANNEL, cfg.P300_FALLBACK)
    if fi is None or pi is None:
        return None
    d = raw.get_data()
    sf = float(raw.info['sfreq'])
    lo = int(round(cfg.TMIN_EPOCH * sf))
    nw = int(round(cfg.TMAX_EPOCH * sf)) - lo + 1
    scalp = mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
    ptp = PP.rolling_ptp_max(d[scalp], nw, lo)
    cont = dict(fz=d[fi], pz=d[pi], sfreq=sf, events=ev[:, 0], n=d.shape[1], ptp=ptp, n_scalp=len(scalp))
    np.savez_compressed(ppath, **cont)
    return cont


def continuous(sub, dataset, cache_dir, pre_tag):
    import mne
    raw, err = load_raw(sub.replace('sub-', ''), dataset)
    if err:
        return (None, None, err, '')
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_of = {}
    for desc, val in event_id.items():
        m = re.search('(\\d+)\\s*$', str(desc))
        code_of[int(val)] = int(m.group(1)) if m else None
    if int(raw.first_samp) != 0:
        return (None, None, f'first_samp_{raw.first_samp}_nonzero', '')
    ppath = os.path.join(cache_dir, f'cont_{dataset}_{pre_tag}_{sub}.npz')
    if os.path.exists(ppath):
        z = np.load(ppath)
        cont = dict(fz=z['fz'].astype(np.float64), pz=z['pz'].astype(np.float64), sfreq=float(z['sfreq']), events=z['events'], n=int(z['n']), ptp=z['ptp'])
        how = 'step-3 cache'
    else:
        cont = build_cache(raw, dataset, ppath)
        if cont is None:
            return (None, None, 'missing_channels', '')
        how = 'load+filter+ICA'
    if len(cont['events']) != len(events) or not np.array_equal(np.asarray(cont['events']), events[:, 0]):
        return (None, None, 'event_samples_differ_from_cache', how)
    return (cont, (events, code_of), None, how)


def real_trials(cont, evinfo, dataset):
    events, code_of = evinfo
    targets, standards = cfg.codes_for(dataset)
    keep = set(targets + standards)
    codes = np.array([code_of.get(int(v)) for v in events[:, 2]], dtype=object)
    sel = np.array([c is not None and c in keep for c in codes])
    samples = events[sel, 0].astype(np.int64)
    codes = codes[sel]
    sf = cont['sfreq']
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    inb = (samples + lo >= 0) & (samples + hi < cont['n'])
    samples, codes = samples[inb], codes[inb]
    ok = cont['ptp'][samples] <= cfg.REJECT_THRESHOLD
    samples, codes = samples[ok], codes[ok]
    idx = samples[:, None] + np.arange(lo, hi + 1)[None, :]
    times = np.arange(lo, hi + 1) / sf
    is_t = np.array([int(c in targets) for c in codes])
    return (cont['fz'][idx], cont['pz'][idx], times, samples, codes, is_t)


def baseline(x, times):
    bm = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[1])
    return x - x[:, bm].mean(axis=1, keepdims=True)


def compare(real, rebuilt, cols):
    dev = {}
    for c in cols:
        a = real[c].values.astype(float)
        b = np.asarray(rebuilt[c], float)
        both = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            dev[c] = np.inf
            continue
        if not both.any():
            dev[c] = 0.0
            continue
        scale = np.maximum(np.abs(a[both]), np.nanmedian(np.abs(a[both])) + 1e-300)
        dev[c] = float(np.max(np.abs(a[both] - b[both]) / scale))
    return dev


def symmetry(x, sf, clean):
    from mne.filter import filter_data
    out = {}
    xc = x[clean]
    out['skew'] = float(stats.skew(xc, bias=False))
    out['kurt'] = float(stats.kurtosis(xc, fisher=True, bias=False))
    a = filter_data(x[None, :], sf, ALPHA[0], ALPHA[1], verbose=False)[0]
    env = np.abs(signal.hilbert(a))
    slow = filter_data(x[None, :], sf, None, SLOW_HZ, verbose=False)[0]
    out['alpha_env_mean'] = float(env[clean].mean())
    out['bsi'] = float(np.corrcoef(env[clean], slow[clean])[0, 1])
    return out


def subject(sub, dataset, cache_dir, pre_tag, real, tag_nb):
    cont, evinfo, err, how = continuous(sub, dataset, cache_dir, pre_tag)
    if err:
        return (None, None, None, dict(subject=sub, status=err, source=how))
    fz, pz, times, samples, codes, is_t = real_trials(cont, evinfo, dataset)
    r = real.sort_values('trial_idx').reset_index(drop=True)
    if len(r) != len(samples):
        return (None, None, None, dict(subject=sub, status=f'misaligned_trial_count_features={len(r)}_rebuilt={len(samples)}', source=how))
    if not np.array_equal(r.trial_idx.values, np.arange(len(r))):
        return (None, None, None, dict(subject=sub, status='misaligned_trial_idx_not_contiguous', source=how))
    if not np.array_equal(r.is_target.values.astype(int), is_t):
        return (None, None, None, dict(subject=sub, status='misaligned_condition_labels', source=how))
    if 'event_code' in r.columns and not np.array_equal(r.event_code.values.astype(int), codes.astype(int)):
        return (None, None, None, dict(subject=sub, status='misaligned_event_codes', source=how))
    sf = cont['sfreq']
    fzb, pzb = baseline(fz, times), baseline(pz, times)
    fb = fastfeat.window_features(fzb, pzb, times, cfg, want_entropy=True)
    cols = [c for c in fb if c in r.columns]
    dev = compare(r, fb, [c for c in cols if c not in ENTROPY])
    worst = max(dev.values()) if dev else np.inf
    if not np.isfinite(worst) or worst > FEATURE_RTOL:
        bad = {k: v for k, v in dev.items() if not np.isfinite(v) or v > FEATURE_RTOL}
        return (None, None, None, dict(subject=sub, status='rebuilt_features_differ_' + ';'.join(f'{k}:{v:.1e}' for k, v in sorted(bad.items())), source=how, max_rel_dev=worst))
    ent = {}
    for c in [c for c in cols if c in ENTROPY]:
        a_ = r[c].values.astype(float)
        b_ = np.asarray(fb[c], float)
        d_ = np.abs(a_ - b_) > FEATURE_RTOL * np.maximum(np.abs(a_), 1e-300)
        d_ |= np.isfinite(a_) != np.isfinite(b_)
        ent[f'entropy_trials_differing_{c}'] = int(np.nansum(d_))
    fn = fastfeat.window_features(fz, pz, times, cfg, want_entropy=True)
    nb = r[[c for c in META if c in r.columns]].copy()
    nb['variant'] = tag_nb
    for c in cols:
        nb[c] = np.asarray(fn[c], float)
    erp = []
    for cond, flag in (('target', 1), ('standard', 0)):
        for ch, x in (('Fz', fzb), ('Pz', pzb)):
            w = x[is_t == flag].mean(axis=0)
            row = dict(subject=sub, condition=cond, channel=ch, n_trials=int((is_t == flag).sum()), sfreq=sf)
            row.update({f'{1000 * t:.3f}': float(v) for t, v in zip(times, w)})
            erp.append(row)
    lo = int(round(cfg.TMIN_EPOCH * sf))
    hi = int(round(cfg.TMAX_EPOCH * sf))
    okon = np.where(cont['ptp'] <= cfg.REJECT_THRESHOLD)[0]
    okon = okon[(okon + lo >= 0) & (okon + hi < cont['n'])]
    span = np.zeros(cont['n'] + 1, dtype=np.int64)
    np.add.at(span, okon + lo, 1)
    np.add.at(span, okon + hi + 1, -1)
    clean = np.cumsum(span)[:-1] > 0
    sym = dict(subject=sub, n_samples=int(cont['n']), clean_fraction=float(clean.mean()), sfreq=sf)
    if clean.sum() > 10 * sf:
        for ch, x in (('fz', cont['fz']), ('pz', cont['pz'])):
            for k, v in symmetry(x, sf, clean).items():
                sym[f'{k}_{ch}'] = v
    log = dict(subject=sub, status='ok', source=how, n_trials=len(r), max_rel_dev=worst, **ent)
    return (nb, erp, sym, log)


def main():
    ap = argparse.ArgumentParser(description='Step 7: from the continuous cache, verify the step-1 features, then write features without baseline correction, per-participant ERP averages and background symmetry measures.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018'])
    ap.add_argument('--subjects', default=None)
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    if cfg.RESAMPLE_HZ or not cfg.APPLY_BASELINE:
        print('step 7 runs on native-rate, baseline-corrected arms only; it writes the no-baseline variant itself')
        return 2
    tag = cfg.variant_tag()
    cfg.APPLY_BASELINE = False
    tag_nb = cfg.variant_tag()
    cfg.APPLY_BASELINE = True
    cfg.banner(f'step 7  from cache  |  dataset={a.dataset}  variant={tag}  ->  {tag_nb}')
    fp = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    if not os.path.exists(fp):
        print(f'features file not found: {fp}\nrun step 1 with the same options first')
        return 2
    feats = pd.read_csv(fp)
    subs = sorted(feats.subject.unique())
    if a.subjects:
        want = {x.strip().replace('sub-', '').zfill(3) for x in a.subjects.replace(';', ',').split(',') if x.strip()}
        subs = [s for s in subs if s.replace('sub-', '') in want]
    cache_dir = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    pre_tag = f"v4_hp{cfg.FILTER_LOW:g}-{cfg.FILTER_HIGH:g}_{cfg.ica_tag()}_{cfg.REREF}"
    frames, erps, syms, logs = [], [], [], []
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        t1 = time.time()
        try:
            nb, erp, sym, log = subject(sub, a.dataset, cache_dir, pre_tag, feats[feats.subject == sub], tag_nb)
        except Exception as exc:
            nb, erp, sym, log = None, None, None, dict(subject=sub, status=f'error_{type(exc).__name__}: {exc}', source='')
        logs.append(log)
        if nb is None:
            print(f'  [{i}/{len(subs)}] {sub}: EXCLUDED {log["status"]}')
            continue
        frames.append(nb)
        erps += erp
        syms.append(sym)
        print(f'  [{i}/{len(subs)}] {sub}: {len(nb)} trials, step-1 features reproduced (max rel. dev. {log["max_rel_dev"]:.1e}) | {log["source"]} | {time.time() - t1:.0f}s')
    if not frames:
        print('no subject processed')
        return 1
    out = {
        'features': os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag_nb}.csv'),
        'log': os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag_nb}_log.csv'),
        'erp': os.path.join(cfg.RESULTS_DIR, f'erp_{a.dataset}_{tag}.csv'),
        'sym': os.path.join(cfg.RESULTS_DIR, f'symmetry_{a.dataset}_{tag}.csv'),
    }
    pd.concat(frames, ignore_index=True).to_csv(out['features'], index=False)
    lg = pd.DataFrame(logs)
    nt = {f.subject.iloc[0]: int(f.is_target.sum()) for f in frames}
    ns = {f.subject.iloc[0]: int(len(f) - f.is_target.sum()) for f in frames}
    lg['n_target'] = lg.subject.map(nt)
    lg['n_standard'] = lg.subject.map(ns)
    lg.to_csv(out['log'], index=False)
    pd.DataFrame(erps).to_csv(out['erp'], index=False, float_format='%.6e')
    pd.DataFrame(syms).to_csv(out['sym'], index=False)
    bad = int((lg.status != 'ok').sum())
    print(f'\n{len(frames)} subjects, {bad} excluded, {time.time() - t0:.0f}s')
    for v in out.values():
        print(f'-> {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
