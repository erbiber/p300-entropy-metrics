import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np
import pandas as pd
import config as cfg
from lib import io_data as IO, preprocess as PP, parity as PAR


def load_raw(sid, dataset):
    if dataset == 'erpcore':
        return IO.load_erpcore(sid, cfg)
    return IO.load_ds006018(sid, cfg)


def continuous(sub, dataset, cache_dir, pre_tag, fresh=False):
    sid = sub.replace('sub-', '')
    raw, err = load_raw(sid, dataset)
    if err:
        return (None, None, err, '')
    import mne
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_of = {}
    for desc, val in event_id.items():
        m = re.search('(\\d+)\\s*$', str(desc))
        code_of[int(val)] = int(m.group(1)) if m else None
    first = int(raw.first_samp)
    ppath = os.path.join(cache_dir, f'cont_{dataset}_{pre_tag}_{sub}.npz')
    if os.path.exists(ppath) and not fresh:
        z = np.load(ppath)
        cont = dict(fz=z['fz'].astype(np.float64), pz=z['pz'].astype(np.float64), sfreq=float(z['sfreq']), events=z['events'], n=int(z['n']), ptp=z['ptp'])
        how = 'step-3 cache'
    else:
        raw, _ = IO.prepare_channel_types(raw)
        raw = PP.bandpass(raw, cfg)
        raw, _ = PP.apply_reference(raw, dataset, cfg.REREF)
        raw, _, _ = PP.run_ica(raw, cfg)
        ev, _ = mne.events_from_annotations(raw, verbose=False)
        ch = raw.ch_names
        _, fi = PP.get_channel_index(ch, cfg.EARLY_CHANNEL, cfg.EARLY_FALLBACK)
        _, pi = PP.get_channel_index(ch, cfg.P300_CHANNEL, cfg.P300_FALLBACK)
        if fi is None or pi is None:
            return (None, None, 'missing_channels', '')
        d = raw.get_data()
        sf_ = float(raw.info['sfreq'])
        lo_ = int(round(cfg.TMIN_EPOCH * sf_))
        nw_ = int(round(cfg.TMAX_EPOCH * sf_)) - lo_ + 1
        scalp = mne.pick_types(raw.info, eeg=True, eog=False, misc=False, exclude=[])
        ptp = PP.rolling_ptp_max(d[scalp], nw_, lo_)
        cont = dict(fz=d[fi], pz=d[pi], sfreq=sf_, events=ev[:, 0], n=d.shape[1], ptp=ptp)
        if not os.path.exists(ppath):
            np.savez_compressed(ppath, fz=cont['fz'], pz=cont['pz'], sfreq=cont['sfreq'], events=cont['events'], n=cont['n'], ptp=cont['ptp'], n_scalp=len(scalp))
        how = 'load+filter+ICA'
        del d
    if first != 0:
        return (None, None, f'first_samp_{first}_nonzero', how)
    if len(cont['events']) != len(events) or not np.array_equal(np.asarray(cont['events']), events[:, 0]):
        return (None, None, 'event_samples_differ_from_cache', how)
    return (cont, (events, code_of), None, how)


def subject_rows(sub, dataset, cont, evinfo, real):
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
    n = hi - lo + 1
    inb = (samples + lo >= 0) & (samples + hi < cont['n'])
    samples = samples[inb]
    codes = codes[inb]
    ok = cont['ptp'][samples] <= cfg.REJECT_THRESHOLD
    samples = samples[ok]
    codes = codes[ok]
    idx = samples[:, None] + np.arange(lo, lo + n)[None, :]
    fz = cont['fz'][idx]
    pz = cont['pz'][idx]
    times = np.arange(lo, hi + 1) / sf
    if cfg.APPLY_BASELINE:
        bm = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[1])
        fz = fz - fz[:, bm].mean(axis=1, keepdims=True)
        pz = pz - pz[:, bm].mean(axis=1, keepdims=True)
    from lib.features import window_mask
    p300 = pz[:, window_mask(times, cfg.P300_WINDOW)].mean(axis=1)
    is_t = np.array([int(c in targets) for c in codes])
    r = real.sort_values('trial_idx')
    if len(r) != len(samples):
        return (None, f'misaligned_trial_count_features={len(r)}_rebuilt={len(samples)}')
    if not np.array_equal(r.trial_idx.values, np.arange(len(r))):
        return (None, 'misaligned_trial_idx_not_contiguous')
    if not np.array_equal(r.is_target.values.astype(int), is_t):
        return (None, 'misaligned_condition_labels')
    dev = np.abs(r.p300_pz.values - p300)
    tol = 1e-9 * np.maximum(1e-6, np.abs(r.p300_pz.values))
    if np.any(dev > tol):
        return (None, f'misaligned_p300_max_abs_diff_{dev.max():.3e}')
    f = PAR.window_set(fz, pz, times, cfg, sf, groups=is_t)
    out = pd.DataFrame(f)
    out.insert(0, 'subject', sub)
    out.insert(1, 'trial_idx', np.arange(len(samples)))
    out.insert(2, 'is_target', is_t)
    out.insert(3, 'sample', samples)
    out.insert(4, 'onset_s', samples / sf)
    out.insert(5, 'p300_pz_rebuilt', p300)
    out.insert(6, 'p300_max_abs_diff', float(dev.max()))
    out['status'] = 'ok'
    return (out, None)


def main():
    ap = argparse.ArgumentParser(description='Step 5: time-reversal parity features on the real trials of step 1.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018'])
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--fresh', action='store_true', help='refit preprocessing and ICA instead of reading the step-3 cache')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    tag = cfg.variant_tag()
    cfg.banner(f'step 5  parity features  |  dataset={a.dataset}  variant={tag}')
    if cfg.RESAMPLE_HZ:
        print('step 5 is defined for native-rate arms only; rerun without --resample')
        return 2
    fails = PAR.selftest()
    if fails:
        for x in fails:
            print('SELFTEST FAIL ' + x)
        return 3
    print('parity self-test: all odd features odd, all even partners even, residual removes constant waveforms')
    fp = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    if not os.path.exists(fp):
        print(f'features file not found: {fp}\nrun step 1 with the same options first')
        return 2
    feats = pd.read_csv(fp, usecols=['subject', 'trial_idx', 'is_target', 'p300_pz'])
    subs = sorted(feats.subject.unique())
    if a.subjects:
        want = {x.strip().replace('sub-', '').zfill(3) for x in a.subjects.replace(';', ',').split(',') if x.strip()}
        subs = [s for s in subs if s.replace('sub-', '') in want]
    cache_dir = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    pre_tag = f"v4_hp{cfg.FILTER_LOW:g}-{cfg.FILTER_HIGH:g}_{cfg.ica_tag()}_{cfg.REREF}"
    frames = []
    log = []
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        t1 = time.time()
        try:
            cont, evinfo, err, how = continuous(sub, a.dataset, cache_dir, pre_tag, a.fresh)
            if err:
                out = None
            else:
                out, err = subject_rows(sub, a.dataset, cont, evinfo, feats[feats.subject == sub])
        except Exception as exc:
            out, err, how = None, f'error_{type(exc).__name__}: {exc}', ''
        if out is None:
            print(f'  [{i}/{len(subs)}] {sub}: EXCLUDED {err}')
            log.append(dict(subject=sub, status=err, n_trials=0, source=how, p300_max_abs_diff=np.nan))
            continue
        frames.append(out)
        log.append(dict(subject=sub, status='ok', n_trials=len(out), source=how, p300_max_abs_diff=float(out.p300_max_abs_diff.iloc[0])))
        print(f'  [{i}/{len(subs)}] {sub}: {len(out)} trials aligned with step 1 (max |dP300| {out.p300_max_abs_diff.iloc[0]:.2e} V) | {how} | {time.time() - t1:.0f}s')
    if not frames:
        print('no subject produced parity features')
        return 1
    res = pd.concat(frames, ignore_index=True)
    p1 = os.path.join(cfg.RESULTS_DIR, f'features_parity_{a.dataset}_{tag}.csv')
    p2 = os.path.join(cfg.RESULTS_DIR, f'features_parity_{a.dataset}_{tag}_log.csv')
    res.to_csv(p1, index=False)
    pd.DataFrame(log).to_csv(p2, index=False)
    bad = sum(1 for r in log if r['status'] != 'ok')
    print(f'\n{res.subject.nunique()} subjects, {len(res)} trials, {bad} excluded, {time.time() - t0:.0f}s')
    nf = {c: int((~np.isfinite(res[c])).sum()) for c in PAR.COLUMNS if (~np.isfinite(res[c])).any()}
    if nf:
        print(f'non-finite parity values (treated as missing downstream): {nf}')
    print(f'-> {p1}\n-> {p2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
