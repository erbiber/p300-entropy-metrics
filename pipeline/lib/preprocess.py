import os
import re
import numpy as np
import mne
mne.set_log_level('WARNING')
MASTOIDS = {'erpcore': ['P9', 'P10'], 'ds006018': ['LM', 'RM']}

def get_channel_index(ch_names, primary, fallbacks):
    if primary in ch_names:
        return (primary, ch_names.index(primary))
    for fb in fallbacks:
        if fb in ch_names:
            return (fb, ch_names.index(fb))
    return (None, None)

def bandpass(raw, cfg):
    raw.filter(cfg.FILTER_LOW, cfg.FILTER_HIGH, fir_design=cfg.FILTER_DESIGN, verbose=False)
    return raw

ONLINE_REFERENCE = {'ds006018': 'Cz'}

def apply_reference(raw, dataset, mode):
    if mode == 'distributed':
        return (raw, 'as distributed')
    if mode == 'mastoid':
        pair = [c for c in MASTOIDS[dataset] if c in raw.ch_names]
        if len(pair) < 2:
            raise RuntimeError(f'mastoid reference requested but only {pair} present in {dataset}')
        raw.set_eeg_reference(ref_channels=pair, verbose=False)
        raw.drop_channels(pair)
        return (raw, f'mastoid pair {pair}, dropped after referencing')
    if mode == 'average':
        added = ''
        online = ONLINE_REFERENCE.get(dataset)
        if online and online not in raw.ch_names:
            raw.add_reference_channels(online)
            added = f' with online reference {online} restored as zero channel'
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
        return (raw, 'average of own scalp channels' + added)
    if mode == 'cz':
        if ONLINE_REFERENCE.get(dataset) == 'Cz' and 'Cz' not in raw.ch_names:
            return (raw, 'Cz, the online reference of the recording')
        if 'Cz' not in raw.ch_names:
            raise RuntimeError(f'Cz reference requested but Cz is not present in {dataset}')
        raw.set_eeg_reference(ref_channels=['Cz'], verbose=False)
        raw.drop_channels(['Cz'])
        return (raw, 'Cz, dropped after referencing')
    raise ValueError(f'unknown reference mode: {mode}')

def ica_cache_path(cfg, dataset, sub):
    d = os.path.join(cfg.RESULTS_DIR, 'cache')
    os.makedirs(d, exist_ok=True)
    name = f'ica_{dataset}_{sub}_{cfg.REREF}_f{cfg.FILTER_LOW:g}-{cfg.FILTER_HIGH:g}_h{cfg.ICA_HIGHPASS_FOR_FIT:g}_n{cfg.ICA_N_COMPONENTS:g}_{cfg.ICA_METHOD}_s{cfg.ICA_RANDOM_STATE}-ica.fif'
    return os.path.join(d, name)

def eog_channels(raw):
    idx = mne.pick_types(raw.info, eeg=False, eog=True)
    return [raw.ch_names[i] for i in idx]

def run_ica(raw, cfg, cache_path=None):
    if not cfg.DO_ICA:
        return (raw, 0, 0)
    if cache_path and os.path.exists(cache_path):
        try:
            ica = mne.preprocessing.read_ica(cache_path, verbose=False)
            n_exc = len(ica.exclude)
            raw = ica.apply(raw, verbose=False)
            return (raw, ica.n_components_, n_exc)
        except Exception as e:
            print(f'    cached ICA unreadable, refitting: {e}')
    try:
        raw_fit = raw.copy().filter(cfg.ICA_HIGHPASS_FOR_FIT, None, fir_design=cfg.FILTER_DESIGN, verbose=False)
        ica = mne.preprocessing.ICA(n_components=cfg.ICA_N_COMPONENTS, method=cfg.ICA_METHOD, random_state=cfg.ICA_RANDOM_STATE, max_iter='auto')
        ica.fit(raw_fit, verbose=False)
        n_comp = ica.n_components_
        n_eog = len(mne.pick_types(raw.info, eeg=False, eog=True))
        if n_eog == 0:
            print('    WARNING: ICA requested but no channel is typed eog; no artifact components can be identified. Add the recording\'s EOG channel names to io_data.EOG_CANDIDATES.')
            ica.exclude = []
            n_exc = 0
        else:
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
                n_exc = len(eog_idx)
            except Exception as exc:
                print(f'    WARNING: find_bads_eog failed ({type(exc).__name__}: {exc}); ICA will remove nothing.')
                ica.exclude = []
                n_exc = 0
        if cache_path:
            try:
                ica.save(cache_path, overwrite=True, verbose=False)
                ica = mne.preprocessing.read_ica(cache_path, verbose=False)
            except Exception as e:
                print(f'    could not cache ICA: {e}')
        raw = ica.apply(raw, verbose=False)
        return (raw, n_comp, n_exc)
    except Exception as e:
        print(f'    ICA failed, continuing without: {e}')
        return (raw, 0, 0)

def make_epochs(raw, cfg, keep_codes=None):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_of = {}
    for desc, val in event_id.items():
        m = re.search('(\\d+)\\s*$', str(desc))
        code_of[int(val)] = int(m.group(1)) if m else None
    n_all = len(events)
    if keep_codes:
        allowed = set(keep_codes)
        mask = np.array([code_of.get(int(v)) in allowed for v in events[:, 2]], dtype=bool)
        events = events[mask]
        if len(events) == 0:
            raise ValueError('no event matches the configured paradigm codes')
    n_kept = len(events)
    uniq, counts = np.unique(events[:, 0], return_counts=True)
    n_dup = int((counts > 1).sum())
    baseline = cfg.BASELINE if cfg.APPLY_BASELINE else None
    epochs = mne.Epochs(raw, events, event_id=None, tmin=cfg.TMIN_EPOCH, tmax=cfg.TMAX_EPOCH, baseline=baseline, reject=dict(eeg=cfg.REJECT_THRESHOLD), preload=True, verbose=False)
    return (epochs, code_of, dict(n_events_all=n_all, n_events_kept=n_kept, n_duplicate_samples=n_dup))

def restrict_to_codes(epochs, cfg, dataset, code_of):
    tgt, std = cfg.codes_for(dataset)
    if not tgt and not std:
        raise ValueError(f'no event codes configured for dataset {dataset!r}; set P300_DS006018_TARGET_CODES (and _STANDARD_CODES if the dataset has both conditions) before running')
    keep = set(tgt + std)
    paradigm = np.array([code_of.get(int(v)) for v in epochs.events[:, 2]], dtype=object)
    valid = np.array([c is not None and c in keep for c in paradigm])
    if not valid.any():
        seen = sorted({c for c in paradigm if c is not None})
        return (epochs[np.zeros(len(epochs), bool)], seen)
    return (epochs[valid], sorted({c for c in paradigm[valid]}))

def rolling_ptp_max(data, n_window, lo):
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    n_s = data.shape[1]
    org = -(n_window // 2)
    out = np.zeros(n_s, dtype=np.float64)
    for c in range(data.shape[0]):
        x = data[c]
        mx = maximum_filter1d(x, size=n_window, origin=org, mode='nearest')
        mn = minimum_filter1d(x, size=n_window, origin=org, mode='nearest')
        np.maximum(out, mx - mn, out=out)
    shifted = np.full(n_s, np.inf)
    if lo >= 0:
        shifted[:n_s - lo] = out[lo:]
    else:
        shifted[-lo:] = out[:n_s + lo]
    return shifted

def resample_if_asked(epochs, cfg):
    if cfg.RESAMPLE_HZ and abs(epochs.info['sfreq'] - cfg.RESAMPLE_HZ) > 1e-09:
        epochs.resample(cfg.RESAMPLE_HZ, verbose=False)
        return True
    return False
