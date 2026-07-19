import zlib, warnings
import numpy as np
import mne
import phase013_cache as C
from config_ds006018 import (
    DATASET_ID, CACHE_DIR, TASK_FILTER,
    TMIN_EPOCH, TMAX_EPOCH, BASELINE,
    EARLY_WINDOW, P300_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    EOG_CHANNELS, MASTOID_CHANNELS,
    TRIAL_CODES,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT, ICA_METHOD, ICA_RANDOM_STATE,
    MIN_TRIALS_REQUIRED, PSEUDOTRIAL_SEED,
    get_channel_index,
)
from phase013_engine import (
    generate_pseudotrial_samples, features_from_epochs, subject_slopes_real_and_pseudo, clean_pseudo_mask,
)
warnings.filterwarnings('ignore', category=Warning)
mne.set_log_level('WARNING')

def _subject_rng(sid):
    return np.random.default_rng(PSEUDOTRIAL_SEED + zlib.crc32(str(sid).encode()) % (2**31))

def _load_recordings():
    from eegdash.dataset import DS006018
    print(f"Loading {DATASET_ID} via eegdash (cache: {CACHE_DIR}) ...")
    dataset = DS006018(cache_dir=CACHE_DIR)
    recs = []
    for rec in dataset.datasets:
        try:
            desc = rec.description
            task = desc.get('task', None) if hasattr(desc, 'get') else desc['task']
            subj = desc.get('subject', None) if hasattr(desc, 'get') else desc['subject']
        except Exception:
            task, subj = None, None
        if task == TASK_FILTER:
            recs.append((str(subj), rec))
    print(f"  visualoddball recordings: {len(recs)}")
    return recs

def _preprocess(rec):
    raw = rec.raw
    if raw is None: return None
    try: raw.load_data()
    except Exception: pass
    type_map = {}
    for c in EOG_CHANNELS:
        if c in raw.ch_names: type_map[c] = 'eog'
    for c in MASTOID_CHANNELS:
        if c in raw.ch_names: type_map[c] = 'misc'
    if type_map:
        try: raw.set_channel_types(type_map)
        except Exception: pass
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design=FILTER_DESIGN, verbose=False)
    if DO_ICA:
        try:
            raw_for_ica = raw.copy().filter(ICA_HIGHPASS_FOR_FIT, None,
                                            fir_design=FILTER_DESIGN, verbose=False)
            ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
                                        random_state=ICA_RANDOM_STATE, max_iter='auto')
            ica.fit(raw_for_ica, verbose=False)
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False); ica.exclude = eog_idx
            except Exception:
                ica.exclude = []
            raw = ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"    ICA skipped (matches repo behaviour): {e}")
    return raw

def iter_subjects(config, K=1000, subjects=None, subset_n=None, cache_dir=None, clean_pseudo=False, resample_hz=None):
    cname, min_gap, reject = config
    if clean_pseudo:
        cname = cname + "_clean"
    if resample_hz:
        cname = cname + f"_rs{int(resample_hz)}"
    recs = _load_recordings()
    if subset_n: recs = recs[:subset_n]
    seen = set()
    for subj, rec in recs:
        if subjects and subj not in set(subjects): continue
        if subj in seen:
            print(f"  {subj}: duplicate recording, skipping (first kept)"); continue
        cp = None
        if cache_dir:
            cp = C.cache_path(cache_dir, "ds006018", cname, K, subj)
            cached = C.load(cp)
            if cached is not None:
                print(f"  {subj}: cached (pseudo_draws={cached['n_draws']})")
                seen.add(subj)
                yield subj, cached['real_slope'], cached['pseudo_slopes'], cached['real_feats']
                continue
        try:
            raw = _preprocess(rec)
        except Exception as e:
            print(f"  {subj}: preprocess_failed {e}"); continue
        if raw is None:
            print(f"  {subj}: no_raw"); continue
        sfreq = raw.info['sfreq']; n_samples = raw.n_times
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        real_sample_indices = events[:, 0]
        trial_event_id = {k: v for k, v in event_id.items() if v in TRIAL_CODES and v in events[:, 2]}
        if not trial_event_id:
            print(f"  {subj}: no_trial_events"); continue
        epochs_real = mne.Epochs(raw, events, event_id=trial_event_id, tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                                 baseline=BASELINE, reject=dict(eeg=reject), preload=True, verbose=False)
        if len(epochs_real) < MIN_TRIALS_REQUIRED:
            print(f"  {subj}: insufficient_real ({len(epochs_real)})"); continue
        if resample_hz and abs(sfreq - resample_hz) > 1:
            epochs_real.resample(resample_hz, verbose=False)
        n_real = len(epochs_real)
        ch = epochs_real.ch_names
        fz_name, fz_idx = get_channel_index(ch, EARLY_CHANNEL, EARLY_FALLBACK)
        pz_name, pz_idx = get_channel_index(ch, P300_CHANNEL, P300_FALLBACK)
        if fz_name is None or pz_name is None:
            print(f"  {subj}: missing_channels"); continue
        times = epochs_real.times
        real_feats = features_from_epochs(epochs_real.get_data(), times, fz_idx, pz_idx,
                                          EARLY_WINDOW, P300_WINDOW)
        stim_onsets = epochs_real.events[:, 0]
        rng = _subject_rng(subj)
        pseudo_feats_list = []
        for k in range(K):
            ps = generate_pseudotrial_samples(n_real, sfreq, n_samples, real_sample_indices,
                                              min_gap, TMIN_EPOCH, TMAX_EPOCH, rng,
                                              attempt_factor=20, attempt_floor=5000)
            if clean_pseudo and len(ps):
                ps = ps[clean_pseudo_mask(ps, stim_onsets, sfreq, EARLY_WINDOW, P300_WINDOW)]
            if len(ps) < MIN_TRIALS_REQUIRED: continue
            pev = np.column_stack([ps, np.zeros(len(ps), int), np.full(len(ps), 99, int)])
            ep = mne.Epochs(raw, pev, event_id={'pseudo': 99}, tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                            baseline=BASELINE, reject=dict(eeg=reject), preload=True, verbose=False)
            if len(ep) < MIN_TRIALS_REQUIRED: continue
            if resample_hz and abs(sfreq - resample_hz) > 1:
                ep.resample(resample_hz, verbose=False)
            pseudo_feats_list.append(features_from_epochs(ep.get_data(), ep.times, fz_idx, pz_idx,
                                                          EARLY_WINDOW, P300_WINDOW))
        real_slope, pseudo_slopes = subject_slopes_real_and_pseudo(real_feats, pseudo_feats_list)
        seen.add(subj)
        if cp is not None:
            C.save(cp, dict(real_slope=real_slope, pseudo_slopes=pseudo_slopes,
                            real_feats=real_feats, n_draws=len(pseudo_feats_list)))
        tag = "ok" if len(pseudo_feats_list) >= 200 else f"LOW_PSEUDO({len(pseudo_feats_list)})"
        mtr = np.mean([len(pf["p300_pz"]) for pf in pseudo_feats_list]) if pseudo_feats_list else 0
        print(f"  {subj}: {tag}  real_n={n_real}  pseudo_draws={len(pseudo_feats_list)}  trials/draw={mtr:.0f}")
        yield subj, real_slope, pseudo_slopes, real_feats
