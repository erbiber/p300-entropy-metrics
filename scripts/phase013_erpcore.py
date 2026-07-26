
import os, zlib, warnings
import numpy as np
import mne
import phase013_cache as C
from config import (
    DATA_ROOT,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    BASELINE,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT, ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, P300_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
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

def _preprocess(set_path):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
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

def iter_subjects(config, K=1000, subjects=None, subset_n=None, cache_dir=None, targets_only=False,
                  clean_pseudo=False, resample_hz=None):
    """config = (name, min_gap_seconds, reject_threshold).
    targets_only=True keeps only TARGET_CODES (condition-matched to ds006018).
    clean_pseudo=True drops pseudotrials whose early/P300 window overlaps a real evoked period.
    resample_hz sets a target rate: epochs are formed at the native rate (trigger timing intact)
    then resampled, to test sampling-rate sensitivity of the window measures (e.g. RMS/complexity)."""
    cname, min_gap, reject = config
    if targets_only:
        cname = cname + "_targets"   # separate cache namespace so it can't collide with the full run
    if clean_pseudo:
        cname = cname + "_clean"
    if resample_hz:
        cname = cname + f"_rs{int(resample_hz)}"
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")
    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]
    if subjects: sub_ids = [s for s in sub_ids if s in set(subjects)]
    if subset_n: sub_ids = sub_ids[:subset_n]
    for sid in sub_ids:
        cp = None
        if cache_dir:
            cp = C.cache_path(cache_dir, "erp_core", cname, K, f"sub-{sid}")
            cached = C.load(cp)
            if cached is not None:
                print(f"  sub-{sid}: cached (pseudo_draws={cached['n_draws']})")
                yield f"sub-{sid}", cached['real_slope'], cached['pseudo_slopes'], cached['real_feats']
                continue
        set_path = os.path.join(DATA_ROOT, f"sub-{sid}", "ses-P3", "eeg",
                                f"sub-{sid}_ses-P3_task-P3_eeg.set")
        if not os.path.exists(set_path):
            print(f"  sub-{sid}: file_not_found"); continue
        try:
            raw = _preprocess(set_path)
        except Exception as e:
            print(f"  sub-{sid}: preprocess_failed {e}"); continue
        sfreq = raw.info['sfreq']; n_samples = raw.n_times
        events, _ = mne.events_from_annotations(raw, verbose=False)
        real_sample_indices = events[:, 0]
        epochs_real = mne.Epochs(raw, events, event_id=None, tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
                                 baseline=BASELINE, reject=dict(eeg=reject), preload=True, verbose=False)
        if len(epochs_real) < MIN_TRIALS_REQUIRED:
            print(f"  sub-{sid}: insufficient_real ({len(epochs_real)})"); continue
        codes = TARGET_CODES if targets_only else (STANDARD_CODES + TARGET_CODES)
        keep = np.isin(epochs_real.events[:, 2], codes)
        epochs_real = epochs_real[keep]
        if len(epochs_real) < MIN_TRIALS_REQUIRED:
            print(f"  sub-{sid}: insufficient_valid_real ({len(epochs_real)})"); continue
        if resample_hz and abs(sfreq - resample_hz) > 1:
            epochs_real.resample(resample_hz, verbose=False)
        n_real = len(epochs_real)
        ch = epochs_real.ch_names
        fz_name, fz_idx = get_channel_index(ch, EARLY_CHANNEL, EARLY_FALLBACK)
        pz_name, pz_idx = get_channel_index(ch, P300_CHANNEL, P300_FALLBACK)
        if fz_name is None or pz_name is None:
            print(f"  sub-{sid}: missing_channels"); continue
        times = epochs_real.times
        real_feats = features_from_epochs(epochs_real.get_data(), times, fz_idx, pz_idx,
                                          EARLY_WINDOW, P300_WINDOW)
        stim_onsets = epochs_real.events[:, 0]
        rng = _subject_rng(sid)
        pseudo_feats_list = []
        for k in range(K):
            ps = generate_pseudotrial_samples(n_real, sfreq, n_samples, real_sample_indices,
                                              min_gap, TMIN_EPOCH, TMAX_EPOCH, rng,
                                              attempt_factor=10, attempt_floor=1000)
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
        if cp is not None:
            C.save(cp, dict(real_slope=real_slope, pseudo_slopes=pseudo_slopes,
                            real_feats=real_feats, n_draws=len(pseudo_feats_list)))
        tag = "ok" if len(pseudo_feats_list) >= 200 else f"LOW_PSEUDO({len(pseudo_feats_list)})"
        mtr = np.mean([len(pf["p300_pz"]) for pf in pseudo_feats_list]) if pseudo_feats_list else 0
        print(f"  sub-{sid}: {tag}  real_n={n_real}  pseudo_draws={len(pseudo_feats_list)}  trials/draw={mtr:.0f}")
        yield f"sub-{sid}", real_slope, pseudo_slopes, real_feats
