import numpy as np

def forbidden_mask(n_continuous_samples, real_event_samples, min_gap_seconds, sfreq):
    min_gap_samples = int(min_gap_seconds * sfreq)
    forbidden = np.zeros(n_continuous_samples, dtype=bool)
    for ev in real_event_samples:
        lo = max(0, int(ev) - min_gap_samples)
        hi = min(n_continuous_samples, int(ev) + min_gap_samples)
        forbidden[lo:hi] = True
    return forbidden

def generate_pseudotrial_samples(n_pseudo, sfreq, n_continuous_samples, real_event_samples, min_gap_seconds, tmin, tmax, rng, forbidden=None, scratch=None):
    min_gap_samples = int(min_gap_seconds * sfreq)
    epoch_samples = int((tmax - tmin) * sfreq)
    pre_buffer = int(abs(tmin) * sfreq) + 1
    post_buffer = int(tmax * sfreq) + 1
    valid_low = pre_buffer
    valid_high = n_continuous_samples - post_buffer
    if valid_high <= valid_low:
        return np.array([], dtype=int)
    if forbidden is None:
        forbidden = forbidden_mask(n_continuous_samples, real_event_samples, min_gap_seconds, sfreq)
    placed_samples = []
    n_attempts = max(n_pseudo * 10, 1000)
    if scratch is None:
        placed_set_for_gap = np.zeros(n_continuous_samples, dtype=bool)
    else:
        placed_set_for_gap = scratch
        placed_set_for_gap[:] = False
    for _ in range(n_attempts):
        if len(placed_samples) >= n_pseudo:
            break
        cand = int(rng.integers(valid_low, valid_high))
        if forbidden[cand]:
            continue
        if placed_set_for_gap[cand]:
            continue
        placed_samples.append(cand)
        lo = max(0, cand - epoch_samples)
        hi = min(n_continuous_samples, cand + epoch_samples)
        placed_set_for_gap[lo:hi] = True
    placed_samples.sort()
    return np.array(placed_samples, dtype=int)

def placement_metadata(placed_samples, real_event_samples, sfreq, tmin, tmax, evoked_window_s=0.8):
    placed = np.asarray(placed_samples, dtype=np.int64)
    events = np.sort(np.asarray(real_event_samples, dtype=np.int64))
    if placed.size == 0:
        return []
    epoch_lo = placed + int(round(tmin * sfreq))
    epoch_hi = placed + int(round(tmax * sfreq))
    evoked_len = int(round(evoked_window_s * sfreq))
    ins = np.searchsorted(events, placed)
    rows = []
    for k in range(placed.size):
        i = ins[k]
        cands = []
        if i > 0:
            cands.append(events[i - 1])
        if i < events.size:
            cands.append(events[i])
        if cands:
            dd = np.abs(np.asarray(cands) - placed[k])
            gap = int(dd.min())
            nearest = int(np.asarray(cands)[int(dd.argmin())])
        else:
            gap, nearest = (-1, -1)
        lo, hi = (int(epoch_lo[k]), int(epoch_hi[k]))
        j0 = np.searchsorted(events, lo - evoked_len, side='left')
        j1 = np.searchsorted(events, hi, side='right')
        n_ovl = ovl = 0
        for ev in events[j0:j1]:
            a, b = (max(lo, int(ev)), min(hi, int(ev) + evoked_len))
            if b > a:
                n_ovl += 1
                ovl += b - a
        rows.append(dict(draw_order=k, placement_sample=int(placed[k]), epoch_lo_sample=lo, epoch_hi_sample=hi, gap_to_nearest_event_samples=gap, gap_to_nearest_event_s=gap / sfreq if gap >= 0 else np.nan, nearest_event_sample=nearest, sequence_position=int(i), n_preceding_events=int(i), n_following_events=int(events.size - i), n_overlapping_evoked=n_ovl, overlap_samples=ovl, overlap_fraction=ovl / (hi - lo) if hi > lo else np.nan))
    return rows

def clean_pseudo_mask(pseudo_samples, stim_samples, fs, early=(0.0, 0.15), p300=(0.3, 0.6), evoked_end=0.8):
    P = np.asarray(pseudo_samples, float)[:, None]
    S = np.asarray(stim_samples, float)[None, :]
    e_lo, e_hi = (P + early[0] * fs, P + early[1] * fs)
    p_lo, p_hi = (P + p300[0] * fs, P + p300[1] * fs)
    ev_lo, ev_hi = (S, S + evoked_end * fs)
    early_hit = ((e_lo < ev_hi) & (e_hi > ev_lo)).any(axis=1)
    p300_hit = ((p_lo < ev_hi) & (p_hi > ev_lo)).any(axis=1)
    return ~(early_hit | p300_hit)

def selftest_hoist(seed=99):
    sf = 1024.0
    nc = int(600 * sf)
    ev = np.arange(int(5 * sf), nc - int(5 * sf), int(1.5 * sf))
    kw = dict(n_pseudo=175, sfreq=sf, n_continuous_samples=nc, real_event_samples=ev, min_gap_seconds=0.5, tmin=-0.2, tmax=0.8)
    r1 = np.random.default_rng(seed)
    a = [generate_pseudotrial_samples(rng=r1, **kw) for _ in range(5)]
    s1 = r1.integers(0, 10 ** 9, 5)
    fb = forbidden_mask(nc, ev, 0.5, sf)
    sc = np.zeros(nc, dtype=bool)
    r2 = np.random.default_rng(seed)
    b = [generate_pseudotrial_samples(rng=r2, forbidden=fb, scratch=sc, **kw) for _ in range(5)]
    s2 = r2.integers(0, 10 ** 9, 5)
    return all((np.array_equal(x, y) for x, y in zip(a, b))) and np.array_equal(s1, s2)
