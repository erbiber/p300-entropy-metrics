import numpy as np
from scipy import stats
from .features import entropy_block, feature_block, shape_block

def block(W):
    mean = W.mean(axis=1)
    sd = W.std(axis=1, ddof=0)
    rms = np.sqrt((W ** 2).mean(axis=1))
    return (mean, sd, rms)

def shape(W):
    n = W.shape[1]
    med = np.median(W, axis=1)
    out = {}
    out['median_amp'] = med
    out['mad_amp'] = np.median(np.abs(W - med[:, None]), axis=1)
    out['p2p_amp'] = W.max(axis=1) - W.min(axis=1)
    out['skew_amp'] = stats.skew(W, axis=1, bias=False)
    out['kurt_amp'] = stats.kurtosis(W, axis=1, fisher=True, bias=False)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    out['slope_amp'] = (W - W.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()
    return out

def hjorth(W):
    var_x = W.var(axis=1, ddof=0)
    dx = np.diff(W, axis=1)
    var_dx = dx.var(axis=1, ddof=0)
    ddx = np.diff(dx, axis=1)
    var_ddx = ddx.var(axis=1, ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mob = np.sqrt(var_dx / var_x)
        mob_dx = np.sqrt(var_ddx / var_dx)
        cplx = mob_dx / mob
    bad = ~np.isfinite(mob) | (var_x <= 0) | (W.shape[1] <= 2)
    mob = np.where(bad, np.nan, mob)
    cplx = np.where(bad | ~np.isfinite(cplx) | (mob <= 0), np.nan, cplx)
    return (mob, cplx)

def entropies(W):
    n = W.shape[0]
    pe = np.empty(n)
    se = np.empty(n)
    lz = np.empty(n)
    for i in range(n):
        pe[i], se[i], lz[i], _, _ = entropy_block(W[i])
    return (pe, se, lz)

def window_mask_early(times, cfg):
    from .features import window_mask
    return window_mask(times, cfg.EARLY_WINDOW)

def window_features(fz, pz, times, cfg, want_entropy=True, want_parity=False, groups=None):
    from .features import window_mask
    m_e = window_mask(times, cfg.EARLY_WINDOW)
    m_e200 = window_mask(times, cfg.EARLY_WINDOW_200)
    m_e250 = window_mask(times, cfg.EARLY_WINDOW_250)
    m_p = window_mask(times, cfg.P300_WINDOW)
    m_b = window_mask(times, cfg.BASELINE_CONTROL_WINDOW)
    r = {}
    r['mean_early_fz'], r['sd_early_fz'], r['rms_early_fz'] = block(fz[:, m_e])
    r['mean_early_fz_200'], _, r['rms_early_fz_200'] = block(fz[:, m_e200])
    r['mean_early_fz_250'], _, r['rms_early_fz_250'] = block(fz[:, m_e250])
    r['mean_base_fz'], _, r['rms_base_fz'] = block(fz[:, m_b])
    r['mean_early_pz'], r['sd_early_pz'], r['rms_early_pz'] = block(pz[:, m_e])
    r['mean_base_pz'], _, r['rms_base_pz'] = block(pz[:, m_b])
    r['p300_pz'] = pz[:, m_p].mean(axis=1)
    r['ms_early_pz'] = r['rms_early_pz'] ** 2
    for k, v in shape(fz[:, m_e]).items():
        r[k + '_fz'] = v
    mob, cplx = hjorth(fz[:, m_e])
    r['hjmob_early_fz'] = mob
    r['hjcplx_early_fz'] = cplx
    if want_entropy:
        pe, se, lz = entropies(fz[:, m_e])
        r['pe_early_fz'] = pe
        r['se_early_fz'] = se
        r['lz_early_fz'] = lz
    if want_parity:
        from .parity import window_set
        sf = 1.0 / float(times[1] - times[0])
        r.update(window_set(fz, pz, times, cfg, sf, groups))
    return r

def selftest(cfg, n=12, seed=5, sfreq=1024.0):
    from .features import trial_features
    rng = np.random.default_rng(seed)
    lo = int(round(cfg.TMIN_EPOCH * sfreq))
    hi = int(round(cfg.TMAX_EPOCH * sfreq))
    times = np.arange(lo, hi + 1) / sfreq
    fz = rng.standard_normal((n, times.size)) * 1e-05
    pz = rng.standard_normal((n, times.size)) * 1e-05
    fast = window_features(fz, pz, times, cfg)
    bad = []
    for i in range(n):
        slow = trial_features(fz[i], pz[i], times, cfg)
        for k, v in slow.items():
            a, b = (float(v), float(fast[k][i]))
            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-12 * max(1.0, abs(a)):
                bad.append((k, i, a, b))
    return bad

def parity_classification(seed=13):
    from .parity import classify, block as pblock
    rng = np.random.default_rng(seed)
    out = []
    for sf in (1024.0, 500.0):
        n = int(round(0.15 * sf)) + 1
        t = np.linspace(0, 9, n)
        W = (np.cumsum(rng.standard_normal((12, n)), axis=1) + 3 * np.sin(t) ** 3) * 1e-06
        fns = {}
        fns['mean'] = lambda X: block(X)[0]
        fns['sd'] = lambda X: block(X)[1]
        fns['rms'] = lambda X: block(X)[2]
        for k in ('median_amp', 'mad_amp', 'p2p_amp', 'skew_amp', 'kurt_amp', 'slope_amp'):
            fns[k] = lambda X, k=k: shape(X)[k]
        fns['hjorth_mobility'] = lambda X: hjorth(X)[0]
        fns['hjorth_complexity'] = lambda X: hjorth(X)[1]
        fns['permutation_entropy'] = lambda X: entropies(X)[0]
        fns['sample_entropy'] = lambda X: entropies(X)[1]
        fns['lempel_ziv'] = lambda X: entropies(X)[2]
        for k in ('tasym', 'updown', 'klasym', 'slope', 'dkurt', 'mono', 'klsym'):
            fns['parity_' + k] = lambda X, k=k: pblock(X, sf)[k]
        for name, fn in fns.items():
            out.append((sf, name, classify(fn, W)))
    return out
