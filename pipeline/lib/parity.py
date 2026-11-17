import numpy as np

LAG_MS = 8.0
ODD = ('tasym', 'updown', 'klasym', 'slope')
EVEN = ('dkurt', 'mono', 'klsym')


def lag_samples(sfreq, ms=LAG_MS):
    return max(1, int(round(ms * 1e-3 * sfreq)))


def diff_moments(W, tau):
    d = W[:, tau:] - W[:, :-tau]
    m2 = (d ** 2).mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = (d ** 3).mean(axis=1) / m2 ** 1.5
        k = (d ** 4).mean(axis=1) / m2 ** 2
    bad = ~(m2 > 0)
    return (np.where(bad, np.nan, a), np.where(bad, np.nan, k))


def ordinal(W, tau):
    a = W[:, :-2 * tau]
    b = W[:, tau:-tau]
    c = W[:, 2 * tau:]
    up = ((a < b) & (b < c)).sum(axis=1)
    dn = ((a > b) & (b > c)).sum(axis=1)
    n = a.shape[1]
    return ((up - dn) / n, (up + dn) / n)


def halves_kl(W):
    h = W.shape[1] // 2
    A = W[:, :h]
    B = W[:, -h:]
    ma = A.mean(axis=1)
    mb = B.mean(axis=1)
    va = A.var(axis=1, ddof=1)
    vb = B.var(axis=1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        kba = 0.5 * (np.log(va / vb) + (vb + (mb - ma) ** 2) / va - 1)
        kab = 0.5 * (np.log(vb / va) + (va + (ma - mb) ** 2) / vb - 1)
    bad = ~((va > 0) & (vb > 0))
    return (np.where(bad, np.nan, kba - kab), np.where(bad, np.nan, kba + kab))


def slope(W):
    n = W.shape[1]
    xc = np.arange(n, dtype=float) - (n - 1) / 2
    return (W - W.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()


def loo_residual(W, groups):
    R = np.array(W, dtype=float, copy=True)
    g = np.asarray(groups)
    for v in np.unique(g):
        i = g == v
        n = int(i.sum())
        if n < 2:
            continue
        s = W[i].sum(axis=0)
        R[i] = W[i] - (s[None, :] - W[i]) / (n - 1)
    return R


def block(W, sfreq, odd_only=False):
    tau = lag_samples(sfreq)
    ta, dk = diff_moments(W, tau)
    ud, mo = ordinal(W, tau)
    ka, ks = halves_kl(W)
    out = dict(tasym=ta, updown=ud, klasym=ka, slope=slope(W))
    if not odd_only:
        out.update(dkurt=dk, mono=mo, klsym=ks)
    return out


def window_set(fz, pz, times, cfg, sfreq, groups=None):
    from .features import window_mask
    n = fz.shape[0]
    g = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    m_e = window_mask(times, cfg.EARLY_WINDOW)
    m_b = window_mask(times, cfg.BASELINE_CONTROL_WINDOW)
    r = {}
    for k, v in block(loo_residual(fz[:, m_e], g), sfreq).items():
        r[f'par_{k}_fz'] = v
    for k, v in block(loo_residual(fz[:, m_b], g), sfreq, odd_only=True).items():
        r[f'par_{k}_bfz'] = v
    for k, v in block(loo_residual(pz[:, m_e], g), sfreq, odd_only=True).items():
        r[f'par_{k}_pz'] = v
    for k, v in block(fz[:, m_e], sfreq, odd_only=True).items():
        if k != 'slope':
            r[f'par_{k}_fz_raw'] = v
    return r


COLUMNS = (['par_tasym_fz', 'par_updown_fz', 'par_klasym_fz', 'par_slope_fz',
            'par_dkurt_fz', 'par_mono_fz', 'par_klsym_fz'] +
           [f'par_{k}_bfz' for k in ODD] + [f'par_{k}_pz' for k in ODD] +
           ['par_tasym_fz_raw', 'par_updown_fz_raw', 'par_klasym_fz_raw'])


def classify(fn, W, tol=1e-9):
    a = np.asarray(fn(W), dtype=float)
    b = np.asarray(fn(W[:, ::-1].copy()), dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return 'undefined'
    a = a[ok]
    b = b[ok]
    s = np.maximum(1.0, np.abs(a))
    if np.all(np.abs(a - b) <= tol * s):
        return 'even'
    if np.all(np.abs(a + b) <= tol * s):
        return 'odd'
    rel = float(np.mean(np.abs(a - b)) / max(np.mean(np.abs(a)), 1e-300))
    return f'neither (mean |F(x)-F(rev x)| / mean |F| = {rel:.3f})'


def selftest(seed=11):
    rng = np.random.default_rng(seed)
    fails = []
    for sf in (1024.0, 500.0):
        n = int(round(0.15 * sf)) + 1
        t = np.linspace(0, 9, n)
        W = np.cumsum(rng.standard_normal((40, n)), axis=1) + 3 * np.sin(t) ** 3
        for k in ODD + EVEN:
            c = classify(lambda X, k=k: block(X, sf)[k], W)
            want = 'odd' if k in ODD else 'even'
            if c != want:
                fails.append(f'{k} at {sf:g} Hz is {c}, expected {want}')
        X = rng.standard_normal((30, n))
        erp = np.linspace(-1, 3, n) ** 2
        g = (rng.random(30) < 0.3).astype(int)
        if not np.allclose(loo_residual(X + erp, g), loo_residual(X, g), atol=1e-12):
            fails.append(f'leave-one-out residual does not remove an additive constant waveform at {sf:g} Hz')
    return fails
