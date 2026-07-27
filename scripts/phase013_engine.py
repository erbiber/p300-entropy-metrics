import numpy as np
from scipy import stats

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def robust_z_within_subject(s):
    s = np.asarray(s, float)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

def feature_block(trace):
    mean_amp = float(np.mean(trace))
    sd_amp = float(np.std(trace - mean_amp, ddof=0))
    rms_amp = float(np.sqrt(np.mean(trace ** 2)))
    return mean_amp, sd_amp, rms_amp

def generate_pseudotrial_samples(n_pseudo, sfreq, n_continuous_samples,
                                 real_event_samples, min_gap_seconds,
                                 tmin, tmax, rng, attempt_factor=20, attempt_floor=5000):
    min_gap_samples = int(min_gap_seconds * sfreq)
    epoch_samples = int((tmax - tmin) * sfreq)
    pre_buffer = int(abs(tmin) * sfreq) + 1
    post_buffer = int(tmax * sfreq) + 1
    valid_low = pre_buffer
    valid_high = n_continuous_samples - post_buffer
    if valid_high <= valid_low:
        return np.array([], dtype=int)
    forbidden = np.zeros(n_continuous_samples, dtype=bool)
    for ev in real_event_samples:
        lo = max(0, int(ev) - min_gap_samples)
        hi = min(n_continuous_samples, int(ev) + min_gap_samples)
        forbidden[lo:hi] = True
    placed = []
    blocked = np.zeros(n_continuous_samples, dtype=bool)
    for _ in range(max(int(n_pseudo * attempt_factor), attempt_floor)):
        if len(placed) >= n_pseudo:
            break
        cand = int(rng.integers(valid_low, valid_high))
        if forbidden[cand] or blocked[cand]:
            continue
        placed.append(cand)
        lo = max(0, cand - epoch_samples)
        hi = min(n_continuous_samples, cand + epoch_samples)
        blocked[lo:hi] = True
    placed.sort()
    return np.array(placed, dtype=int)

def _ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return np.nan
    zx = robust_z_within_subject(x); zy = robust_z_within_subject(y)
    zx = zx - zx.mean(); zy = zy - zy.mean()
    return (zx @ zy) / (zx @ zx) if (zx @ zx) > 1e-12 else np.nan

def features_from_epochs(data, times, fz_idx, pz_idx, early_win, p300_win):
    me = window_mask(times, early_win); mp = window_mask(times, p300_win)
    fz = data[:, fz_idx, :]; pz = data[:, pz_idx, :]
    return dict(
        mean_early_fz=fz[:, me].mean(1),
        rms_early_fz=np.sqrt((fz[:, me] ** 2).mean(1)),
        mean_early_pz=pz[:, me].mean(1),
        rms_early_pz=np.sqrt((pz[:, me] ** 2).mean(1)),
        p300_pz=pz[:, mp].mean(1),
    )

MODELS = {
    'M1_RMS_Fz_0_150':  ('rms_early_fz',  'p300_pz'),
    'M4a_mean_Fz':      ('mean_early_fz', 'p300_pz'),
    'M8_RMS_Pz_0_150':  ('rms_early_pz',  'p300_pz'),
    'M9a_mean_Pz':      ('mean_early_pz', 'p300_pz'),
}

def subject_slopes_real_and_pseudo(real_feats, pseudo_feats_list):
    real_slope = {}; pseudo_slopes = {m: [] for m in MODELS}
    for m, (fx, fy) in MODELS.items():
        real_slope[m] = _ols_slope(real_feats[fx], real_feats[fy])
    for pf in pseudo_feats_list:
        for m, (fx, fy) in MODELS.items():
            pseudo_slopes[m].append(_ols_slope(pf[fx], pf[fy]))
    pseudo_slopes = {m: np.array(v, float) for m, v in pseudo_slopes.items()}
    return real_slope, pseudo_slopes

def contrast_from_matrix(real_slopes, pseudo_matrix):
    real_slopes = np.asarray(real_slopes, float)
    P = np.asarray(pseudo_matrix, float)
    beta_pseudo_k = np.nanmean(P, axis=1)
    pbar = np.nanmean(P, axis=0)
    d = real_slopes - pbar
    beta_real = float(np.nanmean(real_slopes))
    dbeta = float(np.nanmean(d))
    bpk = beta_pseudo_k[~np.isnan(beta_pseudo_k)]
    if beta_real >= 0:
        p = (1 + np.sum(bpk >= beta_real)) / (len(bpk) + 1)
    else:
        p = (1 + np.sum(bpk <= beta_real)) / (len(bpk) + 1)
    return dict(beta_real=beta_real, beta_pseudo_mean=float(np.nanmean(bpk)),
                beta_pseudo_sd=float(np.nanstd(bpk)), dbeta=dbeta, d=d, surrogate_p=float(p))

def bca_ci(d, B=4000, alpha=0.05, seed=12345):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3: return (float(np.mean(d)) if n else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed); theta = d.mean()
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    prop = np.mean(boot < theta); z0 = stats.norm.ppf(min(max(prop, 1e-6), 1 - 1e-6))
    jack = np.array([np.delete(d, i).mean() for i in range(n)]); jm = jack.mean()
    den = 6 * (np.sum((jm - jack) ** 2) ** 1.5); a = np.sum((jm - jack) ** 3) / den if den != 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return float(theta), float(np.percentile(boot, 100 * adj(zl))), float(np.percentile(boot, 100 * adj(zu)))

def dataset_feature_interaction(feat, p300, subject, dataset):
    import pandas as pd, statsmodels.formula.api as smf, warnings
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(dict(feat=feat, p300=p300, subject=subject, dataset=dataset))
    df["zf"] = df.groupby("subject")["feat"].transform(robust_z_within_subject)
    df["zp"] = df.groupby("subject")["p300"].transform(robust_z_within_subject)
    m = smf.mixedlm("zp ~ zf * C(dataset)", df, groups=df["subject"]).fit(method="lbfgs")
    term = [t for t in m.params.index if t.startswith("zf:")]
    if not term: return np.nan, np.nan
    return float(m.params[term[0]]), float(m.pvalues[term[0]])

def marginal_r2(feat, p300, subject):
    import pandas as pd, statsmodels.formula.api as smf, warnings
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(dict(feat=feat, p300=p300, subject=subject))
    df["zf"] = df.groupby("subject")["feat"].transform(robust_z_within_subject)
    df["zp"] = df.groupby("subject")["p300"].transform(robust_z_within_subject)
    df = df.dropna(subset=["zf", "zp"])
    if df["subject"].nunique() < 3 or len(df) < 10:
        return np.nan, np.nan, int(df["subject"].nunique())
    m = smf.mixedlm("zp ~ zf", df, groups=df["subject"]).fit(method="lbfgs")
    beta = float(m.params.get("zf", np.nan))
    var_f = beta ** 2 * float(np.var(df["zf"].values, ddof=0))
    var_a = float(m.cov_re.iloc[0, 0]) if m.cov_re.shape[0] > 0 else 0.0
    var_e = float(m.scale)
    denom = var_f + var_a + var_e
    r2m = var_f / denom if denom > 0 else np.nan
    return float(r2m), beta, int(df["subject"].nunique())

def clean_pseudo_mask(pseudo_samples, stim_samples, fs, early=(0.0,0.150), p300=(0.300,0.600),
                      evoked_end=0.8):
    P = np.asarray(pseudo_samples, float)[:, None]
    S = np.asarray(stim_samples, float)[None, :]
    e_lo, e_hi = P + early[0]*fs, P + early[1]*fs
    p_lo, p_hi = P + p300[0]*fs, P + p300[1]*fs
    ev_lo, ev_hi = S, S + evoked_end*fs
    early_hit = ((e_lo < ev_hi) & (e_hi > ev_lo)).any(axis=1)
    p300_hit  = ((p_lo < ev_hi) & (p_hi > ev_lo)).any(axis=1)
    return ~(early_hit | p300_hit)

def contrast_ragged(real_slopes, pseudo_arrays, min_draws=1, n_surrogate=2000, seed=12345):
    real_slopes = np.asarray(real_slopes, float)
    pools = []; reals = []
    for r, a in zip(real_slopes, pseudo_arrays):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        if len(a) >= min_draws and np.isfinite(r):
            pools.append(a); reals.append(r)
    n_used = len(reals); n_dropped = len(real_slopes) - n_used
    if n_used < 3:
        return dict(beta_real=np.nan, n_used=n_used, n_dropped=n_dropped,
                    beta_pseudo_mean=np.nan, beta_pseudo_sd=np.nan, dbeta=np.nan,
                    d=np.array([]), surrogate_p=np.nan, draws_min=0, draws_median=0)
    reals = np.array(reals)
    pbar = np.array([a.mean() for a in pools])
    d = reals - pbar
    beta_real = float(reals.mean())
    rng = np.random.default_rng(seed)
    surro = np.empty(n_surrogate)
    for b in range(n_surrogate):
        surro[b] = np.mean([a[rng.integers(0, len(a))] for a in pools])
    if beta_real >= 0:
        p = (1 + np.sum(surro >= beta_real)) / (n_surrogate + 1)
    else:
        p = (1 + np.sum(surro <= beta_real)) / (n_surrogate + 1)
    draws = [len(a) for a in pools]
    return dict(beta_real=beta_real, n_used=n_used, n_dropped=n_dropped,
                beta_pseudo_mean=float(surro.mean()), beta_pseudo_sd=float(surro.std()),
                dbeta=float(d.mean()), d=d, surrogate_p=float(p),
                draws_min=int(min(draws)), draws_median=int(np.median(draws)))
