import numpy as np
from scipy import stats
MIN_PSEUDO_DRAWS = 200
BCA_B = 4000
BCA_SEED = 12345
SURROGATE_B = 2000
SURROGATE_SEED = 12345

def robust_z_within_subject(s):
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nan)
    med = np.nanmedian(s)
    mad = 1.4826 * np.nanmedian(np.abs(s - med))
    if np.isfinite(mad) and mad > 0:
        return ((s - med) / mad, True)
    return (s - med, False)

def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = (x[ok], y[ok])
    if x.size < 3:
        return np.nan
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    if den <= 0:
        return np.nan
    return float((xc * (y - y.mean())).sum() / den)

def bca_ci(d, B=BCA_B, alpha=0.05, seed=BCA_SEED):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return (float(np.mean(d)) if n else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    theta = d.mean()
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    prop = np.mean(boot < theta)
    z0 = stats.norm.ppf(min(max(prop, 1e-06), 1 - 1e-06))
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jm = jack.mean()
    den = 6 * np.sum((jm - jack) ** 2) ** 1.5
    a = np.sum((jm - jack) ** 3) / den if den != 0 else 0.0
    zl, zu = (stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2))
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return (float(theta), float(np.percentile(boot, 100 * adj(zl))), float(np.percentile(boot, 100 * adj(zu))))

def surrogate_tail_fraction(real_slopes, pseudo_matrix):
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
    return dict(beta_real=beta_real, beta_pseudo_mean=float(np.nanmean(bpk)), beta_pseudo_sd=float(np.nanstd(bpk)), dbeta=dbeta, d=d, tail_fraction=float(p))

def mixedlm_beta_and_marginal_r2(feat, p300, subject):
    import pandas as pd
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings('ignore')
    df = pd.DataFrame(dict(feat=feat, p300=p300, subject=subject))
    df['zf'] = df.groupby('subject')['feat'].transform(lambda s: robust_z_within_subject(s)[0])
    df['zp'] = df.groupby('subject')['p300'].transform(lambda s: robust_z_within_subject(s)[0])
    df = df.dropna(subset=['zf', 'zp'])
    if df['subject'].nunique() < 3 or len(df) < 10:
        return (np.nan, np.nan, int(df['subject'].nunique()))
    m = smf.mixedlm('zp ~ zf', df, groups=df['subject']).fit(method='lbfgs')
    beta = float(m.params.get('zf', np.nan))
    var_f = beta ** 2 * float(np.var(df['zf'].values, ddof=0))
    var_a = float(m.cov_re.iloc[0, 0]) if m.cov_re.shape[0] > 0 else 0.0
    var_e = float(m.scale)
    denom = var_f + var_a + var_e
    r2m = var_f / denom if denom > 0 else np.nan
    return (float(r2m), beta, int(df['subject'].nunique()))

def dataset_feature_interaction(feat, p300, subject, dataset):
    import pandas as pd
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings('ignore')
    df = pd.DataFrame(dict(feat=feat, p300=p300, subject=subject, dataset=dataset))
    df['zf'] = df.groupby('subject')['feat'].transform(lambda s: robust_z_within_subject(s)[0])
    df['zp'] = df.groupby('subject')['p300'].transform(lambda s: robust_z_within_subject(s)[0])
    m = smf.mixedlm('zp ~ zf * C(dataset)', df, groups=df['subject']).fit(method='lbfgs')
    term = [t for t in m.params.index if t.startswith('zf:')]
    if not term:
        return (np.nan, np.nan)
    return (float(m.params[term[0]]), float(m.pvalues[term[0]]))
