import numpy as np
import phase013_engine as E

def robust_z_within_subject(s):
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nan)
    med = np.nanmedian(s)
    mad = 1.4826 * np.nanmedian(np.abs(s - med))
    return (s - med) / mad if (np.isfinite(mad) and mad > 0) else (s - med)

def _ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    zx = robust_z_within_subject(x)
    zy = robust_z_within_subject(y)
    m = np.isfinite(zx) & np.isfinite(zy)
    if m.sum() < 3:
        return np.nan
    zxm = zx[m] - zx[m].mean()
    zym = zy[m] - zy[m].mean()
    den = zxm @ zxm
    return float((zxm @ zym) / den) if den > 1e-12 else np.nan

E.robust_z_within_subject = robust_z_within_subject
E._ols_slope = _ols_slope

print("[phase013_nan_safe_zscore] robust_z_within_subject and _ols_slope patched (NaN-safe)")
