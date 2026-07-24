"""
phase013_nan_safe_zscore.py
---------------------------
Repairs the M14 (sample entropy) all-NaN failure without touching M1/M4a/M8/M9a.

Root cause: a single non-finite trial (sample_entropy() occasionally returns inf/NaN
on short windows) poisons a subject's entire z-scored array, because both
robust_z_within_subject (median/MAD) and _ols_slope (a second .mean() centering)
propagate NaN across the whole vector. One bad trial -> whole subject dropped ->
n_subjects = 0.

Fix: make both functions ignore individual non-finite trials instead of propagating
them. A subject keeps its clean trials; only the genuinely bad trials drop.

Verified against the deposited phase013_engine.py:
  - byte-identical output to the original on fully clean data (max diff 0.0 over
    200 random subjects), so the four headline models are provably unaffected;
  - a subject with one NaN trial now yields a finite slope instead of NaN;
  - an all-NaN feature still returns NaN (no fabrication).

USAGE — put this file next to phase013_engine.py, then in run_phase013.py add it
immediately AFTER the complexity patch:

    import phase013_engine as E
    import phase013_complexity_patch      # adds M13-M15, M22, M23
    import phase013_nan_safe_zscore       # repairs their NaN handling
"""
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


# marginal_r2 / dataset_feature_interaction call robust_z_within_subject through a
# pandas .transform and drop NaN rows themselves, so rebinding the engine name is
# enough for them; _ols_slope must be rebound explicitly because it is referenced
# directly inside subject_slopes_real_and_pseudo.
E.robust_z_within_subject = robust_z_within_subject
E._ols_slope = _ols_slope

print("[phase013_nan_safe_zscore] robust_z_within_subject and _ols_slope patched (NaN-safe)")
