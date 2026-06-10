\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from config import RESULTS_DIR, FILES, banner

warnings.filterwarnings('ignore', category=Warning)

def robust_z_within_subject(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else s - med

def fit_RI(df, formula, predictor):
\
                                 
    for kwargs in [
        dict(method='lbfgs', reml=True),
        dict(method='nm', maxiter=2000, reml=True),
        dict(method='powell', maxiter=2000, reml=True),
    ]:
        try:
            md = smf.mixedlm(formula, df, groups=df['subject'])
            fit = md.fit(**kwargs)
            ll = float(fit.llf)
            if not np.isfinite(ll):
                continue
            beta = float(fit.params[predictor])
            se = float(fit.bse[predictor])
            z = beta / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            X = np.asarray(fit.model.exog)
            betas = np.asarray(fit.fe_params)
            var_f = float(np.var(X @ betas))
            var_e = float(fit.scale)
            try:
                cov_re = np.atleast_2d(np.asarray(fit.cov_re))
                var_r = float(np.trace(cov_re))
            except Exception:
                var_r = 0.0
            denom = var_f + var_r + var_e
            r2m = var_f / denom if denom > 0 else np.nan
            return beta, se, z, p, r2m
        except Exception:
            continue
    return np.nan, np.nan, np.nan, np.nan, np.nan

def main():
    banner("03_per_electrode_lmm.py — per-electrode coupling topomap")

    fpath = os.path.join(RESULTS_DIR, FILES['trial_features_per_elec'])
    if not os.path.exists(fpath):
        print(f"ERROR: per-electrode CSV not found: {fpath}")
        print("Run 01_extract_features.py first.")
        return

    df = pd.read_csv(fpath)
    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects.\n")

    rms_cols = [c for c in df.columns if c.startswith('rms_')
                and c not in ('rms_early', 'rms_base')]
                           
    channels = [(c, c[len('rms_'):]) for c in rms_cols]
                                                                       
    channels = [(col, ch) for col, ch in channels
                if re.match(r'^[A-Za-z]+\d*[A-Za-z]?$', ch)]

    print(f"  Found {len(channels)} electrodes.\n")

                               
    df['p300_amp_z'] = df.groupby('subject')['p300_amp'].transform(
        robust_z_within_subject)

    rows = []
    print("Fitting RI LMM at each electrode (RMS early -> Pz P300, both robust-z)...")
    print("-" * 72)
    for col, ch in channels:
        sub = df[['subject', 'condition', col, 'p300_amp_z']].dropna().copy()
        if len(sub) < 100:
            continue
        sub['x_z'] = sub.groupby('subject')[col].transform(
            robust_z_within_subject)
        beta, se, z, p, r2m = fit_RI(sub,
                                          , 'x_z')
        rows.append({
                     : ch,
                  : beta, 'SE': se, 'z': z, 'p': p,
                         : r2m,
                      : len(sub),
                        : sub['subject'].nunique(),
        })
        print(f"  {ch:6s}  beta={beta:+.4f}  SE={se:.4f}  "
              f"z={z:+.2f}  p={p:.2e}  R2m={r2m:.3f}")

    if not rows:
        print("\nNo rows produced. Aborting.")
        return

    out = pd.DataFrame(rows)
    out['abs_beta'] = out['beta'].abs()
    out = out.sort_values('abs_beta', ascending=False).drop(columns='abs_beta')
    out_csv = os.path.join(RESULTS_DIR, FILES['per_electrode_summary'])
    out.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    print("\nTop 10 channels by |beta|:")
    print(out.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
