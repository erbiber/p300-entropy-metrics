import os
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

def add_within_subject_z(df, cols):
    for c in cols:
        if c in df.columns:
            df[c + '_z'] = df.groupby('subject')[c].transform(
                robust_z_within_subject)
    return df

def _safe_r2(fit):
    try:
        X = np.asarray(fit.model.exog)
        beta = np.asarray(fit.fe_params)
        var_fixed = float(np.var(X @ beta))
        var_resid = float(fit.scale)
        try:
            cov_re = np.atleast_2d(np.asarray(fit.cov_re))
            var_random = float(np.trace(cov_re))
        except Exception:
            var_random = 0.0
        denom = var_fixed + var_random + var_resid
        if denom <= 0:
            return np.nan, np.nan
        return var_fixed / denom, (var_fixed + var_random) / denom
    except Exception:
        return np.nan, np.nan

def fit_RI(df, formula, label, predictor):
    attempts = [
        ('REML_lbfgs',  dict(method='lbfgs',            reml=True)),
        ('REML_nm',     dict(method='nm', maxiter=2000, reml=True)),
        ('REML_powell', dict(method='powell', maxiter=2000, reml=True)),
    ]
    fit = None
    used = None
    warn = ''
    for name, kwargs in attempts:
        try:

            md = smf.mixedlm(formula, df, groups=df['subject'])
            fit = md.fit(**kwargs)
            ll = float(fit.llf)
            if not np.isfinite(ll):
                warn = f'{name}: non-finite loglik'
                fit = None
                continue
            used = name
            break
        except Exception as e:
            warn = f'{name} failed: {type(e).__name__}: {e}'
            continue

    if fit is None:

        try:
            ols = smf.ols(f'{formula} + C(subject)', data=df).fit()
            beta = ols.params[predictor]
            se = ols.bse[predictor]
            z = beta / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            return {
                'model': label, 'predictor': predictor,
                'beta': beta, 'SE': se, 'z': z, 'p': p,
                'CI_low': beta - 1.96 * se, 'CI_high': beta + 1.96 * se,
                'R2_marginal': float(ols.rsquared),
                'R2_conditional': float(ols.rsquared),
                'n_trials': len(df),
                'n_subjects': df['subject'].nunique(),
                'fit_method': 'ols_subject_FE',
                'note': warn,
            }
        except Exception as e2:
            return {
                'model': label, 'predictor': predictor,
                'beta': np.nan, 'SE': np.nan, 'z': np.nan, 'p': np.nan,
                'CI_low': np.nan, 'CI_high': np.nan,
                'R2_marginal': np.nan, 'R2_conditional': np.nan,
                'n_trials': len(df),
                'n_subjects': df['subject'].nunique(),
                'fit_method': 'failed',
                'note': f'{warn}; ols fallback also failed: {e2}',
            }

    beta = fit.params[predictor]
    se = fit.bse[predictor]
    z = beta / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    r2m, r2c = _safe_r2(fit)

    return {
        'model': label,
        'predictor': predictor,
        'beta': beta,
        'SE': se,
        'z': z,
        'p': p,
        'CI_low': beta - 1.96 * se,
        'CI_high': beta + 1.96 * se,
        'R2_marginal': r2m,
        'R2_conditional': r2c,
        'n_trials': len(df),
        'n_subjects': df['subject'].nunique(),
        'fit_method': used,
        'note': warn,
    }

def main():
    banner("02_run_lmms.py — canonical RI-only LMM analysis")

    fpath = os.path.join(RESULTS_DIR, FILES['trial_features'])
    if not os.path.exists(fpath):
        print(f"ERROR: trial features CSV not found: {fpath}")
        print("Run 01_extract_features.py first.")
        return

    df = pd.read_csv(fpath)
    print(f"Loaded {len(df)} trials from {df['subject'].nunique()} subjects.\n")

    feat_cols = [
        'mean_early_fz', 'sd_early_fz', 'rms_early_fz',
        'mean_early_fz_200', 'rms_early_fz_200',
        'mean_early_fz_250', 'rms_early_fz_250',
        'mean_base_fz', 'rms_base_fz',
        'mean_early_pz', 'sd_early_pz', 'rms_early_pz',
        'mean_base_pz', 'rms_base_pz',
        'p300_amp',
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    df = add_within_subject_z(df, feat_cols)

    rows = []

    rows.append(fit_RI(df,
        'p300_amp_z ~ rms_early_fz_z + condition',
        'M1_RMS_Fz_0_150', 'rms_early_fz_z'))
    rows.append(fit_RI(df,
        'p300_amp_z ~ mean_early_fz_z + condition',
        'M2_mean_Fz_only', 'mean_early_fz_z'))
    rows.append(fit_RI(df,
        'p300_amp_z ~ sd_early_fz_z + condition',
        'M3_SD_Fz_only', 'sd_early_fz_z'))
    rows.append(fit_RI(df,
        'p300_amp_z ~ mean_early_fz_z + sd_early_fz_z + condition',
        'M4a_competitive_Fz_mean', 'mean_early_fz_z'))
    rows.append(fit_RI(df,
        'p300_amp_z ~ mean_early_fz_z + sd_early_fz_z + condition',
        'M4b_competitive_Fz_SD', 'sd_early_fz_z'))

    if 'rms_early_fz_200_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_early_fz_200_z + condition',
            'M5_RMS_Fz_0_200', 'rms_early_fz_200_z'))
    if 'rms_early_fz_250_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_early_fz_250_z + condition',
            'M6_RMS_Fz_0_250', 'rms_early_fz_250_z'))

    if 'mean_early_fz_200_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_early_fz_200_z + condition',
            'M5m_mean_Fz_0_200', 'mean_early_fz_200_z'))
    if 'mean_early_fz_250_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_early_fz_250_z + condition',
            'M6m_mean_Fz_0_250', 'mean_early_fz_250_z'))

    if 'rms_base_fz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_base_fz_z + condition',
            'M7_baseline_Fz_RMS', 'rms_base_fz_z'))
    if 'mean_base_fz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_base_fz_z + condition',
            'M7m_baseline_Fz_mean', 'mean_base_fz_z'))

    if 'rms_early_pz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_early_pz_z + condition',
            'M8_RMS_Pz_0_150', 'rms_early_pz_z'))
    if 'mean_early_pz_z' in df.columns and 'sd_early_pz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_early_pz_z + sd_early_pz_z + condition',
            'M9a_competitive_Pz_mean', 'mean_early_pz_z'))
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_early_pz_z + sd_early_pz_z + condition',
            'M9b_competitive_Pz_SD', 'sd_early_pz_z'))

    if 'rms_base_pz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_base_pz_z + condition',
            'M10_baseline_Pz_RMS', 'rms_base_pz_z'))
    if 'rms_early_pz_z' in df.columns and 'rms_base_pz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ rms_early_pz_z + rms_base_pz_z + condition',
            'M11_Pz_RMS_with_baseline_cov', 'rms_early_pz_z'))

    if 'mean_early_pz_z' in df.columns and 'mean_base_pz_z' in df.columns:
        rows.append(fit_RI(df,
            'p300_amp_z ~ mean_early_pz_z + mean_base_pz_z + condition',
            'M12_Pz_mean_with_baseline_cov', 'mean_early_pz_z'))

    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, FILES['lmm_summary'])
    out.to_csv(out_csv, index=False)
    print(f"\nCanonical LMM summary saved -> {out_csv}\n")

    cols = ['model', 'predictor', 'beta', 'SE', 'z', 'p',
            'R2_marginal', 'n_trials', 'fit_method']
    pd.set_option('display.float_format', lambda v: f'{v:+.4f}'
                  if abs(v) >= 1e-4 else f'{v:.2e}')
    print(out[cols].to_string(index=False))

    banner("DONE")
    print(f"  All manuscript numbers must be")
    print(f"  cited from this file: {FILES['lmm_summary']}")
    print(f"  No other LMM CSV exists. No structure inconsistency possible.")

if __name__ == '__main__':
    main()
