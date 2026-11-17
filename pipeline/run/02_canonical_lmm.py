import argparse
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import config as cfg
import spec
from lib.features import COLUMNS
warnings.filterwarnings('ignore')

def robust_z(s):
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nan)
    med = np.nanmedian(s)
    mad = 1.4826 * np.nanmedian(np.abs(s - med))
    if np.isfinite(mad) and mad > 0:
        return ((s - med) / mad, True)
    return (s - med, False)

def standardise(df, columns):
    out = df.copy()
    unscaled = []
    for c in columns:
        if c not in out.columns:
            continue
        zs = np.full(len(out), np.nan)
        for sub, idx in out.groupby('subject').indices.items():
            z, ok = robust_z(out[c].values[idx])
            zs[idx] = z
            if not ok:
                unscaled.append(dict(subject=sub, column=c))
        out[c + '_z'] = zs
    return (out, pd.DataFrame(unscaled))

def fit_one(df, model):
    pred = model.predictor + '_z'
    terms = [pred] + [c + '_z' if c != 'condition' else 'condition' for c in model.covariates]
    need = [t for t in terms if t != 'condition'] + ['p300_pz_z', 'subject']
    sub = df.dropna(subset=need).copy()
    if sub['subject'].nunique() < 3 or len(sub) < cfg.MIN_TRIALS_REQUIRED:
        return dict(model_id=model.id, beta_lmm=np.nan, se=np.nan, z=np.nan, p=np.nan, r2_marginal=np.nan, n_trials=len(sub), n_subjects=int(sub['subject'].nunique()), fit_method='', fit_note='', status='too_few')
    if 'condition' in terms and sub['condition'].nunique() < 2:
        terms = [t for t in terms if t != 'condition']
        note = 'condition dropped: only one level present'
    else:
        note = 'ok'
    formula = 'p300_pz_z ~ ' + ' + '.join(terms)
    md = smf.mixedlm(formula, sub, groups=sub['subject'])
    fit_method = 'REML_lbfgs'
    fit_note = ''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m = md.fit(method='lbfgs')
        except Exception as exc:
            m = None
            fit_note = f'REML_lbfgs raised {type(exc).__name__}'
    if m is None or not getattr(m, 'converged', True):
        if not fit_note:
            fit_note = 'REML_lbfgs: did not converge'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = md.fit(method='nm', maxiter=2000)
        fit_method = 'REML_nm'
        if not getattr(m, 'converged', True):
            note = 'did not converge under either optimiser'
    beta = float(m.params.get(pred, np.nan))
    var_f = beta ** 2 * float(np.var(sub[pred].values, ddof=0))
    var_a = float(m.cov_re.iloc[0, 0]) if m.cov_re.shape[0] > 0 else 0.0
    var_e = float(m.scale)
    den = var_f + var_a + var_e
    return dict(model_id=model.id, beta_lmm=beta, se=float(m.bse.get(pred, np.nan)), z=float(m.tvalues.get(pred, np.nan)), p=float(m.pvalues.get(pred, np.nan)), r2_marginal=var_f / den if den > 0 else np.nan, n_trials=int(len(sub)), n_subjects=int(sub['subject'].nunique()), fit_method=fit_method, fit_note=fit_note, status=note)

def main():
    ap = argparse.ArgumentParser(description='Step 2: canonical LMM family.')
    ap.add_argument('--dataset', default='erpcore', choices=['erpcore', 'ds006018'])
    ap.add_argument('--features', default=None, help='features CSV (default: by variant)')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    tag = cfg.variant_tag()
    cfg.banner(f'step 2  canonical LMM  |  dataset={a.dataset}  variant={tag}')
    fp = a.features or os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    if not os.path.exists(fp):
        print(f'features file not found: {fp}')
        print('run step 1 with the same options first:')
        print(f'  python run\\01_features.py --dataset {a.dataset}' + (f' --reject {int(cfg.REJECT_UV)}' if cfg.REJECT_UV != 150 else ''))
        return 2
    df = pd.read_csv(fp)
    print(f'{len(df)} trials, {df.subject.nunique()} subjects from {os.path.basename(fp)}')
    if 'condition' in df.columns:
        print(f'conditions: {df.condition.value_counts().to_dict()}')
    models = [m for m in spec.CANONICAL]
    needed = sorted({m.predictor for m in models} | {c for m in models for c in m.covariates if c != 'condition'} | {'p300_pz'})
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f'\nfeatures file is missing columns these models need: {missing}')
        return 2
    df, unscaled = standardise(df, needed)
    if len(unscaled):
        print(f'\n{len(unscaled)} participant-by-column cases had MAD = 0 and were median-centred WITHOUT scaling:')
        for _, r in unscaled.head(12).iterrows():
            print(f'  {r.subject}  {r.column}')
        if len(unscaled) > 12:
            print(f'  ... and {len(unscaled) - 12} more')
        print('  these are recorded in the _unscaled.csv file; unscaled and scaled')
        print('  slopes are not commensurable and must not be pooled')
    print(f'\nfitting {len(models)} models')
    rows = []
    for m in models:
        r = fit_one(df, m)
        r.update(dataset=a.dataset, variant=tag, measure=m.measure, predictor=m.predictor, covariates='+'.join(m.covariates), paper_label=m.paper or '', alias_label=m.alias[0] if m.alias else '')
        rows.append(r)
        flag = '' if r['status'] == 'ok' else f"  [{r['status']}]"
        print(f"  {m.id:6s} {m.measure:26s} beta_lmm={r['beta_lmm']:+.6f}  R2m={r['r2_marginal']:.6f}  N={r['n_subjects']:3d}{flag}")
    out = pd.DataFrame(rows)[['dataset', 'variant', 'model_id', 'paper_label', 'alias_label', 'measure', 'predictor', 'covariates', 'beta_lmm', 'se', 'z', 'p', 'r2_marginal', 'n_trials', 'n_subjects', 'fit_method', 'fit_note', 'status']]
    p = os.path.join(cfg.RESULTS_DIR, f'canonical_lmm_{a.dataset}_{tag}.csv')
    out.to_csv(p, index=False)
    print(f'\n{len(out)} models -> {p}')
    if len(unscaled):
        pu = os.path.join(cfg.RESULTS_DIR, f'canonical_lmm_{a.dataset}_{tag}_unscaled.csv')
        unscaled.to_csv(pu, index=False)
        print(f'  -> {pu}')
    fb = out[out.fit_method == 'REML_nm']
    if len(fb):
        print(f'\n{len(fb)} of {len(out)} models fell back to Nelder-Mead:')
        for _, r in fb.iterrows():
            print(f"  {r.model_id}: {r.fit_note}")
    else:
        print(f'\nall {len(out)} models converged under REML_lbfgs')
    bad = out[out.status != 'ok']
    if len(bad):
        print(f'\n{len(bad)} models did not fit cleanly: {dict(zip(bad.model_id, bad.status))}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
