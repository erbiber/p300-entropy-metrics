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
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

from config import RESULTS_DIR, FIG_DIR, LOG_DIR, FILES, banner

warnings.filterwarnings('ignore', category=Warning)
mne_available = False
try:
    import mne
    mne.set_log_level('ERROR')
    mne_available = True
except ImportError:
    pass

                                                                              
def robust_z(s):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return (s - med) / mad if mad > 0 else (s - med)

def add_z(df, cols):
    for c in cols:
        if c in df.columns:
            df[c + '_z'] = df.groupby('subject')[c].transform(robust_z)
    return df

def try_fit(df, formula, random_effects='1'):
    re_formula = None if random_effects in ('1', '~1') else random_effects
    for kwargs in [
        dict(method='lbfgs', reml=False),
        dict(method='nm', maxiter=2000, reml=False),
        dict(method='powell', maxiter=2000, reml=False),
    ]:
        try:
            md = smf.mixedlm(formula, df, groups=df['subject'],
                             re_formula=re_formula)
            fit = md.fit(**kwargs)
            if np.isfinite(float(fit.llf)):
                return fit
        except Exception:
            continue
    return None

def aic_bic(fit, n_obs):
    try:
        ll = float(fit.llf)
        k_fixed = len(fit.fe_params)
        try:
            cov_re = np.atleast_2d(np.asarray(fit.cov_re))
            k_re = cov_re.shape[0] * (cov_re.shape[0] + 1) // 2
        except Exception:
            k_re = 1
        k = k_fixed + k_re + 1
        return ll, 2 * k - 2 * ll, k * np.log(n_obs) - 2 * ll
    except Exception:
        return np.nan, np.nan, np.nan

def diagnostic_panel(fit, ax_qq, ax_rvf, title):
                                             
    try:
        resid = np.asarray(fit.resid)
        fitted = np.asarray(fit.fittedvalues)
    except Exception:
        X = np.asarray(fit.model.exog)
        fitted = X @ np.asarray(fit.fe_params)
        resid = np.asarray(fit.model.endog) - fitted

    rs = (resid - np.mean(resid)) / np.std(resid, ddof=1)
    stats.probplot(rs, dist='norm', plot=ax_qq)
    ax_qq.set_title(f'{title}: Q-Q', fontsize=9)
    for ln in ax_qq.get_lines():
        if ln.get_color() == 'r':
            ln.set_color('#A13544'); ln.set_linewidth(1.2)

    ax_rvf.scatter(fitted, resid, s=3, alpha=0.2, color='#20808D')
    ax_rvf.axhline(0, color='k', lw=0.7)
    try:
        ord_ = np.argsort(fitted)
        fs, rs2 = fitted[ord_], resid[ord_]
        edges = np.linspace(fs.min(), fs.max(), 31)
        bc, bm = [], []
        for i in range(30):
            m = (fs >= edges[i]) & (fs < edges[i+1])
            if m.sum() > 5:
                bc.append(0.5*(edges[i]+edges[i+1]))
                bm.append(rs2[m].mean())
        if bc:
            ax_rvf.plot(bc, bm, color='#A13544', lw=1.2, label='binned mean')
            ax_rvf.legend(fontsize=7, loc='upper right')
    except Exception:
        pass
    ax_rvf.set_title(f'{title}: residuals vs fitted', fontsize=9)

    try:
        sw_w, sw_p = stats.shapiro(rs)
    except Exception:
        sw_w, sw_p = np.nan, np.nan
    return dict(skew=float(stats.skew(rs)),
                excess_kurt=float(stats.kurtosis(rs)),
                shapiro_W=sw_w, shapiro_p=sw_p)

                                                                              
def load_amplitude_features():
                                                                             
    path = os.path.join(RESULTS_DIR, FILES['trial_features'])
    df = pd.read_csv(path)
    cols = ['mean_early_fz', 'sd_early_fz', 'rms_early_fz',
                           , 'sd_early_pz', 'rms_early_pz',
                          , 'rms_base_pz', 'p300_amp']
    return add_z(df, cols)

def load_entropy_features():
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
       
    het = os.path.join(LOG_DIR, 'heterogeneity_primary_trials.csv')
    if not os.path.exists(het):
        print(f"  [warn] heterogeneity trial log not found: {het} (run script 11 first)")
        return None
    df = pd.read_csv(het)
                                                                                   
    if 'dataset' in df.columns:
                                                                         
        prim = df[df['dataset'].astype(str).str.contains('erpcore|primary|erp_core',
                                                          case=False, na=False)]
        if len(prim) > 0:
            df = prim
                                                                       
                                                                              
    df = df.rename(columns={'p300': 'p300_amp'})
                             
    for c in ['pe', 'se', 'lz', 'hjorth_mob', 'hjorth_cplx', 'p300_amp']:
        if c in df.columns:
            df[c + '_z'] = df.groupby('subject')[c].transform(robust_z)
    return df

def load_shape_features():
                                                                    
    path = os.path.join(LOG_DIR, 'extended_endpoint_pseudotrial_results.csv')
    if not os.path.exists(path):
        print(f"  [warn] shape log not found: {path} (run script 08 first)")
        return None
                                                         
                                                                    
                                       
    return None

                                                                              
def main():
    banner("12_model_diagnostics.py — all-family model diagnostics")

    os.makedirs(LOG_DIR, exist_ok=True)

                                                                      
                                                       
                                                                      
    print("Loading amplitude features...")
    try:
        df_amp = load_amplitude_features()
        print(f"  {len(df_amp)} trials, {df_amp['subject'].nunique()} subjects")
        amp_ok = True
    except Exception as e:
        print(f"  [warn] could not load amplitude features: {e}")
        df_amp = None; amp_ok = False

    amp_headlines = [
        ('M1',  'rms_early_fz_z + condition',
                                 ,   'cross-channel energy'),
        ('M4a', 'mean_early_fz_z + sd_early_fz_z + condition',
                                  ,  'cross-channel mean (competitive)'),
        ('M9a', 'mean_early_pz_z + sd_early_pz_z + condition',
                                  ,  'same-channel mean (largest effect)'),
        ('M12', 'mean_early_pz_z + mean_base_pz_z + condition',
                                  ,  'same-channel + baseline covariate'),
    ]

                                                                      
                                     
                                                                      
    print("Loading entropy features...")
    df_ent = load_entropy_features()
    ent_ok = df_ent is not None
    if ent_ok:
        print(f"  {len(df_ent)} trials, {df_ent['subject'].nunique()} subjects")

                                                                           
                                                                 
    ent_headlines = [
        ('M13', 'pe_z', 'pe_z',
                                      ),
        ('M15', 'lz_z', 'lz_z',
                                        ),
    ]

                                                                      
                       
                                                                      
                                                                       
                                                                       
                                                                       
                                                                       
                         
    print("  [info] Hjorth/shape trial features not stored separately;")
    print("         residual behaviour documented from amplitude-family diagnostics.")

                                                                      
                                                
                                                                      
    all_models = []
    if amp_ok:
        for lbl, rhs, focal, desc in amp_headlines:
            all_models.append((lbl, rhs, focal, desc, df_amp))
    if ent_ok:
        for lbl, rhs, focal, desc in ent_headlines:
            all_models.append((lbl, rhs, focal, desc, df_ent))

    if not all_models:
        print("\n[error] No feature data available. Run scripts 01–11 first.")
        return

    n_models = len(all_models)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(12, 3.2 * n_models),
                             squeeze=False)

    rows = []                     
    diag_rows = []                      

    for i, (lbl, rhs, focal, desc, df) in enumerate(all_models):
        n_obs = len(df)
        print(f"\n--- {lbl} ({desc}): p300_amp_z ~ {rhs} ---")

        null_rhs = 'condition' if 'condition' in df.columns else '1'
        f_null = try_fit(df, f'p300_amp_z ~ {null_rhs}', '1')
        f_ri   = try_fit(df, f'p300_amp_z ~ {rhs}', '1')
        f_ris  = try_fit(df, f'p300_amp_z ~ {rhs}', f'~{focal}')

        for tag, fit in [('null', f_null), ('RI', f_ri), ('RIS', f_ris)]:
            if fit is None:
                rows.append(dict(model=lbl, family=desc.split(' ->')[0],
                                 structure=tag, ll=np.nan,
                                 aic=np.nan, bic=np.nan))
                print(f"  {tag:5s} FAILED")
                continue
            ll, aic, bic = aic_bic(fit, n_obs)
            rows.append(dict(model=lbl, family=desc.split(' ->')[0],
                             structure=tag, ll=ll, aic=aic, bic=bic))
            print(f"  {tag:5s}  ll={ll:10.2f}  AIC={aic:10.2f}  BIC={bic:10.2f}")

                                   
        chosen, chosen_tag = None, None
        for fit, t in [(f_ri, 'RI'), (f_ris, 'RIS'), (f_null, 'null')]:
            if fit is not None:
                chosen, chosen_tag = fit, t
                break
        if chosen is not None:
            d = diagnostic_panel(chosen, axes[i, 0], axes[i, 1],
                                 title=f'{lbl} ({chosen_tag})')
            d.update(dict(model=lbl, structure_used=chosen_tag, description=desc))
            diag_rows.append(d)

                                                                      
                  
                                                                      
    out1 = os.path.join(RESULTS_DIR, FILES['diagnostics'])
    pd.DataFrame(rows).to_csv(out1, index=False)
    out2 = os.path.join(LOG_DIR, 'lmm_residual_diagnostics.csv')
    pd.DataFrame(diag_rows).to_csv(out2, index=False)
    print(f"\nAIC/BIC table       -> {out1}")
    print(f"Residual diagnostics -> {out2}")

    fig.suptitle(
                                                                             ,
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(FIG_DIR, 'figS1_diagnostics.png')
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Diagnostic figure    -> {out_png}\n")

                                                                      
                            
                                                                      
    banner("AIC preference summary")
    df_rows = pd.DataFrame(rows)
    all_ri_preferred = True
    for lbl in df_rows['model'].unique():
        sub = df_rows[df_rows['model'] == lbl].dropna(subset=['aic'])
        ri_row  = sub[sub['structure'] == 'RI']
        ris_row = sub[sub['structure'] == 'RIS']
        if not ri_row.empty and not ris_row.empty:
            d_aic = float(ris_row['aic'].iloc[0]) - float(ri_row['aic'].iloc[0])
            pref = 'RI' if d_aic > 0 else 'RIS'
            if pref != 'RI':
                all_ri_preferred = False
            print(f"  {lbl:5s}  ΔAIC(RIS−RI) = {d_aic:+7.2f}  → {pref}-preferred")

    print()
    if all_ri_preferred:
        print("  ✓ RI is AIC-preferred for all models.")
        print("    The canonical RI-only specification (script 02) is justified.")
    else:
        print("  ✗ RIS preferred for at least one model — review specification.")

if __name__ == '__main__':
    main()
