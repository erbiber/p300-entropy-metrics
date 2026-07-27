import numpy as np
import antropy as ant
import phase013_engine as E

def hjorth_block(trace):
    try:
        n = len(trace)
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or n <= 2:
            return np.nan, np.nan
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        mob_x = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan
        ddx = np.diff(dx)
        var_ddx = float(np.var(ddx, ddof=0))
        mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan
        mob = float(mob_x) if np.isfinite(mob_x) else np.nan
        cplx = (mob_dx / mob_x) if (np.isfinite(mob_x) and mob_x > 0
                                    and np.isfinite(mob_dx)) else np.nan
        cplx = float(cplx) if np.isfinite(cplx) else np.nan
        return mob, cplx
    except Exception:
        return np.nan, np.nan

def entropy_block(trace):

    trace = np.ascontiguousarray(trace, dtype=np.float64)
    try:
        pe = ant.perm_entropy(trace, order=3, delay=1, normalize=True)
    except Exception:
        pe = np.nan
    try:
        se = ant.sample_entropy(trace)
    except Exception:
        se = np.nan
    try:
        sign_seq = np.sign(trace - np.median(trace)).astype(int)
        sign_seq[sign_seq == 0] = 1
        bin_str = ''.join('1' if v > 0 else '0' for v in sign_seq)
        if len(set(bin_str)) < 2:
            lz = 0.0
        else:
            lz = ant.lziv_complexity(bin_str, normalize=True)
    except Exception:
        lz = np.nan
    hjorth_mob, hjorth_cplx = hjorth_block(trace)
    return pe, se, lz, hjorth_mob, hjorth_cplx

_orig_features_from_epochs = E.features_from_epochs

def features_from_epochs(data, times, fz_idx, pz_idx, early_win, p300_win):
    out = _orig_features_from_epochs(data, times, fz_idx, pz_idx, early_win, p300_win)
    me = E.window_mask(times, early_win)
    fz_early = data[:, fz_idx, :][:, me]
    n = fz_early.shape[0]
    pe = np.empty(n); se = np.empty(n); lz = np.empty(n)
    mob = np.empty(n); cpx = np.empty(n)
    for i in range(n):
        pe[i], se[i], lz[i], mob[i], cpx[i] = entropy_block(fz_early[i])
    out.update(pe_early_fz=pe, se_early_fz=se, lz_early_fz=lz,
               hjmob_early_fz=mob, hjcplx_early_fz=cpx)
    return out

E.features_from_epochs = features_from_epochs

for _modname in ('phase013_erpcore', 'phase013_ds006018'):
    import sys
    _m = sys.modules.get(_modname)
    if _m is not None and hasattr(_m, 'features_from_epochs'):
        _m.features_from_epochs = features_from_epochs

COMPLEXITY_MODELS = {
    'M13_PE_Fz_0_150':       ('pe_early_fz',     'p300_pz'),
    'M14_SE_Fz_0_150':       ('se_early_fz',     'p300_pz'),
    'M15_LZ_Fz_0_150':       ('lz_early_fz',     'p300_pz'),
    'M22_HJORTHMOB_Fz_0_150': ('hjmob_early_fz',  'p300_pz'),
    'M23_HJORTHCPLX_Fz_0_150': ('hjcplx_early_fz', 'p300_pz'),
}
E.MODELS.update(COMPLEXITY_MODELS)

print(f"[phase013_complexity_patch] MODELS now: {list(E.MODELS)}")
