import numpy as np
from scipy import stats
try:
    import antropy as ant
except ImportError:
    ant = None

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

def feature_block(trace):
    mean_amp = float(np.mean(trace))
    sd_amp = float(np.std(trace - mean_amp, ddof=0))
    rms_amp = float(np.sqrt(np.mean(trace ** 2)))
    return (mean_amp, sd_amp, rms_amp)

def hjorth_block(trace):
    try:
        n = len(trace)
        var_x = float(np.var(trace, ddof=0))
        if var_x <= 0 or n <= 2:
            return (np.nan, np.nan)
        dx = np.diff(trace)
        var_dx = float(np.var(dx, ddof=0))
        mob_x = np.sqrt(var_dx / var_x) if var_x > 0 else np.nan
        ddx = np.diff(dx)
        var_ddx = float(np.var(ddx, ddof=0))
        mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else np.nan
        mob = float(mob_x) if np.isfinite(mob_x) else np.nan
        cplx = mob_dx / mob_x if np.isfinite(mob_x) and mob_x > 0 and np.isfinite(mob_dx) else np.nan
        cplx = float(cplx) if np.isfinite(cplx) else np.nan
        return (mob, cplx)
    except Exception:
        return (np.nan, np.nan)

def entropy_block(trace):
    if ant is None:
        raise ImportError('antropy is required: pip install antropy==0.2.2')
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
        bin_str = ''.join(('1' if v > 0 else '0' for v in sign_seq))
        if len(set(bin_str)) < 2:
            lz = 0.0
        else:
            lz = ant.lziv_complexity(bin_str, normalize=True)
    except Exception:
        lz = np.nan
    mob, cplx = hjorth_block(trace)
    return (pe, se, lz, mob, cplx)

def shape_block(trace):
    n = len(trace)
    out = {}
    out['median_amp'] = float(np.median(trace))
    out['mad_amp'] = float(np.median(np.abs(trace - np.median(trace))))
    out['p2p_amp'] = float(np.max(trace) - np.min(trace))
    try:
        out['skew_amp'] = float(stats.skew(trace, bias=False))
    except Exception:
        out['skew_amp'] = np.nan
    try:
        out['kurt_amp'] = float(stats.kurtosis(trace, fisher=True, bias=False))
    except Exception:
        out['kurt_amp'] = np.nan
    try:
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, trace, 1)
        out['slope_amp'] = float(slope)
    except Exception:
        out['slope_amp'] = np.nan
    return out

def trial_features(fz_trace, pz_trace, times, cfg):
    m_e = window_mask(times, cfg.EARLY_WINDOW)
    m_e200 = window_mask(times, cfg.EARLY_WINDOW_200)
    m_e250 = window_mask(times, cfg.EARLY_WINDOW_250)
    m_p = window_mask(times, cfg.P300_WINDOW)
    m_b = window_mask(times, cfg.BASELINE_CONTROL_WINDOW)
    r = {}
    r['mean_early_fz'], r['sd_early_fz'], r['rms_early_fz'] = feature_block(fz_trace[m_e])
    r['mean_early_fz_200'], _, r['rms_early_fz_200'] = feature_block(fz_trace[m_e200])
    r['mean_early_fz_250'], _, r['rms_early_fz_250'] = feature_block(fz_trace[m_e250])
    r['mean_base_fz'], _, r['rms_base_fz'] = feature_block(fz_trace[m_b])
    r['mean_early_pz'], r['sd_early_pz'], r['rms_early_pz'] = feature_block(pz_trace[m_e])
    r['mean_base_pz'], _, r['rms_base_pz'] = feature_block(pz_trace[m_b])
    r['p300_pz'] = float(np.mean(pz_trace[m_p]))
    r['ms_early_pz'] = r['rms_early_pz'] ** 2
    pe, se, lz, mob, cplx = entropy_block(fz_trace[m_e])
    r['pe_early_fz'] = pe
    r['se_early_fz'] = se
    r['lz_early_fz'] = lz
    r['hjmob_early_fz'] = mob
    r['hjcplx_early_fz'] = cplx
    for k, v in shape_block(fz_trace[m_e]).items():
        r[k + '_fz'] = v
    return r
COLUMNS = ['mean_early_fz', 'sd_early_fz', 'rms_early_fz', 'mean_early_fz_200', 'rms_early_fz_200', 'mean_early_fz_250', 'rms_early_fz_250', 'mean_base_fz', 'rms_base_fz', 'mean_early_pz', 'sd_early_pz', 'rms_early_pz', 'ms_early_pz', 'mean_base_pz', 'rms_base_pz', 'p300_pz', 'pe_early_fz', 'se_early_fz', 'lz_early_fz', 'hjmob_early_fz', 'hjcplx_early_fz', 'median_amp_fz', 'mad_amp_fz', 'p2p_amp_fz', 'skew_amp_fz', 'kurt_amp_fz', 'slope_amp_fz']
