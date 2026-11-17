import os
_HERE = os.path.dirname(os.path.abspath(__file__))

def _first_existing(*paths):
    for p in paths:
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    return os.path.abspath(paths[0]) if paths else _HERE
_DOF = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_BASE = os.environ.get('P300_BASE') or _first_existing(_DOF)
SCRIPTS_DIR = os.environ.get('P300_SCRIPTS', _HERE)
RESULTS_DIR = os.environ.get('P300_RESULTS', os.path.join(_BASE, 'results', 'P300'))
ERPCORE_DIR = os.environ.get('P300_ERPCORE', os.path.join(_BASE, 'data', 'erp_core_P3'))
DS006018_DIR = os.environ.get('P300_DS006018', os.path.join(_BASE, 'data_ds006018'))
FIGURES_DIR = os.environ.get('P300_FIGURES', os.path.join(RESULTS_DIR, 'figures'))

def _f(n, d):
    return float(os.environ.get(n, d))

def _i(n, d):
    return int(os.environ.get(n, d))

def _b(n, d):
    return os.environ.get(n, str(d)).lower() in ('1', 'true', 'yes')

def _ica_n(n, d):
    v = os.environ.get(n)
    if v is None:
        return d
    f = float(v)
    if f < 1.0:
        return f
    return int(round(f))

def _codes(n, d):
    v = os.environ.get(n)
    if not v:
        return list(d)
    return [int(x) for x in v.replace(';', ',').split(',') if x.strip()]
FILTER_LOW = _f('P300_FILTER_LOW', 0.1)
FILTER_HIGH = _f('P300_FILTER_HIGH', 30.0)
FILTER_DESIGN = os.environ.get('P300_FILTER_DESIGN', 'firwin')
TMIN_EPOCH = _f('P300_TMIN', -0.2)
TMAX_EPOCH = _f('P300_TMAX', 0.8)
BASELINE = (_f('P300_BASE_LO', -0.2), _f('P300_BASE_HI', 0.0))
APPLY_BASELINE = _b('P300_APPLY_BASELINE', True)
REJECT_UV = _f('P300_REJECT_UV', 150.0)
DO_ICA = _b('P300_DO_ICA', True)
ICA_N_COMPONENTS = _ica_n('P300_ICA_N', 20)
ICA_HIGHPASS_FOR_FIT = _f('P300_ICA_HPF', 1.0)
ICA_METHOD = os.environ.get('P300_ICA_METHOD', 'fastica')
ICA_RANDOM_STATE = _i('P300_ICA_SEED', 42)
EARLY_WINDOW = (0.0, _f('P300_EARLY_HI', 0.15))
EARLY_WINDOW_200 = (0.0, 0.2)
EARLY_WINDOW_250 = (0.0, 0.25)
P300_WINDOW = (_f('P300_LATE_LO', 0.3), _f('P300_LATE_HI', 0.6))
BASELINE_CONTROL_WINDOW = (-0.15, 0.0)
EARLY_CHANNEL = os.environ.get('P300_EARLY_CH', 'Fz')
P300_CHANNEL = os.environ.get('P300_LATE_CH', 'Pz')
EARLY_FALLBACK = ['FCz', 'F1', 'F2', 'Cz']
P300_FALLBACK = ['P1', 'P2', 'CPz', 'Cz']
TARGET_CODES = _codes('P300_TARGET_CODES', [11, 22, 33, 44, 55])
STANDARD_CODES = _codes('P300_STANDARD_CODES', [12, 13, 14, 15, 21, 23, 24, 25, 31, 32, 34, 35, 41, 42, 43, 45, 51, 52, 53, 54])
DS006018_TARGET_CODES = _codes('P300_DS006018_TARGET_CODES', [11, 22, 33, 44, 55])
DS006018_STANDARD_CODES = _codes('P300_DS006018_STANDARD_CODES', [12, 13, 14, 15, 21, 23, 24, 25, 31, 32, 34, 35, 41, 42, 43, 45, 51, 52, 53, 54])
DS006018_TASK = os.environ.get('P300_DS006018_TASK', 'visualoddball')
DS006018_SUBDIR = os.environ.get('P300_DS006018_SUBDIR', 'ds006018')
MIN_TRIALS_REQUIRED = _i('P300_MIN_TRIALS', 10)
MIN_TRIALS_PER_CONDITION = _i('P300_MIN_PER_COND', 10)
MIN_PSEUDO_DRAWS = _i('P300_MIN_PSEUDO', 200)
MIN_GAP_S = _f('P300_MIN_GAP_S', 0.5)
N_DRAWS = _i('P300_N_DRAWS', 1000)
PSEUDOTRIAL_SEED = _i('P300_PSEUDO_SEED', 12345)
REREF = os.environ.get('P300_REREF', 'distributed')
_rs = os.environ.get('P300_RESAMPLE_HZ', '')
RESAMPLE_HZ = float(_rs) if _rs else None
REJECT_THRESHOLD = REJECT_UV * 1e-06

def add_common_args(ap):
    ap.add_argument('--reject', type=float, metavar='UV', help='peak-to-peak rejection threshold in uV (default 150)')
    ap.add_argument('--resample', type=float, metavar='HZ', help='decimate to this rate (e.g. 500)')
    ap.add_argument('--reref', choices=['distributed', 'mastoid', 'average', 'cz'], help='recording reference (default distributed)')
    ap.add_argument('--no-baseline', action='store_true', help='omit baseline correction')
    ap.add_argument('--no-ica', action='store_true', help='skip ICA')
    ap.add_argument('--ica-n', type=float, metavar='N', help='ICA components: an integer count, or a fraction below 1 for a variance criterion (default 20)')
    ap.add_argument('--min-gap', type=float, metavar='S', help='minimum gap from a real event in seconds (default 0.5)')
    ap.add_argument('--min-pseudo', type=int, metavar='N', help='valid draws needed to include a participant (default 200)')
    ap.add_argument('--min-per-cond', type=int, metavar='N', help='minimum trials per condition (default 10); 0 disables')
    ap.add_argument('--task', metavar='NAME', help='ds006018 task to analyse (default visualoddball)')
    ap.add_argument('--ds006018-subdir', metavar='NAME', help='BIDS root inside the ds006018 folder (default ds006018); pass an empty string to search the folder itself')
    ap.add_argument('--results', metavar='DIR', help='override the results folder')
    ap.add_argument('--erpcore', metavar='DIR', help='override the ERP CORE folder')
    ap.add_argument('--ds006018', metavar='DIR', help='override the ds006018 folder')
    return ap

def apply_args(a):
    g = globals()
    if getattr(a, 'reject', None) is not None:
        g['REJECT_UV'] = float(a.reject)
        g['REJECT_THRESHOLD'] = g['REJECT_UV'] * 1e-06
    if getattr(a, 'resample', None) is not None:
        g['RESAMPLE_HZ'] = float(a.resample)
    if getattr(a, 'reref', None):
        g['REREF'] = a.reref
    if getattr(a, 'no_baseline', False):
        g['APPLY_BASELINE'] = False
    if getattr(a, 'no_ica', False):
        g['DO_ICA'] = False
    if getattr(a, 'ica_n', None) is not None:
        g['ICA_N_COMPONENTS'] = float(a.ica_n) if float(a.ica_n) < 1.0 else int(round(a.ica_n))
    if getattr(a, 'task', None):
        g['DS006018_TASK'] = a.task
    if getattr(a, 'ds006018_subdir', None) is not None:
        g['DS006018_SUBDIR'] = a.ds006018_subdir
    if getattr(a, 'min_per_cond', None) is not None:
        g['MIN_TRIALS_PER_CONDITION'] = int(a.min_per_cond)
    if getattr(a, 'min_pseudo', None) is not None:
        g['MIN_PSEUDO_DRAWS'] = int(a.min_pseudo)
    if getattr(a, 'min_gap', None) is not None:
        g['MIN_GAP_S'] = float(a.min_gap)
    for attr, key in (('results', 'RESULTS_DIR'), ('erpcore', 'ERPCORE_DIR'), ('ds006018', 'DS006018_DIR')):
        v = getattr(a, attr, None)
        if v:
            g[key] = os.path.abspath(v)
    g['FIGURES_DIR'] = os.path.join(g['RESULTS_DIR'], 'figures')

def ica_tag():
    if not DO_ICA:
        return 'noica'
    n = ICA_N_COMPONENTS
    if isinstance(n, float) and n < 1.0:
        return 'icv' + ('%g' % n).replace('.', '')
    return 'ic%d' % int(n)

def codes_for(dataset):
    if dataset == 'erpcore':
        return (list(TARGET_CODES), list(STANDARD_CODES))
    if dataset == 'ds006018':
        return (list(DS006018_TARGET_CODES), list(DS006018_STANDARD_CODES))
    raise ValueError(f'no event codes configured for dataset {dataset!r}')

def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)

def variant_tag():
    parts = [f'r{int(REJECT_UV)}', 'g' + f'{MIN_GAP_S:g}'.replace('.', '')]
    if RESAMPLE_HZ:
        parts.append(f'rs{int(RESAMPLE_HZ)}')
    if REREF != 'distributed':
        parts.append(REREF)
    if not APPLY_BASELINE:
        parts.append('nobase')
    parts.append(ica_tag())
    if MIN_TRIALS_PER_CONDITION != 10:
        parts.append(f'mc{MIN_TRIALS_PER_CONDITION}')
    if DS006018_TASK != 'visualoddball':
        parts.append('task' + DS006018_TASK)
    return '_'.join(parts)

def describe():
    return dict(scripts=SCRIPTS_DIR, results=RESULTS_DIR, erpcore=ERPCORE_DIR, ds006018=DS006018_DIR, reject_uv_ptp=REJECT_UV, min_gap_s=MIN_GAP_S, resample_hz=RESAMPLE_HZ, reref=REREF, baseline=APPLY_BASELINE, ica=DO_ICA, ica_n_components=ICA_N_COMPONENTS, early=EARLY_WINDOW, late=P300_WINDOW, min_trials=MIN_TRIALS_REQUIRED, min_per_condition=MIN_TRIALS_PER_CONDITION, min_pseudo_draws=MIN_PSEUDO_DRAWS, variant=variant_tag())

def banner(title):
    print('=' * 72)
    print(title)
    for k, v in describe().items():
        print(f'  {k:14s} {v}')
    print('=' * 72)
if __name__ == '__main__':
    for k, v in describe().items():
        print(f'{k:14s} {v}')
    for n, d in (('ERP CORE', ERPCORE_DIR), ('ds006018', DS006018_DIR), ('results', RESULTS_DIR)):
        print(f"{n:10s} {('exists' if os.path.isdir(d) else 'MISSING')}  {d}")
