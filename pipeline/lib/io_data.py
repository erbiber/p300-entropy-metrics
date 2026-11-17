import os
import glob
import mne
mne.set_log_level('WARNING')
EOG_CANDIDATES = ['HEOG', 'VEOG', 'HEOG_left', 'HEOG_right', 'VEOG_lower', 'heog', 'veog', 'EOG', 'LHEOG', 'RHEOG', 'LVEOG', 'UVEOG', 'HEL', 'HER', 'VEL', 'VER']
MISC_CANDIDATES = ['LM', 'RM', 'M1', 'M2', 'A1', 'A2']

def erpcore_subjects(cfg):
    pat = os.path.join(cfg.ERPCORE_DIR, 'sub-*', 'ses-P3', 'eeg', 'sub-*_ses-P3_task-P3_eeg.set')
    out = []
    for p in sorted(glob.glob(pat)):
        sub = os.path.basename(p).split('_')[0].replace('sub-', '')
        out.append(sub)
    return out

def load_erpcore(sub, cfg):
    p = os.path.join(cfg.ERPCORE_DIR, f'sub-{sub}', 'ses-P3', 'eeg', f'sub-{sub}_ses-P3_task-P3_eeg.set')
    if not os.path.exists(p):
        return (None, 'file_not_found')
    return (mne.io.read_raw_eeglab(p, preload=True, verbose=False), None)

def ds006018_dataset(cfg):
    from eegdash.dataset import DS006018
    return DS006018(cache_dir=cfg.DS006018_DIR)
DS006018_EXTS = ('.vhdr', '.set', '.fif', '.edf', '.bdf')

DS006018_SKIP_DIRS = ('derivatives', 'sourcedata', 'stimuli', 'code')

def _has_payload(path):
    if not path.endswith('.vhdr'):
        return True
    stem = path[:-5]
    for ext in ('.eeg', '.dat', '.seg'):
        if os.path.exists(stem + ext):
            return True
    return False

def ds006018_root(cfg):
    sub = getattr(cfg, 'DS006018_SUBDIR', '')
    if sub:
        p = os.path.join(cfg.DS006018_DIR, sub)
        if os.path.isdir(p):
            return p
    return cfg.DS006018_DIR

def _ds006018_files(cfg, task=None):
    want = task if task is not None else getattr(cfg, 'DS006018_TASK', '')
    root = ds006018_root(cfg)
    out = []
    for ext in DS006018_EXTS:
        out.extend(glob.glob(os.path.join(root, '**', '*' + ext), recursive=True))
    keep = []
    for p in out:
        rel = os.path.relpath(p, root).replace('\\', '/').split('/')
        if any(part in DS006018_SKIP_DIRS for part in rel):
            continue
        if not _has_payload(p):
            continue
        if want and f'task-{want}' not in os.path.basename(p):
            continue
        keep.append(p)
    return sorted(keep)

def _subject_of(path):
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part.startswith('sub-'):
            return part[4:].split('_')[0]
    return None

def ds006018_subjects(cfg, task=None):
    seen = []
    for p in _ds006018_files(cfg, task):
        sub = _subject_of(p)
        if sub and sub not in seen:
            seen.append(sub)
    return seen

def _read_any(path):
    if path.endswith('.vhdr'):
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    if path.endswith('.set'):
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    if path.endswith('.fif'):
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    if path.endswith('.bdf'):
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    return mne.io.read_raw_edf(path, preload=True, verbose=False)

def load_ds006018(sub, cfg, task=None):
    cands = [p for p in _ds006018_files(cfg, task) if _subject_of(p) == sub]
    if not cands:
        return (None, 'file_not_found')
    if len(cands) > 1:
        rels = [os.path.relpath(c, ds006018_root(cfg)).replace('\\', '/') for c in cands]
        return (None, 'ambiguous_%d_recordings: %s' % (len(cands), ' | '.join(rels)))
    try:
        return (_read_any(cands[0]), None)
    except Exception as exc:
        return (None, 'read_failed_%s' % type(exc).__name__)

def prepare_channel_types(raw):
    m = {}
    for ch in raw.ch_names:
        if ch in EOG_CANDIDATES:
            m[ch] = 'eog'
        elif ch in MISC_CANDIDATES:
            m[ch] = 'misc'
    if m:
        raw.set_channel_types(m, verbose=False)
    return (raw, m)

def mastoids_present(raw, dataset):
    from .preprocess import MASTOIDS
    return [c for c in MASTOIDS[dataset] if c in raw.ch_names]
