import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import config as cfg
from lib import features as F, io_data as IO, preprocess as P

def process_subject(sub, cfg, dataset):
    if dataset == 'erpcore':
        raw, err = IO.load_erpcore(sub, cfg)
    elif dataset == 'ds006018':
        raw, err = IO.load_ds006018(sub, cfg)
    else:
        return (None, f'unknown_dataset_{dataset}', {})
    targets, standards = cfg.codes_for(dataset)
    if err:
        return (None, err, {})
    raw, _ = IO.prepare_channel_types(raw)
    eogs = P.eog_channels(raw)
    raw = P.bandpass(raw, cfg)
    raw, refnote = P.apply_reference(raw, dataset, cfg.REREF)
    raw, n_comp, n_exc = P.run_ica(raw, cfg)
    epochs, code_of, evinfo = P.make_epochs(raw, cfg, keep_codes=targets + standards)
    if len(epochs) < cfg.MIN_TRIALS_REQUIRED:
        return (None, f'insufficient_trials_{len(epochs)}', {})
    all_codes = sorted({c for c in code_of.values() if c is not None})
    mne_values = sorted({int(v) for v in code_of.keys()})
    epochs, seen = P.restrict_to_codes(epochs, cfg, dataset, code_of)
    if len(epochs) < cfg.MIN_TRIALS_REQUIRED:
        return (None, f'insufficient_valid_trials_{len(epochs)}_codes_seen={all_codes}', {})
    resampled = P.resample_if_asked(epochs, cfg)
    ch = epochs.ch_names
    fz_name, fz_i = P.get_channel_index(ch, cfg.EARLY_CHANNEL, cfg.EARLY_FALLBACK)
    pz_name, pz_i = P.get_channel_index(ch, cfg.P300_CHANNEL, cfg.P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return (None, f'missing_channels_fz={fz_name}_pz={pz_name}', {})
    data = epochs.get_data()
    times = epochs.times
    rows = []
    for i in range(data.shape[0]):
        raw_val = int(epochs.events[i, 2])
        code = code_of.get(raw_val)
        cond = 'Target' if code in targets else 'Standard'
        r = F.trial_features(data[i, fz_i, :], data[i, pz_i, :], times, cfg)
        r.update(subject=f'sub-{sub}', trial_idx=i, event_code=code, condition=cond, is_target=int(cond == 'Target'))
        rows.append(r)
    n_t = int(sum((r['is_target'] for r in rows)))
    n_s = int(len(rows) - n_t)
    mc = cfg.MIN_TRIALS_PER_CONDITION
    if mc > 0 and (n_t < mc or n_s < mc):
        return (None, f'below_per_condition_minimum_{mc}_target={n_t}_standard={n_s}', {})
    meta = dict(subject=f'sub-{sub}', n_trials=len(rows), sfreq=float(epochs.info['sfreq']), fz=fz_name, pz=pz_name, reference=refnote, resampled=int(resampled), ica_components=n_comp, ica_excluded=n_exc, eog_channels=';'.join(eogs), n_eog=len(eogs),
                n_events_all=evinfo['n_events_all'], n_events_kept=evinfo['n_events_kept'],
                n_duplicate_samples=evinfo['n_duplicate_samples'], event_codes_in_file=';'.join((str(c) for c in all_codes)), mne_event_values=';'.join((str(v) for v in mne_values)), codes_kept=';'.join((str(c) for c in seen)), n_target=n_t, n_standard=n_s, status='ok')
    return (pd.DataFrame(rows), None, meta)

def main():
    ap = argparse.ArgumentParser(description='Step 1: extract all trial features for one dataset.')
    ap.add_argument('--dataset', required=True, choices=['erpcore', 'ds006018'])
    ap.add_argument('--subjects', default=None, help='comma-separated subject ids; default is all found')
    ap.add_argument('--limit', type=int, default=None, help='first N subjects only')
    cfg.add_common_args(ap)
    a = ap.parse_args()
    cfg.apply_args(a)
    cfg.ensure_dirs()
    tag = cfg.variant_tag()
    cfg.banner(f'step 1  features  |  dataset={a.dataset}  variant={tag}')
    if a.subjects:
        raw_ids = [x.strip() for x in a.subjects.replace(';', ',').split(',') if x.strip()]
        subs = []
        for sid in raw_ids:
            if sid.startswith('sub-'):
                sid = sid[4:]
            subs.append(sid.zfill(3) if sid.isdigit() and len(sid) < 3 else sid)
        if subs != raw_ids:
            print(f'note: subject ids normalised to {subs}')
            print('      (an unquoted 009,036 becomes 9,36 in PowerShell - quote it)')
    elif a.dataset == 'erpcore':
        subs = IO.erpcore_subjects(cfg)
    else:
        subs = IO.ds006018_subjects(cfg)
    if not subs:
        if a.dataset == 'erpcore':
            print(f'no subjects found under {cfg.ERPCORE_DIR}')
            print('expected sub-XXX/ses-P3/eeg/sub-XXX_ses-P3_task-P3_eeg.set')
        else:
            print(f'no subjects found under {cfg.DS006018_DIR}')
            print(f'expected a recording whose name contains task-{cfg.DS006018_TASK}')
            print('a .vhdr is only accepted when its .eeg payload sits beside it')
        return 2
    if a.limit:
        subs = subs[:a.limit]
    if a.dataset == 'ds006018':
        print(f'task filter: task-{cfg.DS006018_TASK}')
    print(f'{len(subs)} subjects to process\n')
    frames, log = ([], [])
    failures = []
    reported = [False]
    for s in subs:
        try:
            df, err, meta = process_subject(s, cfg, a.dataset)
        except Exception as exc:
            err = f'raised_{type(exc).__name__}: {exc}'
            print(f'  sub-{s}: FAILED  {err}')
            log.append(dict(subject=f'sub-{s}', status=err, n_trials=0))
            failures.append((s, err))
            continue
        if err:
            print(f'  sub-{s}: SKIPPED  {err}')
            log.append(dict(subject=f'sub-{s}', status=err, n_trials=0))
            continue
        print(f"  sub-{s}: {len(df):4d} trials ({meta['n_target']} target / {meta['n_standard']} standard)  sfreq={meta['sfreq']:.0f}  ref={meta['reference']}")
        if not reported[0]:
            reported[0] = True
            parsed = [int(c) for c in meta['event_codes_in_file'].split(';') if c]
            assigned = [int(v) for v in meta['mne_event_values'].split(';') if v]
            tgt, std = cfg.codes_for(a.dataset)
            paradigm = set(tgt + std)
            kept = [int(c) for c in meta['codes_kept'].split(';') if c]
            print()
            print('  ' + '-' * 68)
            print('  codes parsed from the annotation descriptions:')
            print(f'    {parsed}')
            print('  integers mne.events_from_annotations assigned to them:')
            print(f'    {assigned}')
            print(f'  paradigm codes configured for {a.dataset}:')
            print(f'    targets   {sorted(tgt)}')
            print(f'    standards {sorted(std)}')
            print(f'  codes kept after restriction: {kept}')
            missing = sorted(paradigm - set(parsed))
            extra = sorted(set(parsed) - paradigm)
            if missing:
                print(f'  paradigm codes NOT present in this recording: {missing}')
            if extra:
                print(f'  codes present but outside the paradigm, dropped: {extra}')
            if set(assigned) != set(parsed):
                print('  note: the assigned integers differ from the parsed codes, so')
                print('        selection must be done on the description, as it is here')
            print('  ' + '-' * 68)
            print()
        frames.append(df)
        log.append(meta)
    if failures:
        print(f'\n{len(failures)} subject(s) raised and were skipped:')
        for sub, err in failures:
            print(f'  sub-{sub}: {err}')
    if not frames:
        print('\nno subject produced features; nothing written')
        return 1
    out = pd.concat(frames, ignore_index=True)
    out['dataset'] = a.dataset
    out['variant'] = tag
    lead = ['dataset', 'variant', 'subject', 'trial_idx', 'event_code', 'condition', 'is_target']
    out = out[lead + [c for c in F.COLUMNS if c in out.columns]]
    p = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}.csv')
    out.to_csv(p, index=False)
    pl = os.path.join(cfg.RESULTS_DIR, f'features_{a.dataset}_{tag}_log.csv')
    pd.DataFrame(log).to_csv(pl, index=False)
    kept = out.subject.nunique()
    nt = int(out.is_target.sum())
    print(f'\n{len(out)} trials from {kept} subjects ({nt} target / {len(out) - nt} standard, ratio 1:{(len(out) - nt) / nt:.2f})' if nt else '')
    print(f'  -> {p}')
    print(f'  -> {pl}')
    nan = {c: int(out[c].isna().sum()) for c in F.COLUMNS if c in out.columns and out[c].isna().any()}
    if nan:
        print(f'\ncolumns with missing values: {nan}')
    else:
        print('\nno missing values in any feature column')
    return 0
if __name__ == '__main__':
    sys.exit(main())
