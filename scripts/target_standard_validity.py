import os, glob, argparse
import numpy as np, pandas as pd

TARGET_CODES = {11, 22, 33, 44, 55}
CORRECT_CODE, INCORRECT_CODE = 201, 202

def analyse_subject(events_path, max_rt=2.0):
    ev = pd.read_csv(events_path, sep='\t').sort_values('onset').reset_index(drop=True)
    stim = ev[ev.trial_type == 'stimulus'].copy()
    resp = ev[ev.trial_type == 'response'].copy()
    resp_onsets = resp['onset'].values; resp_vals = resp['value'].values
    rows = []
    for _, s in stim.iterrows():
        is_tgt = int(s['value']) in TARGET_CODES
        after = resp_onsets > s['onset']
        rt = np.nan; correct = np.nan
        if after.any():
            j = np.argmax(after)
            if resp_onsets[j] - s['onset'] <= max_rt:
                rt = resp_onsets[j] - s['onset']
                correct = int(resp_vals[j] == CORRECT_CODE)
        rows.append(dict(condition='target' if is_tgt else 'standard', rt=rt, correct=correct))
    d = pd.DataFrame(rows)
    def agg(sub):
        return dict(n=len(sub),
                    accuracy=np.nanmean(sub['correct']) if len(sub) else np.nan,
                    median_rt=np.nanmedian(sub['rt']) if len(sub) else np.nan)
    out = {'n_target': int((d.condition=='target').sum()),
           'n_standard': int((d.condition=='standard').sum())}
    for c in ('target', 'standard'):
        a = agg(d[d.condition == c])
        out[f'{c}_accuracy'] = a['accuracy']; out[f'{c}_median_rt'] = a['median_rt']
    out['overall_accuracy'] = np.nanmean(d['correct']) if len(d) else np.nan
    out['overall_median_rt'] = np.nanmedian(d['rt']) if len(d) else np.nan
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=r'C:\Users\erkan\Documents\dof_validation\data\erp_core_P3')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.data_root, 'sub-*', 'ses-P3', 'eeg',
                                          'sub-*_ses-P3_task-P3_events.tsv')))
    if not paths:

        paths = sorted(glob.glob(os.path.join(a.data_root, '**', '*_task-P3_events.tsv'), recursive=True))
    print(f"found {len(paths)} events files under {a.data_root}")
    rows = []
    for p in paths:
        sid = os.path.basename(p).split('_')[0]
        try:
            r = analyse_subject(p); r['subject'] = sid; rows.append(r)
            print(f"  {sid}: target={r['n_target']} standard={r['n_standard']} "
                  f"acc(t/s)={r['target_accuracy']:.2f}/{r['standard_accuracy']:.2f} "
                  f"RT(t/s)={1000*r['target_median_rt']:.0f}/{1000*r['standard_median_rt']:.0f}ms")
        except Exception as e:
            print(f"  {sid}: FAILED {e}")
    df = pd.DataFrame(rows)
    op = os.path.join(a.out, 'target_standard_validity.csv'); df.to_csv(op, index=False)
    print(f"\nWrote {op}")
    if len(df):
        print("\n=== AGGREGATE (mean +/- SD across subjects) ===")
        for col in ['n_target','n_standard','target_accuracy','standard_accuracy',
                    'target_median_rt','standard_median_rt','overall_accuracy']:
            v = df[col].values
            unit = ' ms' if 'rt' in col else ''
            scale = 1000 if 'rt' in col else 1
            print(f"  {col:22s}: {np.nanmean(v)*scale:.3f} +/- {np.nanstd(v)*scale:.3f}{unit}")

if __name__ == '__main__':
    main()
