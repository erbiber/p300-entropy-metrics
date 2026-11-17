import argparse
import csv
import datetime
import importlib
import json
import os
import subprocess
import sys
import re
import shutil
import zipfile
import time
import io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

PLAN = [
    dict(id='ds_feat',      step=1, dataset='ds006018', args=[],
         why='ds006018 features, primary arm'),
    dict(id='ec_feat',      step=1, dataset='erpcore',  args=[],
         why='ERP CORE features on the current ICA setting'),
    dict(id='ds_mech',      step=4, dataset='ds006018', args=[],
         why='mechanism tests, ds006018'),
    dict(id='ec_mech',      step=4, dataset='erpcore',  args=[],
         why='mechanism tests, ERP CORE: decides whether the window-position sign differs by dataset or by ICA setting'),
    dict(id='ec_feat_rs',   step=1, dataset='erpcore',  args=['--resample', '500'],
         why='ERP CORE at 500 Hz, to match ds006018 natively'),
    dict(id='ec_feat_r100', step=1, dataset='erpcore',  args=['--reject', '100'],
         why='ERP CORE robustness threshold'),
    dict(id='ds_canon',     step=2, dataset='ds006018', args=[],
         why='canonical LMM, ds006018'),
    dict(id='ec_canon',     step=2, dataset='erpcore',  args=[],
         why='canonical LMM, ERP CORE'),
    dict(id='ec_canon_r100', step=2, dataset='erpcore', args=['--reject', '100'],
         why='canonical LMM at the robustness threshold'),
    dict(id='ds_match',     step=3, dataset='ds006018', args=['--k', '300', '--jobs', '4'],
         why='matched contrast, ds006018 (long)'),
    dict(id='ec_match',     step=3, dataset='erpcore',  args=['--k', '300', '--jobs', '4'],
         why='matched contrast, ERP CORE (long)'),
    dict(id='ec_match_rs',  step=3, dataset='erpcore',  args=['--resample', '500', '--k', '300', '--jobs', '4'],
         why='matched contrast at 500 Hz (long)'),
    dict(id='ec_match_r100', step=3, dataset='erpcore', args=['--reject', '100', '--k', '300', '--jobs', '4'],
         why='matched contrast at the robustness threshold (long)'),
    dict(id='ds_feat_mast', step=1, dataset='ds006018', args=['--reref', 'mastoid'],
         why='ds006018 features, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ec_feat_mast', step=1, dataset='erpcore', args=['--reref', 'mastoid'],
         why='ERP CORE features, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ds_mech_mast', step=4, dataset='ds006018', args=['--reref', 'mastoid'],
         why='mechanism tests, ds006018, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ec_mech_mast', step=4, dataset='erpcore', args=['--reref', 'mastoid'],
         why='mechanism tests, ERP CORE, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ds_canon_mast', step=2, dataset='ds006018', args=['--reref', 'mastoid'],
         why='canonical LMM, ds006018, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ec_canon_mast', step=2, dataset='erpcore', args=['--reref', 'mastoid'],
         why='canonical LMM, ERP CORE, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM)'),
    dict(id='ds_match_mast', step=3, dataset='ds006018', args=['--reref', 'mastoid', '--k', '300', '--jobs', '4'],
         why='matched contrast, ds006018, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM) (long)'),
    dict(id='ec_match_mast', step=3, dataset='erpcore', args=['--reref', 'mastoid', '--k', '300', '--jobs', '4'],
         why='matched contrast, ERP CORE, linked mastoids (ERP CORE P9/P10, ds006018 LM/RM) (long)'),
    dict(id='ds_feat_aver', step=1, dataset='ds006018', args=['--reref', 'average'],
         why='ds006018 features, common average'),
    dict(id='ec_feat_aver', step=1, dataset='erpcore', args=['--reref', 'average'],
         why='ERP CORE features, common average'),
    dict(id='ds_mech_aver', step=4, dataset='ds006018', args=['--reref', 'average'],
         why='mechanism tests, ds006018, common average'),
    dict(id='ec_mech_aver', step=4, dataset='erpcore', args=['--reref', 'average'],
         why='mechanism tests, ERP CORE, common average'),
    dict(id='ds_canon_aver', step=2, dataset='ds006018', args=['--reref', 'average'],
         why='canonical LMM, ds006018, common average'),
    dict(id='ec_canon_aver', step=2, dataset='erpcore', args=['--reref', 'average'],
         why='canonical LMM, ERP CORE, common average'),
    dict(id='ds_match_aver', step=3, dataset='ds006018', args=['--reref', 'average', '--k', '300', '--jobs', '4'],
         why='matched contrast, ds006018, common average (long)'),
    dict(id='ec_match_aver', step=3, dataset='erpcore', args=['--reref', 'average', '--k', '300', '--jobs', '4'],
         why='matched contrast, ERP CORE, common average (long)'),
    dict(id='ds_par_mast', step=5, dataset='ds006018', args=['--reref', 'mastoid'],
         why='time-reversal parity features on the real trials, ds006018, linked mastoids'),
    dict(id='ec_par_mast', step=5, dataset='erpcore', args=['--reref', 'mastoid'],
         why='time-reversal parity features on the real trials, ERP CORE, linked mastoids'),
    dict(id='ds_par_aver', step=5, dataset='ds006018', args=['--reref', 'average'],
         why='time-reversal parity features on the real trials, ds006018, common average'),
    dict(id='ec_par_aver', step=5, dataset='erpcore', args=['--reref', 'average'],
         why='time-reversal parity features on the real trials, ERP CORE, common average'),
    dict(id='ds_parm_mast', step=3, dataset='ds006018', args=['--reref', 'mastoid', '--family', 'parity', '--k', '300', '--jobs', '4'],
         why='matched contrast of the parity family, ds006018, linked mastoids'),
    dict(id='ec_parm_mast', step=3, dataset='erpcore', args=['--reref', 'mastoid', '--family', 'parity', '--k', '300', '--jobs', '4'],
         why='matched contrast of the parity family, ERP CORE, linked mastoids'),
    dict(id='ds_parm_aver', step=3, dataset='ds006018', args=['--reref', 'average', '--family', 'parity', '--k', '300', '--jobs', '4'],
         why='matched contrast of the parity family, ds006018, common average'),
    dict(id='ec_parm_aver', step=3, dataset='erpcore', args=['--reref', 'average', '--family', 'parity', '--k', '300', '--jobs', '4'],
         why='matched contrast of the parity family, ERP CORE, common average'),
    dict(id='ec_feat_rs_mast', step=1, dataset='erpcore', args=['--reref', 'mastoid', '--resample', '500'],
         why='ERP CORE at 500 Hz on linked mastoids: sampling-rate check on a valid reference'),
    dict(id='ec_match_rs_mast', step=3, dataset='erpcore', args=['--reref', 'mastoid', '--resample', '500', '--k', '300', '--jobs', '4'],
         why='matched contrast at 500 Hz on linked mastoids (long)'),
    dict(id='ec_par_dist', step=5, dataset='erpcore', args=[],
         why='checks that the as-distributed continuous cache of ERP CORE reproduces the step-1 trials'),
    dict(id='ds_par_dist', step=5, dataset='ds006018', args=[],
         why='checks that the as-distributed continuous cache of ds006018 reproduces the step-1 trials'),
    dict(id='ec_cache_dist', step=7, dataset='erpcore', args=[],
         why='ERP CORE as distributed: features without baseline, participant ERPs, background symmetry'),
    dict(id='ec_cache_mast', step=7, dataset='erpcore', args=['--reref', 'mastoid'],
         why='ERP CORE linked mastoids: features without baseline, participant ERPs, background symmetry'),
    dict(id='ec_cache_aver', step=7, dataset='erpcore', args=['--reref', 'average'],
         why='ERP CORE common average: features without baseline, participant ERPs, background symmetry'),
    dict(id='ds_cache_dist', step=7, dataset='ds006018', args=[],
         why='ds006018 as distributed: features without baseline, participant ERPs, background symmetry'),
    dict(id='ds_cache_mast', step=7, dataset='ds006018', args=['--reref', 'mastoid'],
         why='ds006018 linked mastoids: features without baseline, participant ERPs, background symmetry'),
    dict(id='ds_cache_aver', step=7, dataset='ds006018', args=['--reref', 'average'],
         why='ds006018 common average: features without baseline, participant ERPs, background symmetry'),
    dict(id='ec_nb_dist', step=3, dataset='erpcore', args=['--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ERP CORE as distributed'),
    dict(id='ec_nb_mast', step=3, dataset='erpcore', args=['--reref', 'mastoid', '--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ERP CORE linked mastoids'),
    dict(id='ec_nb_aver', step=3, dataset='erpcore', args=['--reref', 'average', '--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ERP CORE common average'),
    dict(id='ds_nb_dist', step=3, dataset='ds006018', args=['--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ds006018 as distributed'),
    dict(id='ds_nb_mast', step=3, dataset='ds006018', args=['--reref', 'mastoid', '--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ds006018 linked mastoids'),
    dict(id='ds_nb_aver', step=3, dataset='ds006018', args=['--reref', 'average', '--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ds006018 common average'),
    dict(id='ec_feat_cz', step=1, dataset='erpcore', args=['--reref', 'cz'],
         why='ERP CORE re-referenced to Cz, the recording reference of ds006018 (long)'),
    dict(id='ec_match_cz', step=3, dataset='erpcore', args=['--reref', 'cz', '--k', '300', '--jobs', '4'],
         why='matched contrast, ERP CORE on Cz (long)'),
    dict(id='ec_cache_cz', step=7, dataset='erpcore', args=['--reref', 'cz'],
         why='ERP CORE on Cz: features without baseline, participant ERPs, background symmetry'),
    dict(id='ec_nb_cz', step=3, dataset='erpcore', args=['--reref', 'cz', '--no-baseline', '--k', '300', '--jobs', '4'],
         why='matched contrast without baseline correction, ERP CORE on Cz'),
    dict(id='ec_common', step=8, dataset='erpcore', args=['--jobs', '4'],
         why='ERP CORE: one ICA, references derived after cleaning, identical trials, event-free pseudotrials, condition-adjusted slopes'),
    dict(id='ds_common', step=8, dataset='ds006018', args=['--jobs', '4'],
         why='ds006018: one ICA, references derived after cleaning, identical trials, event-free pseudotrials, condition-adjusted slopes'),
    dict(id='common_analysis', step=9, dataset='all', args=[],
         why='common cleaning: transportability, absolute disagreement, location statistics, battery robustness, matched contrasts with Holm and equivalence, condition differences'),
    dict(id='ec_reffree', step=10, dataset='erpcore', args=['--jobs', '4'],
         why='ERP CORE reference-free (surface Laplacian) coupling on the step-8 cleaning, trials and pseudotrial placements; monitor-delay-corrected real trials (26 ms); real trials subsampled to the pseudotrial count; positive control (long)'),
    dict(id='ds_reffree', step=10, dataset='ds006018', args=['--jobs', '4'],
         why='ds006018 reference-free coupling, delay-corrected real trials (20 ms), subsampled real trials, positive control (long)'),
    dict(id='reffree_analysis', step=10, dataset='all', args=[],
         why='cross-dataset agreement under the reference-free derivation; within-dataset agreement of each reference with it for raw and background-corrected coupling; matched contrasts; positive control'),
    dict(id='robustness', step=11, dataset='all', args=[],
         why='equivalence and sampling-noise floor of real-trial slopes under every derivation, effective number of statistics, absolute-scale interaction, BCa intervals for profile agreement, Holm adjustment with both bootstrap p values, distribution of participant-level slopes with robust location estimates, neighbouring-trial adjustment on participants common to all references'),
    dict(id='simulation', step=12, dataset='all', args=[],
         why='calibration of the estimators by simulation with known true slopes: interval coverage, false-positive rate with Holm adjustment, equivalence at the margin, sampling-noise floor and profile agreement'),
    dict(id='followup', step=6, dataset='all', args=[], always=True,
         why='equivalence (TOST), cross-dataset agreement, Holm, parity decision, lags in seconds, rate check; reruns every time'),
    dict(id='summaries', step=13, dataset='all', args=[], always=True,
         why='values reported in the tables and figures collected into paper_values.json from the step 1 to 12 results; reruns every time'),
]

INTERRUPT_CODES = {-2, 130, 3221225786, -1073741510}

SCRIPT = {1: '01_features.py', 2: '02_canonical_lmm.py', 3: '03_matched.py', 4: '04_mechanism.py', 5: '05_parity_features.py', 6: '06_followup.py', 7: '07_from_cache.py', 8: '08_common.py', 9: '09_common_analysis.py', 10: '10_reference_free.py', 11: '11_robustness.py', 12: '12_simulation.py', 13: '13_summaries.py'}


def _retry(fn, what, tries=20, wait=0.5):
    for i in range(tries):
        try:
            return fn()
        except PermissionError as err:
            last = err
            time.sleep(wait)
    print(f'  WARNING: could not write {what} ({last}); pipeline continues, it is rewritten after the next step')
    return None

def _write_text(path, text):
    tmp = path + '.tmp'
    def go():
        with open(tmp, 'w', newline='') as fh:
            fh.write(text)
        os.replace(tmp, path)
        return True
    return _retry(go, os.path.basename(path))

def tag_for(args):
    importlib.reload(cfg)
    ap = argparse.ArgumentParser()
    cfg.add_common_args(ap)
    out = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ('--k', '--jobs', '--family'):
            skip = True
            continue
        if a == '--rate500':
            continue
        out.append(a)
    cfg.apply_args(ap.parse_args(out))
    return cfg.variant_tag()

def family_of(entry):
    args = entry['args']
    if '--family' in args and args[args.index('--family') + 1] == 'parity':
        return '_parity'
    return ''

def outputs_for(entry, results):
    tag = tag_for(entry['args'])
    ds = entry['dataset']
    if entry['step'] == 1:
        return [os.path.join(results, f'features_{ds}_{tag}.csv'),
                os.path.join(results, f'features_{ds}_{tag}_log.csv')], tag
    if entry['step'] == 2:
        return [os.path.join(results, f'canonical_lmm_{ds}_{tag}.csv')], tag
    if entry['step'] == 3:
        fam = family_of(entry)
        return [os.path.join(results, f'matched{fam}_{ds}_{tag}.csv'),
                os.path.join(results, f'matched{fam}_persubject_{ds}_{tag}.csv')], tag
    if entry['step'] == 5:
        return [os.path.join(results, f'features_parity_{ds}_{tag}.csv'),
                os.path.join(results, f'features_parity_{ds}_{tag}_log.csv')], tag
    if entry['step'] == 6:
        return [os.path.join(results, f) for f in ('followup_summary.txt', 'followup_equivalence.csv', 'followup_agreement.csv')], 'all arms'
    if entry['step'] == 8:
        return [os.path.join(results, f'common_persubject_{ds}_free.csv'),
                os.path.join(results, f'common_persubject_{ds}_onset.csv'),
                os.path.join(results, f'common_condition_{ds}.csv'),
                os.path.join(results, f'common_subjects_{ds}.csv')], 'common'
    if entry['step'] == 13:
        return [os.path.join(results, 'paper_values.json')], 'all arms'
    if entry['step'] == 12:
        return [os.path.join(results, f'simulation_{n}.csv') for n in ('calibration', 'summary', 'agreement')], 'common'
    if entry['step'] == 11:
        return [os.path.join(results, f'robustness_{n}.csv') for n in ('equivalence', 'summary', 'absolute_interaction', 'bca_agreement', 'tests', 'distribution', 'location_agreement', 'neighbour_adjustment', 'neighbour_share')], 'common'
    if entry['step'] == 10:
        if ds == 'all':
            return [os.path.join(results, f'reffree_{n}.csv') for n in ('agreement', 'dbeta', 'poscontrol_summary')], 'common'
        return [os.path.join(results, f'reffree_{n}_{ds}.csv') for n in ('persubject', 'subjects', 'validation', 'poscontrol')], 'common'
    if entry['step'] == 9:
        return [os.path.join(results, f'common_{n}_{m}.csv') for m in ('free', 'onset') for n in ('transport', 'absdis', 'location', 'robust', 'loo', 'dbeta', 'profiles', 'threshold')] + [os.path.join(results, 'common_condition_summary.csv')], 'common'
    if entry['step'] == 7:
        importlib.reload(cfg)
        ap = argparse.ArgumentParser()
        cfg.add_common_args(ap)
        cfg.apply_args(ap.parse_args(list(entry['args']) + ['--no-baseline']))
        nb = cfg.variant_tag()
        return [os.path.join(results, f'features_{ds}_{nb}.csv'),
                os.path.join(results, f'features_{ds}_{nb}_log.csv'),
                os.path.join(results, f'erp_{ds}_{tag}.csv'),
                os.path.join(results, f'symmetry_{ds}_{tag}.csv')], tag
    return [os.path.join(results, f'mechanism_{ds}_{tag}.csv')], tag

def read_csv(p):
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))

def num(v, d=float('nan')):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d

def validate_features(paths):
    msgs = []
    log = [p for p in paths if p.endswith('_log.csv')]
    if not log or not os.path.exists(log[0]):
        return ['FAIL no log file written']
    rows = read_csv(log[0])
    ok = [r for r in rows if r.get('status') == 'ok']
    msgs.append(f'{len(ok)} of {len(rows)} subjects ok')
    for r in rows:
        if r.get('status') != 'ok':
            msgs.append(f"  excluded {r['subject']}: {r['status']}")
    if not ok:
        return msgs + ['FAIL no subject produced features']
    comp = sorted({r.get('ica_components', '') for r in ok})
    msgs.append(f'ica_components {comp}')
    if any(num(c, -1) == 0 for c in comp):
        msgs.append('WARN a subject fitted 0 ICA components, so ICA did nothing there')
    dup = sorted({r.get('n_duplicate_samples', '') for r in ok})
    msgs.append(f'n_duplicate_samples {dup}')
    if any(num(d, 0) > 0 for d in dup):
        msgs.append('WARN duplicate event sample times present')
    eog = sorted({r.get('n_eog', '') for r in ok})
    msgs.append(f'n_eog {eog}')
    if any(num(e, 0) == 0 for e in eog):
        msgs.append('WARN a subject has no EOG channel, so no ocular component can be found')
    exc = [num(r.get('ica_excluded'), 0) for r in ok]
    msgs.append(f'ica_excluded min {min(exc):.0f} max {max(exc):.0f} '
                f'zero in {sum(1 for x in exc if x == 0)} of {len(exc)}')
    shares, tot, tt = [], 0, 0
    for r in ok:
        t, s = num(r.get('n_target'), 0), num(r.get('n_standard'), 0)
        if t + s > 0:
            shares.append(100 * t / (t + s))
        tot += num(r.get('n_trials'), 0)
        tt += t
    if shares:
        msgs.append(f'target share min {min(shares):.1f}% max {max(shares):.1f}% '
                    f'mean {sum(shares) / len(shares):.1f}% (design 20.0%)')
        far = [r['subject'] for r, sh in zip(ok, shares) if abs(sh - 20) > 5]
        if far:
            msgs.append(f'WARN target share more than 5pp off 20% in {far}')
    msgs.append(f'total {tot:.0f} trials, {tt:.0f} target, ratio 1:{(tot - tt) / tt:.2f}'
                if tt else f'total {tot:.0f} trials')
    feat = [p for p in paths if not p.endswith('_log.csv')]
    if feat and os.path.exists(feat[0]):
        rows2 = read_csv(feat[0])
        msgs.append(f'feature rows {len(rows2)}')
        if rows2:
            missing = {}
            nonfinite = {}
            for c in rows2[0]:
                if c in ('dataset', 'variant', 'subject', 'condition'):
                    continue
                for r in rows2:
                    v = r[c]
                    try:
                        x = float(v)
                    except ValueError:
                        missing[c] = missing.get(c, 0) + 1
                        continue
                    if x != x:
                        missing[c] = missing.get(c, 0) + 1
                    elif x in (float('inf'), float('-inf')):
                        nonfinite[c] = nonfinite.get(c, 0) + 1
            if not missing and not nonfinite:
                msgs.append('no missing or non-finite values in any feature column')
            if missing:
                msgs.append(f'WARN missing or non-numeric values: {missing}')
            if nonfinite:
                msgs.append(f'NOTE infinite values (treated as missing downstream): {nonfinite}')
            if abs(len(rows2) - tot) > 0.5:
                msgs.append(f'FAIL feature rows {len(rows2)} != log trial total {tot:.0f}')
    return msgs

def validate_canonical(paths):
    p = paths[0]
    if not os.path.exists(p):
        return ['FAIL no output written']
    rows = read_csv(p)
    msgs = [f'{len(rows)} models']
    bad = [r['model_id'] for r in rows if r.get('status') != 'ok']
    if bad:
        msgs.append(f'WARN models not ok: {bad}')
    fm = {}
    for r in rows:
        fm[r.get('fit_method', '')] = fm.get(r.get('fit_method', ''), 0) + 1
    msgs.append(f'fit_method {fm}')
    if fm.get('REML_nm'):
        msgs.append(f"WARN {fm['REML_nm']} models fell back to Nelder-Mead")
    nt = sorted({r.get('n_trials', '') for r in rows})
    ns = sorted({r.get('n_subjects', '') for r in rows})
    msgs.append(f'n_trials {nt}  n_subjects {ns}')
    sig = [r['model_id'] for r in rows if num(r.get('p'), 1) < 0.05]
    msgs.append(f'{len(sig)} of {len(rows)} models p<0.05')
    return msgs

def validate_matched(paths):
    p = paths[0]
    if not os.path.exists(p):
        return ['FAIL no output written']
    rows = read_csv(p)
    msgs = [f'{len(rows)} models']
    bad = [r['model_id'] for r in rows if r.get('status') != 'ok']
    if bad:
        msgs.append(f'WARN models not ok: {bad}')
    nu = sorted({r.get('n_used', '') for r in rows})
    nt = sorted({r.get('n_total', '') for r in rows})
    msgs.append(f'n_used {nu}  n_total {nt}')
    dmin = [num(r.get('draws_min'), 0) for r in rows]
    msgs.append(f'draws_min min {min(dmin):.0f} max {max(dmin):.0f}')
    excl = []
    for r in rows:
        lo, hi = num(r.get('bca_lo')), num(r.get('bca_hi'))
        if lo == lo and hi == hi and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0)):
            excl.append(f"{r['model_id']}({num(r.get('dbeta')):+.4f})")
    msgs.append(f'{len(excl)} of {len(rows)} models exclude zero: {excl}' if excl
                else f'0 of {len(rows)} models exclude zero')
    uns = sum(int(num(r.get('n_unscaled'), 0)) for r in rows)
    if uns:
        msgs.append(f'note {uns} unscaled predictor instances across models')
    return msgs

def validate_mechanism(paths):
    p = paths[0]
    if not os.path.exists(p):
        return ['FAIL no output written']
    rows = read_csv(p)
    msgs = []
    for r in rows:
        if r['test'].startswith(('1_', '2_summary', '3_', '0_')):
            msgs.append(f"{r['test']:18} {r['quantity']:28} {num(r['value']):+.4f}  {r['note']}")
    return msgs

RESULT_PREFIXES = ('features_', 'canonical_lmm_', 'matched_', 'matched_persubject_',
                   'placement_meta_', 'mechanism_')
CURRENT = re.compile(r'_(ic\d+|icv\d+|noica)(_|\.csv$)')

def archive_superseded(results, apply):
    dest = os.path.join(results, 'superseded')
    moves = []
    for f in sorted(os.listdir(results)):
        full = os.path.join(results, f)
        if not os.path.isfile(full) or not f.endswith('.csv'):
            continue
        if not f.startswith(RESULT_PREFIXES):
            continue
        named = '_erpcore_' in f or '_ds006018_' in f
        if named and CURRENT.search(f):
            continue
        moves.append(f)
    if apply and moves:
        os.makedirs(dest, exist_ok=True)
        for f in moves:
            shutil.move(os.path.join(results, f), os.path.join(dest, f))
    return moves, dest

def gate_selftests():
    from lib import fastfeat
    import placement
    report, failed = [], False
    for sf in (1024.0, 500.0):
        try:
            bad = fastfeat.selftest(cfg, sfreq=sf)
        except Exception as exc:
            report.append(f'fastfeat vs features at {sf:g} Hz: RAISED {type(exc).__name__}: {exc}')
            failed = True
            continue
        if bad:
            failed = True
            keys = sorted({b[0] for b in bad})
            report.append(f'fastfeat vs features at {sf:g} Hz: {len(bad)} MISMATCHES in {keys}')
        else:
            report.append(f'fastfeat vs features at {sf:g} Hz: identical')
    try:
        from lib import parity
        pf = parity.selftest()
        if pf:
            failed = True
            report += [f'parity self-test FAILED: {x}' for x in pf]
        else:
            report.append('parity self-test: odd features odd, even partners even, residual removes constant waveforms')
        cls = fastfeat.parity_classification()
        for sf in (1024.0, 500.0):
            row = [f'{n}={c}' for f_, n, c in cls if f_ == sf]
            report.append(f'time-reversal class at {sf:g} Hz: ' + '; '.join(row))
            for f_, n, c in cls:
                if f_ == sf and n.startswith('parity_') and c not in ('odd', 'even'):
                    failed = True
    except Exception as exc:
        report.append(f'parity checks: RAISED {type(exc).__name__}: {exc}')
        failed = True
    try:
        r = bool(placement.selftest_hoist())
        report.append(f'placement hoist self-test: {"ok" if r else "FAILED"}')
        if not r:
            failed = True
    except Exception as exc:
        report.append(f'placement hoist self-test: RAISED {type(exc).__name__}: {exc}')
        failed = True
    return report, failed

def validate_parity_features(paths):
    log = [p for p in paths if p.endswith('_log.csv')]
    if not log or not os.path.exists(log[0]):
        return ['FAIL no log file written']
    rows = read_csv(log[0])
    ok = [r for r in rows if r.get('status') == 'ok']
    msgs = [f'{len(ok)} of {len(rows)} subjects aligned with step 1']
    for r in rows:
        if r.get('status') != 'ok':
            msgs.append(f"  excluded {r['subject']}: {r['status']}")
    if ok:
        dev = max(num(r.get('p300_max_abs_diff'), 0) for r in ok)
        msgs.append(f'max |P300 rebuilt - step 1| {dev:.2e} V')
        src = {}
        for r in ok:
            src[r.get('source', '')] = src.get(r.get('source', ''), 0) + 1
        msgs.append(f'source {src}')
    if not ok:
        msgs.append('FAIL no subject aligned')
    return msgs

def validate_followup(paths):
    p = paths[0]
    if not os.path.exists(p):
        return ['FAIL no summary written']
    with open(p, encoding='utf-8') as fh:
        lines = [x.rstrip() for x in fh]
    keep = [x for x in lines if x.startswith(('BCa recomputation', '  VERDICT', '  equivalent at')) or 'sign agree' in x]
    return keep or lines[:10]

def validate_from_cache(paths):
    log = [p for p in paths if p.endswith('_log.csv')]
    if not log or not os.path.exists(log[0]):
        return ['FAIL no log file written']
    rows = read_csv(log[0])
    ok = [r for r in rows if r.get('status') == 'ok']
    msgs = [f'{len(ok)} of {len(rows)} subjects: step-1 features reproduced from the cache']
    for r in rows:
        if r.get('status') != 'ok':
            msgs.append(f"  excluded {r['subject']}: {r['status']}")
    if ok:
        msgs.append(f"max relative deviation of amplitude and shape features {max(num(r.get('max_rel_dev'), 0) for r in ok):.1e}")
        for c in ('pe_early_fz', 'se_early_fz', 'lz_early_fz'):
            k = f'entropy_trials_differing_{c}'
            tot = sum(int(num(r.get(k), 0)) for r in ok)
            msgs.append(f'{c}: {tot} trials differ from step 1 by more than 1e-6 (rounding at thresholds)')
    sym = [p for p in paths if os.path.basename(p).startswith('symmetry_')]
    if sym and os.path.exists(sym[0]):
        s = read_csv(sym[0])
        for c in ('skew_fz', 'skew_pz', 'bsi_fz', 'bsi_pz'):
            v = [num(r.get(c)) for r in s]
            v = [x for x in v if x == x]
            if v:
                msgs.append(f'{c}: mean {sum(v) / len(v):+.3f} over {len(v)} subjects')
    if not ok:
        msgs.append('FAIL no subject processed')
    return msgs

def validate_common(paths):
    sp = [p for p in paths if os.path.basename(p).startswith('common_subjects_')]
    if not sp or not os.path.exists(sp[0]):
        return ['FAIL no subject table written']
    rows = read_csv(sp[0])
    ok = [r for r in rows if r.get('status') == 'ok']
    msgs = [f'{len(ok)} of {len(rows)} subjects processed']
    for r in rows:
        if r.get('status') != 'ok':
            msgs.append(f"  excluded {r['subject']}: {r.get('status')}")
    if ok:
        v = sorted(num(r.get('pseudo_per_draw_median_free')) for r in ok)
        msgs.append(f'event-free pseudotrials per draw: median {v[len(v) // 2]:.0f}, min {v[0]:.0f}, max {v[-1]:.0f}')
        e = sorted(num(r.get('candidate_seconds_free')) for r in ok)
        msgs.append(f'eligible event-free seconds per recording: median {e[len(e) // 2]:.0f}, min {e[0]:.0f}')
        x = [num(r.get('n_ica_excluded')) for r in ok]
        msgs.append(f'ICA components removed: mean {sum(x) / len(x):.2f}, min {min(x):.0f}, max {max(x):.0f}')
        msgs.append(f"real trials: total {sum(int(num(r.get('n_real'))) for r in ok)}")
    for p in paths:
        if os.path.basename(p).startswith('common_persubject_') and os.path.exists(p):
            rr = read_csv(p)
            used = {r['subject'] for r in rr if r.get('used') in ('True', 'true', '1')}
            msgs.append(f'{os.path.basename(p)}: {len(used)} subjects used under every reference')
    if not ok:
        msgs.append('FAIL no subject processed')
    return msgs

def validate_common_analysis(paths):
    msgs = []
    miss = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if miss:
        return ['FAIL missing ' + ', '.join(miss)]
    for p in paths:
        rows = read_csv(p)
        bad = sum(1 for r in rows for k, v in r.items() if v in ('nan', 'inf', '-inf') and k not in ('p_le0',))
        msgs.append(f'{os.path.basename(p)}: {len(rows)} rows' + (f', {bad} non-finite values' if bad else ''))
    for m in ('free', 'onset'):
        t = [r for r in read_csv(os.path.join(os.path.dirname(paths[0]), f'common_transport_{m}.csv')) if r.get('quantity') == 'real_slope' and r.get('contrast') in ('distributed', 'mastoid', 'average')]
        msgs.append(f'{m}: real-trial r ' + ', '.join(f"{r['contrast']} {num(r['T']):+.2f}" for r in t))
    return msgs

def validate_reffree(paths):
    miss = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if miss:
        return ['FAIL missing ' + ', '.join(miss)]
    msgs = []
    for p in paths:
        b = os.path.basename(p)
        rows = read_csv(p)
        if b.startswith('reffree_validation_'):
            for r in rows:
                v = num(r.get('max_abs_diff'))
                msgs.append(f"reproduces step 8 {r.get('check')}: max |difference| {v:.2e} over {r.get('n_pairs')} pairs" + ('' if v == v and v < 1e-9 else '  FAIL'))
        elif b.startswith('reffree_subjects_'):
            msgs.append(f"{sum(1 for r in rows if r.get('status') == 'ok')} of {len(rows)} subjects ok")
        elif b.startswith('reffree_persubject_'):
            msgs.append(f"{len({r['subject'] for r in rows if r.get('used') in ('True', 'true', '1')})} subjects used")
        elif b == 'reffree_agreement.csv':
            for r in rows:
                if r.get('analysis') == 'cross_dataset' and r.get('quantity') == 'real_slope' and r.get('contrast') in ('distributed', 'mastoid', 'average', 'csd', 'csd18'):
                    msgs.append(f"real-trial r {r['contrast']} {num(r['value']):+.2f} [{num(r['lo']):+.2f}, {num(r['hi']):+.2f}]")
        else:
            msgs.append(f'{b}: {len(rows)} rows')
    return msgs

def validate_robustness(paths):
    miss = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if miss:
        return ['FAIL missing ' + ', '.join(miss)]
    msgs = []
    for r in read_csv([p for p in paths if p.endswith('robustness_summary.csv')][0]):
        msgs.append(f"{r['dataset']} {r['reference']}: mean |b| {num(r['mean_abs']):.3f}, floor {num(r['noise_floor']):.3f}, equivalent {r['n_equivalent']}/17, effective n {num(r['effective_n']):.1f}")
    return msgs

def validate_simulation(paths):
    miss = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if miss:
        return ['FAIL missing ' + ', '.join(miss)]
    msgs = []
    for r in read_csv([p for p in paths if p.endswith('simulation_summary.csv')][0]):
        msgs.append(f"{r['scenario']} {r['dataset']} {r['quantity']}: coverage {num(r['coverage95']):.3f}, Holm FWER {num(r['fwer_holm_inv']):.3f}, all-17 equivalent {num(r['p_all17_equivalent']):.3f}")
    return msgs

def validate_summaries(paths):
    if not os.path.exists(paths[0]):
        return ['FAIL missing paper_values.json']
    with open(paths[0], encoding='utf-8') as fh:
        v = json.load(fh)
    need = ['agreement', 'common_subjects', 'common_transport_free', 'drift', 'lags_trials', 'refstruct', 'specificity', 'symmetry_assoc', 'robustness_summary', 'sim_summary']
    miss = [k for k in need if k not in v]
    if miss:
        return ['FAIL missing entries ' + ', '.join(miss)]
    return [f'{len(v)} entries']

VALIDATOR = {1: validate_features, 2: validate_canonical, 3: validate_matched, 4: validate_mechanism, 5: validate_parity_features, 6: validate_followup, 7: validate_from_cache, 8: validate_common, 9: validate_common_analysis, 10: validate_reffree, 11: validate_robustness, 12: validate_simulation, 13: validate_summaries}

def main():
    ap = argparse.ArgumentParser(description='run the whole P300 pipeline in dependency order')
    ap.add_argument('--results', metavar='DIR', help='override the results folder')
    ap.add_argument('--only', metavar='IDS', help='comma-separated step ids to run')
    ap.add_argument('--skip', metavar='IDS', help='comma-separated step ids to skip')
    ap.add_argument('--steps', metavar='NUMS', help='comma-separated step numbers, e.g. 1,4')
    ap.add_argument('--force', action='store_true', help='rerun even when outputs exist')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    ap.add_argument('--stop-on-fail', action='store_true', help='halt at the first failing step')
    ap.add_argument('--bundle-features', action='store_true',
                    help='include the large per-trial feature files in the bundle')
    ap.add_argument('--no-bundle', action='store_true', help='skip writing the zip')
    ap.add_argument('--archive-superseded', action='store_true',
                    help='move result files whose names lack the dataset or the ICA setting into results/superseded/, then exit')
    ap.add_argument('--skip-selftest', action='store_true',
                    help='do not run the equivalence gate before the plan')
    a = ap.parse_args()

    importlib.reload(cfg)
    results = a.results or cfg.RESULTS_DIR
    results = os.path.abspath(results)
    os.makedirs(results, exist_ok=True)

    if a.archive_superseded:
        moves, dest = archive_superseded(results, apply=not a.dry_run)
        verb = 'would move' if a.dry_run else 'moved'
        print(f'{verb} {len(moves)} superseded result files -> {dest}')
        for f in moves:
            print(f'  {f}')
        kept = sorted(f for f in os.listdir(results)
                      if os.path.isfile(os.path.join(results, f)) and f.endswith('.csv')
                      and f.startswith(RESULT_PREFIXES) and f not in moves)
        print(f'\n{len(kept)} result files {"would remain" if a.dry_run else "remain"} in {results}:')
        for f in kept:
            print(f'  {f}')
        return 0

    plan = list(PLAN)
    if a.steps:
        want = {int(x) for x in a.steps.split(',')}
        plan = [e for e in plan if e['step'] in want]
    if a.only:
        want = {x.strip() for x in a.only.split(',')}
        plan = [e for e in plan if e['id'] in want]
    if a.skip:
        drop = {x.strip() for x in a.skip.split(',')}
        plan = [e for e in plan if e['id'] not in drop]

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('=' * 72)
    print(f'run_all   {len(plan)} steps   results {results}')
    print('=' * 72)
    for e in plan:
        outs, tag = outputs_for(e, results)
        have = all(os.path.exists(p) for p in outs) and not e.get('always')
        mark = 'SKIP (exists)' if have and not a.force else 'run'
        print(f'  {e["id"]:16} step {e["step"]}  {e["dataset"]:9} {tag:28} {mark}')
        print(f'                   {e["why"]}')
    print()
    if a.dry_run:
        return 0

    gate_lines = []
    if not a.skip_selftest:
        importlib.reload(cfg)
        print('step 0  equivalence gate: real-trial and pseudotrial feature code must agree')
        gate_lines, gate_failed = gate_selftests()
        for g in gate_lines:
            print(f'  {g}')
        if gate_failed:
            print('\nGATE FAILED: step 1 and step 3 would compute features differently.')
            print('Nothing has been run. Send this output.')
            return 3
        print()

    manifest, summary = [], []
    t_all = datetime.datetime.now()
    mp = os.path.join(results, f'pipeline_manifest_{stamp}.csv')
    sp = os.path.join(results, f'pipeline_summary_{stamp}.txt')
    bp = os.path.join(results, f'bundle_{stamp}.zip')

    def write_outputs(final):
        total = (datetime.datetime.now() - t_all).total_seconds()
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=['id', 'step', 'dataset', 'variant', 'status', 'seconds', 'outputs'])
        w.writeheader()
        w.writerows(manifest)
        _write_text(mp, buf.getvalue())
        lines = [f'pipeline run {stamp}', f'results {results}',
                 f'{"complete" if final else "IN PROGRESS"}: {len(manifest)} of {len(plan)} steps, {total:.0f}s so far']
        lines += [f'gate: {g}' for g in gate_lines]
        lines.append('=' * 72)
        for s_ in summary:
            lines += [s_, '-' * 72]
        _write_text(sp, '\n'.join(lines) + '\n')
        nb = 0
        if not a.no_bundle:
            files = set()
            for m in manifest:
                if m['status'] not in ('ok', 'skipped'):
                    continue
                entry = next(e for e in PLAN if e['id'] == m['id'])
                outs, tag = outputs_for(entry, results)
                for o in outs:
                    if os.path.basename(o).startswith('features_') and not o.endswith('_log.csv') \
                            and not a.bundle_features:
                        continue
                    files.add(o)
                if entry['step'] == 3:
                    files.add(os.path.join(results, f"placement_meta{family_of(entry)}_{entry['dataset']}_{tag}.csv"))
                if entry['step'] == 6:
                    for f_ in ('followup_lags_seconds.csv', 'followup_soa.csv', 'followup_timescale_bounds.csv', 'followup_rate_mastoid.csv'):
                        files.add(os.path.join(results, f_))
                if entry['step'] == 2:
                    files.add(os.path.join(results, f"canonical_lmm_{entry['dataset']}_{tag}_unscaled.csv"))
            files.add(mp)
            files.add(sp)
            files = sorted(f for f in files if os.path.isfile(f))
            tmp = bp + '.tmp'
            def go():
                with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
                    for f in files:
                        z.write(f, os.path.basename(f))
                os.replace(tmp, bp)
                return True
            nb = len(files) if _retry(go, os.path.basename(bp)) else 0
        return total, nb

    for e in plan:
        outs, tag = outputs_for(e, results)
        have = all(os.path.exists(p) for p in outs) and not e.get('always')
        head = f'[{e["id"]}] step {e["step"]} {e["dataset"]} {tag}'
        if have and not a.force:
            print(f'{head}  SKIPPED, outputs already present')
            manifest.append(dict(id=e['id'], step=e['step'], dataset=e['dataset'], variant=tag,
                                 status='skipped', seconds=0, outputs=';'.join(os.path.basename(p) for p in outs)))
            summary.append(f'{head}\n  SKIPPED, outputs already present')
            continue
        cmd = [sys.executable, os.path.join(HERE, 'run', SCRIPT[e['step']]),
               '--dataset', e['dataset'], '--results', results] + e['args']
        print('=' * 72)
        print(f'{head}  RUNNING')
        print('  ' + ' '.join(cmd[1:]))
        t0 = datetime.datetime.now()
        try:
            r = subprocess.run(cmd, cwd=HERE)
            interrupted = r.returncode in INTERRUPT_CODES
        except KeyboardInterrupt:
            interrupted = True
        secs = (datetime.datetime.now() - t0).total_seconds()
        if interrupted:
            print(f'\n  INTERRUPTED during {e["id"]} after {secs:.0f}s; stopping the whole run')
            manifest.append(dict(id=e['id'], step=e['step'], dataset=e['dataset'], variant=tag,
                                 status='interrupted', seconds=round(secs), outputs=''))
            summary.append(f'{head}\n  INTERRUPTED after {secs:.0f}s; later steps not run')
            write_outputs(False)
            print(f'  partial results kept: {bp}')
            print('  rerun python run_all.py to continue; finished steps will be skipped')
            return 130
        if r.returncode != 0:
            print(f'  FAILED exit {r.returncode} after {secs:.0f}s')
            manifest.append(dict(id=e['id'], step=e['step'], dataset=e['dataset'], variant=tag,
                                 status=f'failed_exit_{r.returncode}', seconds=round(secs),
                                 outputs=''))
            summary.append(f'{head}\n  FAILED exit {r.returncode} after {secs:.0f}s')
            write_outputs(False)
            if a.stop_on_fail:
                print('  halting')
                break
            continue
        checks = VALIDATOR[e['step']](outs)
        print(f'  done in {secs:.0f}s')
        for m in checks:
            print(f'  check: {m}')
        manifest.append(dict(id=e['id'], step=e['step'], dataset=e['dataset'], variant=tag,
                             status='ok', seconds=round(secs),
                             outputs=';'.join(os.path.basename(p) for p in outs)))
        summary.append(f'{head}\n  done in {secs:.0f}s\n' +
                       '\n'.join(f'  {m}' for m in checks))
        if write_outputs(False)[1]:
            print(f'  bundle updated: {os.path.basename(bp)}')

    total, nb = write_outputs(True)
    print('=' * 72)
    nfail = sum(1 for m in manifest if m['status'].startswith('failed'))
    nskip = sum(1 for m in manifest if m['status'] == 'skipped')
    print(f'{len(manifest)} steps: {len(manifest) - nfail - nskip} ran, {nskip} skipped, {nfail} failed'
          f'   total {total / 60:.1f} min')
    print(f'  -> {mp}')
    print(f'  -> {sp}')
    if not a.no_bundle:
        print(f'  -> {bp}  ({nb} files, {os.path.getsize(bp) / 1e6:.1f} MB)')
        print('     only this run\'s outputs; superseded files in the folder are not included')
        print('     send that one file; it is also kept up to date after every step')
    if nfail:
        print(f'\n{nfail} step(s) failed; see the summary')
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
