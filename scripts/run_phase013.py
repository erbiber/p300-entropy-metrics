import argparse
import os
import numpy as np
import pandas as pd
import phase013_engine as E
import phase013_complexity_patch
import phase013_nan_safe_zscore

CONFIGS = {'config1': ('config1', 1.0, 100e-6), 'config2': ('config2', 0.5, 100e-6),
           'config3': ('config3', 1.0, 150e-6), 'config4': ('config4', 0.5, 150e-6)}

def collect(iterator):
    subs = []
    real_by = {m: [] for m in E.MODELS}
    pseudo_by = {m: [] for m in E.MODELS}
    inter = {m: dict(feat=[], p300=[], subject=[]) for m in E.MODELS}
    for sid, real_slope, pseudo_slopes, real_feats in iterator:
        subs.append(sid)
        for m in E.MODELS:
            real_by[m].append(real_slope[m])
            pseudo_by[m].append(np.asarray(pseudo_slopes[m], float))
            fx, fy = E.MODELS[m]
            inter[m]['feat'].extend(list(real_feats[fx]))
            inter[m]['p300'].extend(list(real_feats[fy]))
            inter[m]['subject'].extend([sid] * len(real_feats[fx]))
    return subs, real_by, pseudo_by, inter

def run_config(cfg, K, datasets, subset, cache_dir, min_pseudo, targets_only=False, clean_pseudo=False,
               resample_hz=None):
    cname = cfg[0]
    makers = {}
    if 'erpcore' in datasets:
        import phase013_erpcore as EC
        makers['erp_core'] = lambda: EC.iter_subjects(cfg, K=K, subset_n=subset, cache_dir=cache_dir,
                                                      targets_only=targets_only, clean_pseudo=clean_pseudo,
                                                      resample_hz=resample_hz)
    if 'ds006018' in datasets:
        import phase013_ds006018 as DS
        makers['ds006018'] = lambda: DS.iter_subjects(cfg, K=K, subset_n=subset, cache_dir=cache_dir,
                                                      clean_pseudo=clean_pseudo, resample_hz=resample_hz)

    dbeta_rows = []
    inter_store = {}
    persubject_rows = []
    r2_rows = []
    for tag, make_iter in makers.items():
        print(
            f"\n===== {tag} | {cname} | K={K} | min_pseudo={min_pseudo} =====")
        subs, real_by, pseudo_by, inter = collect(make_iter())
        if not subs:
            print(f"  no usable subjects for {tag}")
            continue
        inter_store[tag] = inter
        for m in E.MODELS:
            res = E.contrast_ragged(
                real_by[m], pseudo_by[m], min_draws=min_pseudo)
            theta, lo, hi = E.bca_ci(res['d'], B=4000, seed=12345)
            dbeta_rows.append(dict(dataset=tag, config=cname, model=m,
                                   n_subjects_processed=len(subs), n_used=res['n_used'], n_dropped=res['n_dropped'],
                                   draws_min=res['draws_min'], draws_median=res['draws_median'],
                                   beta_real=res['beta_real'], beta_pseudo_mean=res['beta_pseudo_mean'],
                                   beta_pseudo_sd=res['beta_pseudo_sd'], dbeta=theta, bca_lo=lo, bca_hi=hi,
                                   surrogate_p=res['surrogate_p']))

            r2m, r2beta, r2n = E.marginal_r2(
                inter[m]['feat'], inter[m]['p300'], inter[m]['subject'])
            r2_rows.append(dict(dataset=tag, config=cname, model=m, marginal_r2=r2m,
                                lmm_beta=r2beta, n_subjects=r2n, n_trials=len(inter[m]['feat'])))

            for i, sid in enumerate(subs):
                pj = np.asarray(pseudo_by[m][i], float)
                pmean = float(np.mean(pj)) if pj.size else float('nan')
                rj = float(real_by[m][i])
                persubject_rows.append(dict(dataset=tag, config=cname, model=m, subject=sid,
                                            real_slope=rj, pseudo_mean=pmean, d=rj - pmean, n_pseudo_draws=int(pj.size),
                                            used=bool(pj.size >= min_pseudo)))
            print(f"  {m:16s} n_used={res['n_used']:2d}(drop {res['n_dropped']}) "
                  f"beta_real={res['beta_real']:+.4f}  dbeta={theta:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"surrogate_p={res['surrogate_p']:.4f}  marginal_R2={r2m:.4f}")
    df = pd.DataFrame(dbeta_rows)
    df.attrs['persubject'] = pd.DataFrame(persubject_rows)
    df.attrs['marginal_r2'] = pd.DataFrame(r2_rows)

    inter_out = []
    if len(inter_store) >= 2:
        tags = list(inter_store.keys())
        for m in E.MODELS:
            feat = []
            p3 = []
            subj = []
            dset = []
            for tag in tags:
                r = inter_store[tag][m]
                feat += r['feat']
                p3 += r['p300']
                subj += [f"{tag}:{s}" for s in r['subject']]
                dset += [tag] * len(r['feat'])
            ib, ip = E.dataset_feature_interaction(feat, p3, subj, dset)
            inter_out.append(dict(config=cname, model=m, datasets="+".join(tags),
                                  interaction_beta=ib, interaction_p=ip, n_trials=len(feat)))
            print(f"  [interaction] {m:16s} beta={ib:+.4f} p={ip:.4g}")
    return df, pd.DataFrame(inter_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config4')
    ap.add_argument('--k', type=int, default=1000)
    ap.add_argument('--datasets', default='erpcore,ds006018')
    ap.add_argument('--subset', type=int, default=None,
                    help='limit #subjects (quick test)')
    ap.add_argument('--min-pseudo', type=int, default=200, dest='min_pseudo',
                    help='exclude subjects with fewer usable pseudo draws than this')
    ap.add_argument('--cache-dir', default='./phase013_cache',
                    dest='cache_dir')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--targets-only', action='store_true', dest='targets_only',
                    help='ERP CORE: keep only TARGET trials, condition-matched to ds006018')
    ap.add_argument('--clean-pseudo', action='store_true', dest='clean_pseudo',
                    help='drop pseudotrials whose early/P300 window overlaps a real evoked period')
    ap.add_argument('--resample', type=float, default=None, dest='resample_hz',
                    help='resample epochs to this rate after native-rate epoching (e.g. 500 to match ds006018)')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rs_suf = f"_rs{int(a.resample_hz)}" if a.resample_hz else ""

    import phase013_erpcore as _ec
    import phase013_engine as _en
    print("="*70)
    print(f"[phase013] driver file : {os.path.abspath(__file__)}")
    print(f"[phase013] erpcore file: {_ec.__file__}")
    print(f"[phase013] engine file : {_en.__file__}")
    print(f"[phase013] FLAGS -> clean_pseudo={a.clean_pseudo}  targets_only={a.targets_only}  "
          f"resample_hz={a.resample_hz}  config={a.config}  K={a.k}")
    print(f"[phase013] cache namespace suffix: '{a.config}"
          f"{'_targets' if a.targets_only else ''}{'_clean' if a.clean_pseudo else ''}{rs_suf}'  "
          f"(resample/clean flags force a REPROCESS; watch trials/draw)")
    print("="*70)
    cache_dir = None if a.no_cache else a.cache_dir
    datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    cfgs = list(CONFIGS.values()) if a.config == 'all' else [CONFIGS[a.config]]
    suffix = (f"{a.config}_K{a.k}" + ("_targets" if a.targets_only else "")
              + ("_clean" if a.clean_pseudo else "") + rs_suf)
    all_db = []
    all_it = []
    all_ps = []
    all_r2 = []
    for cfg in cfgs:
        db, it = run_config(cfg, a.k, datasets, a.subset, cache_dir, a.min_pseudo, a.targets_only,
                            a.clean_pseudo, a.resample_hz)
        if len(db):
            all_db.append(db)
            if 'persubject' in db.attrs and len(db.attrs['persubject']):
                all_ps.append(db.attrs['persubject'])
            if 'marginal_r2' in db.attrs and len(db.attrs['marginal_r2']):
                all_r2.append(db.attrs['marginal_r2'])
        if len(it):
            all_it.append(it)
    if all_db:
        db = pd.concat(all_db, ignore_index=True)
        p = os.path.join(a.out, f"phase013_dbeta_{suffix}.csv")
        db.to_csv(p, index=False)
        print(f"\nWrote {p}\n")
        print(db.to_string(index=False))
    if all_it:
        it = pd.concat(all_it, ignore_index=True)
        p = os.path.join(a.out, f"phase013_interaction_{suffix}.csv")
        it.to_csv(p, index=False)
        print(f"\nWrote {p}\n")
        print(it.to_string(index=False))
    if all_ps:
        ps = pd.concat(all_ps, ignore_index=True)
        p = os.path.join(a.out, f"phase013_persubject_{suffix}.csv")
        ps.to_csv(p, index=False)
        print(
            f"\nWrote {p} ({len(ps)} per-subject rows for participant-level figures)")
    if all_r2:
        r2 = pd.concat(all_r2, ignore_index=True)
        p = os.path.join(a.out, f"phase013_marginal_r2_{suffix}.csv")
        r2.to_csv(p, index=False)
        print(f"\nWrote {p} (canonical marginal R^2 for headline models)\n")
        print(r2.to_string(index=False))

if __name__ == '__main__':
    main()
