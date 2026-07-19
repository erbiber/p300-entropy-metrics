import numpy as np
import phase013_engine as E

def feats(T, a, g, h, rng, real):
    sm = 1.0; c = rng.standard_normal(T)
    if real:
        u = g*rng.standard_normal(T)
        x = a*c+u+h*rng.standard_normal(T)+sm*rng.standard_normal(T)
        y = a*c+u+h*rng.standard_normal(T)+sm*rng.standard_normal(T)
    else:
        cp = rng.standard_normal(T)
        x = a*cp+sm*rng.standard_normal(T); y = a*cp+sm*rng.standard_normal(T)
    z = rng.standard_normal(T)
    return dict(rms_early_fz=x, p300_pz=y, mean_early_fz=z, mean_early_pz=z, rms_early_pz=z)

def run(name, a, g, h, N=27, T=40, K=1000, inject_bad=False, seed=1):
    rng = np.random.default_rng(seed)
    real_by = {m: [] for m in E.MODELS}; pseudo_by = {m: [] for m in E.MODELS}
    for s in range(N):
        rf = feats(T, a, g, h, rng, True)

        if inject_bad and s == 0: Ki = 0
        elif inject_bad and s == 1: Ki = 5
        elif inject_bad and s == 2: Ki = 0
        else: Ki = K
        pfs = [feats(T, a, 0, 0, rng, False) for _ in range(Ki)]
        rs, ps = E.subject_slopes_real_and_pseudo(rf, pfs)
        for m in E.MODELS:
            real_by[m].append(rs[m]); pseudo_by[m].append(ps[m])
    m = 'M1_RMS_Fz_0_150'
    res = E.contrast_ragged(real_by[m], pseudo_by[m], min_draws=200)
    theta, lo, hi = E.bca_ci(res['d'], B=3000, seed=7)
    verdict = ('incl0' if lo <= 0 <= hi else ('CI>0' if lo > 0 else 'CI<0'))
    ok = {'background': lo <= 0 <= hi, 'stim_locked': lo > 0, 'inflation': hi < 0}[name]
    print(f"{name:11s} | used={res['n_used']:2d} drop={res['n_dropped']} draws_min={res['draws_min']:4d} "
          f"| dbeta={theta:+.3f} [{lo:+.3f},{hi:+.3f}] {verdict:5s} | surr_p={res['surrogate_p']:.4f} | {'PASS' if ok else 'FAIL'}")

print("mechanism   | n used/drop | draws_min | Delta-beta [BCa 95% CI]        | surrogate p | verdict")
run('background',  1.0, 0.0, 0.0, seed=11)
run('stim_locked', 0.05, 1.0, 0.0, seed=12)
run('inflation',   1.6, 0.0, 1.2, seed=13)
print("--- with 2 zero-draw + 1 five-draw subjects injected (must be excluded; surrogate must NOT collapse) ---")
run('background',  1.0, 0.0, 0.0, inject_bad=True, seed=21)
run('stim_locked', 0.05, 1.0, 0.0, inject_bad=True, seed=22)
