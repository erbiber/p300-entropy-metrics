import numpy as np
import pandas as pd
import os

SEED = 20260519
N_SIM = 4000
N_TRIALS = 40
K_SINGLE_DRAW = 1
K_MATCHED = 1000
ESTIMATORS = [('single_draw', K_SINGLE_DRAW), ('matched_contrast', K_MATCHED)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

OPERATING_POINTS = [('same_channel', 0.31), ('small_effect', 0.01)]
SAMPLE_SIZES = [27, 28, 84, 90]
RATES = [('1024Hz', 154), ('500Hz', 75)]

def robust_z(v):
    med = np.median(v, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(v - med), axis=-1, keepdims=True)
    mad = np.where(mad > 0, mad, 1.0)
    return (v - med) / mad

def ols_slope(x, y):
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    denom = (xm ** 2).sum(axis=-1)
    denom = np.where(denom > 0, denom, np.nan)
    return (xm * ym).sum(axis=-1) / denom

def draw_slopes(rng, n_sub, n_trials, beta, meas_noise_sd):
    x = rng.standard_normal((n_sub, n_trials))
    e = rng.standard_normal((n_sub, n_trials))
    y = beta * x + np.sqrt(max(1e-12, 1 - beta ** 2)) * e
    if meas_noise_sd > 0:
        x = x + meas_noise_sd * rng.standard_normal((n_sub, n_trials))
        y = y + meas_noise_sd * rng.standard_normal((n_sub, n_trials))
    return ols_slope(robust_z(x), robust_z(y))

def one_replicate(rng, n_sub, beta_real, beta_pseudo, meas_noise_sd, k_eff):
    real = draw_slopes(rng, n_sub, N_TRIALS, beta_real, meas_noise_sd)
    single = draw_slopes(rng, n_sub, N_TRIALS, beta_pseudo, meas_noise_sd)
    shrink = 1.0 / np.sqrt(k_eff)
    pseudo = beta_pseudo + (single - beta_pseudo) * shrink
    return np.nanmean(real), np.nanmean(pseudo)

def run():
    rng = np.random.default_rng(SEED)
    band_rows, power_rows = [], []

    for point_name, r2 in OPERATING_POINTS:
        beta = np.sqrt(r2)
        for n_sub in SAMPLE_SIZES:
            for rate_name, n_samp in RATES:
                meas_sd = np.sqrt(1.0 / n_samp)
                for est_name, k_eff in ESTIMATORS:

                    aur, dbeta = [], []
                    for _ in range(N_SIM):
                        br, bp = one_replicate(rng, n_sub, beta, beta, meas_sd, k_eff)
                        if abs(br) > 1e-9:
                            aur.append(abs(bp / br))
                        dbeta.append(br - bp)
                    aur = np.array(aur); dbeta = np.array(dbeta)
                    band_rows.append(dict(
                        estimator=est_name, k_placements=k_eff,
                        mechanism='pure_background', operating_point=point_name,
                        target_r2=r2, beta_true=beta, n_subjects=n_sub, rate=rate_name,
                        aur_lo=np.percentile(aur, 2.5), aur_hi=np.percentile(aur, 97.5),
                        aur_median=np.median(aur), dbeta_mean=dbeta.mean(),
                        dbeta_lo=np.percentile(dbeta, 2.5),
                        dbeta_hi=np.percentile(dbeta, 97.5), n_sim=N_SIM))

                    aur_sl = []
                    for _ in range(N_SIM):
                        br, bp = one_replicate(rng, n_sub, beta, 0.5 * beta, meas_sd, k_eff)
                        if abs(br) > 1e-9:
                            aur_sl.append(abs(bp / br))
                    aur_sl = np.array(aur_sl)
                    band_rows.append(dict(
                        estimator=est_name, k_placements=k_eff,
                        mechanism='stimulus_locked', operating_point=point_name,
                        target_r2=r2, beta_true=beta, n_subjects=n_sub, rate=rate_name,
                        aur_lo=np.percentile(aur_sl, 2.5), aur_hi=np.percentile(aur_sl, 97.5),
                        aur_median=np.median(aur_sl), dbeta_mean=np.nan,
                        dbeta_lo=np.nan, dbeta_hi=np.nan, n_sim=N_SIM))

                    aur_dil = []
                    for _ in range(N_SIM):
                        br, bp = one_replicate(rng, n_sub, 0.5 * beta, beta, meas_sd, k_eff)
                        if abs(br) > 1e-9:
                            aur_dil.append(abs(bp / br))
                    aur_dil = np.array(aur_dil)
                    band_rows.append(dict(
                        estimator=est_name, k_placements=k_eff,
                        mechanism='dilution', operating_point=point_name,
                        target_r2=r2, beta_true=beta, n_subjects=n_sub, rate=rate_name,
                        aur_lo=np.percentile(aur_dil, 2.5), aur_hi=np.percentile(aur_dil, 97.5),
                        aur_median=np.median(aur_dil), dbeta_mean=np.nan,
                        dbeta_lo=np.nan, dbeta_hi=np.nan, n_sim=N_SIM))

                    if est_name != 'matched_contrast':
                        continue
                    for increment in [0.02, 0.05, 0.10, 0.15, 0.20]:
                        detected = 0
                        for _ in range(1000):
                            br, bp = one_replicate(rng, n_sub, beta + increment, beta,
                                                   meas_sd, k_eff)
                            real = draw_slopes(rng, n_sub, N_TRIALS, beta + increment, meas_sd)
                            se = np.nanstd(real, ddof=1) / np.sqrt(n_sub)
                            if (br - bp) - 1.96 * se > 0:
                                detected += 1
                        power_rows.append(dict(
                            estimator=est_name, operating_point=point_name, target_r2=r2,
                            n_subjects=n_sub, rate=rate_name, true_increment=increment,
                            power=detected / 1000.0, n_sim=1000))

    os.makedirs(OUT, exist_ok=True)
    bands = pd.DataFrame(band_rows)
    power = pd.DataFrame(power_rows)
    bands.to_csv(os.path.join(OUT, 'calibration_bands.csv'), index=False)
    power.to_csv(os.path.join(OUT, 'calibration_power.csv'), index=False)

    print('=== AUR null bands, pure-background mechanism (central 95%) ===')
    pb = bands[bands.mechanism == 'pure_background']
    for _, r in pb.iterrows():
        print(f"  {r['estimator']:17s} {r['operating_point']:13s} R2={r['target_r2']:.2f}  "
              f"N={r['n_subjects']:3d}  {r['rate']:7s}  "
              f"[{r['aur_lo']:.2f}, {r['aur_hi']:.2f}]  median {r['aur_median']:.2f}")
    return bands, power

if __name__ == '__main__':
    run()
