import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
mne.set_log_level('ERROR')

from config import (RESULTS_DIR, FIG_DIR, LOG_DIR, FILES,
                    P300_WINDOW, EARLY_WINDOW, banner)

os.makedirs(FIG_DIR, exist_ok=True)
TARGET_COLOR   = '#A13544'
STANDARD_COLOR = '#20808D'
PSEUDO_COLOR   = '#F5A623'

def _build_topo_info(ch_names, montage_name='standard_1020'):
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()['ch_pos']
    pos_lc = {k.lower(): v for k, v in pos.items()}
    kept = [c for c in ch_names
            if c.lower() in pos_lc
            and not np.any(np.isnan(pos_lc[c.lower()]))]
    if not kept:
        return None, []
    info = mne.create_info(kept, sfreq=1000., ch_types='eeg')
    sub = mne.channels.make_dig_montage(
        ch_pos={c: np.asarray(pos_lc[c.lower()]) for c in kept},
        coord_frame='head')
    info.set_montage(sub)
    return info, kept

def _topo_vec(df, kind, kept, value='R2_marginal', measure=None):
    d = df[df['kind'] == kind]
    if measure is not None:
        d = d[d['measure'] == measure]
    d = d.set_index('electrode')
    return np.array([d.loc[c, value] if c in d.index else np.nan
                     for c in kept])

def _load_npz():
    path = os.path.join(RESULTS_DIR, FILES['figure1_data'])
    if not os.path.exists(path):
        print(f"  [skip] grand-average data not found: {path}")
        return None
    d = np.load(path, allow_pickle=True)
    return d['gas']

def figure_1_erp_and_topo():
    gas = _load_npz()
    if gas is None:
        return
    times = gas[0]['times']
    t_ms  = times * 1000

    erp_t_pz = np.nanmean([g['erp_target_pz']   for g in gas], axis=0) * 1e6
    erp_s_pz = np.nanmean([g['erp_standard_pz'] for g in gas], axis=0) * 1e6
    erp_t_fz = np.nanmean([g['erp_target_fz']   for g in gas], axis=0) * 1e6
    erp_s_fz = np.nanmean([g['erp_standard_fz'] for g in gas], axis=0) * 1e6

    topo_t = np.nanmean([g['topo_target_p300']   for g in gas], axis=0) * 1e6
    topo_s = np.nanmean([g['topo_standard_p300'] for g in gas], axis=0) * 1e6
    topo_d = topo_t - topo_s
    ch_names = list(gas[0]['ch_names'])
    scalp = [c for c in ch_names if 'EOG' not in c.upper()]
    info, kept = _build_topo_info(scalp)
    if info is None:
        print("  [warn] fig1: no channels match montage; topo panels skipped")

    n = len(gas)
    n_scalp = len(kept) if info is not None else len(scalp)

    fig = plt.figure(figsize=(14, 10))

    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.45,
                          wspace=0.30)

    ax_pz = fig.add_subplot(gs[0, 0])
    ax_fz = fig.add_subplot(gs[0, 1])

    for ax, t, s, title in [
        (ax_pz, erp_t_pz, erp_s_pz, 'P300 site (Pz)'),
        (ax_fz, erp_t_fz, erp_s_fz, 'Early site (Fz)'),
    ]:
        ax.plot(t_ms, t, color=TARGET_COLOR,   lw=1.5, label='Target')
        ax.plot(t_ms, s, color=STANDARD_COLOR, lw=1.5, label='Standard')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.axvspan(P300_WINDOW[0]*1000, P300_WINDOW[1]*1000,
                   alpha=0.18, color='#F5D76E', zorder=0, label='P300 window')
        ax.axvspan(EARLY_WINDOW[0]*1000, EARLY_WINDOW[1]*1000,
                   alpha=0.18, color='#9DB8D2', zorder=0, label='Early window')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='lower right', frameon=False, fontsize=8)

        ax.set_xticks(np.arange(-200, 801, 200))
        ax.set_xticklabels([f'{v/1000:.1f}' for v in np.arange(-200, 801, 200)])

    fig.text(0.5, 0.95, 'Figure 1A. Grand-average ERPs (target vs standard)',
             ha='center', va='bottom', fontsize=13)

    if info is not None:
        idx = [ch_names.index(c) for c in kept]
        t_vec = topo_t[idx];  s_vec = topo_s[idx];  d_vec = topo_d[idx]
        vmax_ts = max(np.nanmax(np.abs(t_vec)), np.nanmax(np.abs(s_vec)))
        vmax_d  = np.nanmax(np.abs(d_vec))
        p0, p1 = int(P300_WINDOW[0]*1000), int(P300_WINDOW[1]*1000)

        for col, data, vmax, title in [
            (0, t_vec,  vmax_ts, f'Target ({p0}–{p1} ms)'),
            (1, s_vec,  vmax_ts, 'Standard'),
            (2, d_vec,  vmax_d,  'Target − Standard'),
        ]:
            ax = fig.add_subplot(gs[1, col])
            im, _ = mne.viz.plot_topomap(
                data, info, axes=ax, show=False,
                cmap='RdBu_r', vlim=(-vmax, vmax),
                contours=5, outlines='head')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         orientation='horizontal')
            ax.set_title(title, fontsize=11)

        fig.text(0.5, 0.50,
                 f'Figure 1B. P300-window topographies '
                 f'(average reference, N = {n}, {n_scalp} scalp channels)',
                 ha='center', va='bottom', fontsize=13)

    out = os.path.join(FIG_DIR, 'fig1_erp_grand_average.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig1 -> {out}")

def figure_2_rectified_fz():
    gas = _load_npz()
    if gas is None:
        return
    times = gas[0]['times']
    t_ms  = times * 1000

    rect_t = np.nanmean([np.abs(g['erp_target_fz'])   for g in gas], axis=0) * 1e6
    rect_s = np.nanmean([np.abs(g['erp_standard_fz']) for g in gas], axis=0) * 1e6

    e0, e1 = int(EARLY_WINDOW[0]*1000), int(EARLY_WINDOW[1]*1000)
    p0, p1 = int(P300_WINDOW[0]*1000), int(P300_WINDOW[1]*1000)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t_ms, rect_t, color=TARGET_COLOR,   lw=1.5, label='Target')
    ax.plot(t_ms, rect_s, color=STANDARD_COLOR, lw=1.5, label='Standard')
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4)
    ax.axvspan(EARLY_WINDOW[0]*1000, EARLY_WINDOW[1]*1000,
               alpha=0.18, color='#9DB8D2',
               label=f'Early window ({e0}–{e1} ms)')
    ax.axvspan(P300_WINDOW[0]*1000, P300_WINDOW[1]*1000,
               alpha=0.18, color='#F5D76E', label='P300 window')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|ERP| at Fz (µV)')
    ax.set_xticks(np.arange(-200, 801, 200))
    ax.set_xticklabels([f'{v/1000:.1f}' for v in np.arange(-200, 801, 200)])
    ax.set_title('Time-course of rectified Fz signal (proxy for RMS)',
                 fontsize=13)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_rectified_fz.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig2 -> {out}")

def figure_3_pseudotrial():

    amp_path = os.path.join(RESULTS_DIR, FILES['pseudotrial_summary'])

    ent_path = os.path.join(LOG_DIR, 'entropy_pseudotrial_results.csv')

    rows = []
    if os.path.exists(amp_path):
        amp = pd.read_csv(amp_path)
        for _, r in amp.iterrows():
            lbl = r['model'].split('_')[0]
            rows.append(dict(label=lbl,
                             beta_real=r['beta_real'],
                             beta_pseudo=r['beta_pseudo'],
                             R2_real=r['R2_real'],
                             R2_pseudo=r['R2_pseudo']))
    else:
        print(f"  [warn] fig3: amplitude pseudotrial file not found: {amp_path}")

    if os.path.exists(ent_path):
        ent = pd.read_csv(ent_path)

        er = ent[ent['kind']=='real']
        ep = ent[(ent['kind']=='pseudo')&(ent['config']=='config4')]
        model_map = {'M_PE_Fz_0_150':'M13',
                     'M_SE_Fz_0_150':'M14',
                     'M_LZ_Fz_0_150':'M15'}
        for orig, lbl in model_map.items():
            rr = er[er['model']==orig]
            pp = ep[ep['model']==orig]
            if len(rr) and len(pp):
                rows.append(dict(
                    label=lbl,
                    beta_real=rr['beta'].iloc[0],
                    beta_pseudo=pp['beta'].iloc[0],
                    R2_real=rr['R2_marginal'].iloc[0],
                    R2_pseudo=pp['R2_marginal'].iloc[0]))
    else:
        print(f"  [warn] fig3: entropy log not found: {ent_path}")

    if not rows:
        print("  fig3 skipped: no data available")
        return

    df = pd.DataFrame(rows)
    n = len(df)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].bar(x - w/2, df['beta_real'],   w, label='Real',   color=STANDARD_COLOR)
    axes[0].bar(x + w/2, df['beta_pseudo'], w, label='Pseudo', color=TARGET_COLOR,
                alpha=0.85)
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df['label'])
    axes[0].set_ylabel(r'$\beta$ (standardised)')
    axes[0].set_title('A.  Coupling coefficient', loc='left', fontweight='bold')
    axes[0].legend(frameon=False)

    axes[1].bar(x - w/2, df['R2_real'],   w, label='Real',   color=STANDARD_COLOR)
    axes[1].bar(x + w/2, df['R2_pseudo'], w, label='Pseudo', color=TARGET_COLOR,
                alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df['label'])
    axes[1].set_ylabel(r'$R^2$ (marginal)')
    axes[1].set_title('B.  Variance explained', loc='left', fontweight='bold')
    axes[1].legend(frameon=False)

    fig.suptitle(
        'Figure 3.  Real vs pseudotrial comparison — '
        'amplitude (M1, M4a, M9a, M12) and entropy (M13–M15) families',
        fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_pseudotrial_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig3 -> {out}")

def _aur_vec(beta_r, beta_p, cap=3.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        aur = np.where(np.abs(beta_r) > 1e-9,
                       np.abs(beta_p) / np.abs(beta_r), np.nan)
    return np.clip(aur, 0, cap)

def _draw_topo_triplet(fig, axes, r2, beta, aur, row_label=None,
                       aur_cap=3.0):
    from matplotlib.colors import TwoSlopeNorm
    bmax = np.nanmax(np.abs(beta))
    aur_top = min(aur_cap, max(2.0, np.nanpercentile(aur, 95)))
    aur_norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=aur_top)

    im0, _ = mne.viz.plot_topomap(
        r2, _draw_topo_triplet.info, axes=axes[0], show=False,
        cmap='viridis', vlim=(0, np.nanmax(r2)),
        contours=4, outlines='head')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label('marginal $R^2$')
    axes[0].set_title(r'$R^2$', fontsize=11, fontweight='bold')

    im1, _ = mne.viz.plot_topomap(
        beta, _draw_topo_triplet.info, axes=axes[1], show=False,
        cmap='RdBu_r', vlim=(-bmax, bmax),
        contours=4, outlines='head')
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r'standardised $\beta$')
    axes[1].set_title(r'$\beta$', fontsize=11, fontweight='bold')

    im2, _ = mne.viz.plot_topomap(
        aur, _draw_topo_triplet.info, axes=axes[2], show=False,
        cmap='RdBu_r', cnorm=aur_norm,
        contours=4, outlines='head')
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label('AUR  (< 1 = stimulus-locked)')
    axes[2].set_title('AUR', fontsize=11, fontweight='bold')

    if row_label is not None:
        axes[0].annotate(
            row_label, xy=(-0.22, 0.5), xycoords='axes fraction',
            ha='right', va='center', fontsize=12,
            fontweight='bold', rotation=90)

def figure_4_cross_channel():
    path = os.path.join(RESULTS_DIR, FILES['interelectrode_cross'])
    if not os.path.exists(path):
        print(f"  fig4 skipped: {path} not found (run script 09 first)")
        return
    v1 = pd.read_csv(path)
    scalp = [c for c in v1['electrode'].unique() if 'EOG' not in c.upper()]
    info, kept = _build_topo_info(scalp)
    if info is None:
        print("  fig4 skipped: channel-montage mismatch"); return
    _draw_topo_triplet.info = info

    r2r    = _topo_vec(v1, 'real',   kept)
    beta_r = _topo_vec(v1, 'real',   kept, 'beta')
    beta_p = _topo_vec(v1, 'pseudo', kept, 'beta')
    aur    = _aur_vec(beta_r, beta_p)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    _draw_topo_triplet(fig, axes, r2r, beta_r, aur)
    fig.suptitle(
        'Figure 4.  Cross-channel coupling topography  '
        '(early activity at each electrode → P300 at Pz)\n'
        '$R^2$ and $\\beta$ are real-trial values; AUR = '
        '$|\\beta_{pseudo}|/|\\beta_{real}|$  '
        '(> 1 = autocorrelation; capped at 3; diverging at 1)',
        fontsize=11, y=1.06)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig4_cross_channel.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig4 -> {out}")

def figure_5_same_channel():
    path = os.path.join(RESULTS_DIR, FILES['interelectrode_same'])
    if not os.path.exists(path):
        print(f"  fig5 skipped: {path} not found (run script 09 first)")
        return
    v2 = pd.read_csv(path)
    scalp = [c for c in v2['electrode'].unique() if 'EOG' not in c.upper()]
    info, kept = _build_topo_info(scalp)
    if info is None:
        print("  fig5 skipped: channel-montage mismatch"); return
    _draw_topo_triplet.info = info

    r2r    = _topo_vec(v2, 'real',   kept)
    beta_r = _topo_vec(v2, 'real',   kept, 'beta')
    beta_p = _topo_vec(v2, 'pseudo', kept, 'beta')
    aur    = _aur_vec(beta_r, beta_p)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    _draw_topo_triplet(fig, axes, r2r, beta_r, aur)
    fig.suptitle(
        'Figure 5.  Same-channel coupling topography  '
        '(early → late at the same electrode)\n'
        '$R^2$ and $\\beta$ are real-trial values; AUR = '
        '$|\\beta_{pseudo}|/|\\beta_{real}|$  '
        '(≈ 1 = preserved under pseudotrials; capped at 3; diverging at 1)',
        fontsize=11, y=1.06)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig5_same_channel.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig5 -> {out}")

def figure_6_complexity_topo():
    path = os.path.join(RESULTS_DIR, FILES['interelectrode_shape'])
    if not os.path.exists(path):
        print(f"  fig6 skipped: {path} not found (run script 09 first)")
        return
    v3 = pd.read_csv(path)
    scalp = [c for c in v3['electrode'].unique() if 'EOG' not in c.upper()]
    info, kept = _build_topo_info(scalp)
    if info is None:
        print("  fig6 skipped: channel-montage mismatch"); return
    _draw_topo_triplet.info = info

    measures = [('perm_entropy_to_Pz', 'Permutation entropy'),
                ('hjorth_mob_to_Pz',   'Hjorth mobility')]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row, (m, label) in enumerate(measures):
        r2r    = _topo_vec(v3, 'real',   kept, 'R2_marginal', m)
        beta_r = _topo_vec(v3, 'real',   kept, 'beta', m)
        beta_p = _topo_vec(v3, 'pseudo', kept, 'beta', m)
        aur    = _aur_vec(beta_r, beta_p)
        _draw_topo_triplet(fig, axes[row], r2r, beta_r, aur, row_label=label)

        for col in range(3):
            base = axes[row, col].get_title()
            axes[row, col].set_title(f'{base}\n{label} \u2192 Pz',
                                     fontsize=10, fontweight='bold')

    fig.suptitle(
        'Figure 6.  Complexity coupling topography to the Pz P300\n'
        '$R^2$ and $\\beta$ are real-trial values; AUR = '
        '$|\\beta_{pseudo}|/|\\beta_{real}|$  '
        '(blue < 1 = stimulus-locked; red > 1 = autocorrelation; '
        'capped at 3; diverging at 1)',
        fontsize=11, y=1.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig6_complexity_topo.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig6 -> {out}")

def _slopes(df, col, n_min=5):
    def rz(s):
        med = np.median(s)
        mad = 1.4826 * np.median(np.abs(s - med))
        return (s - med) / mad if mad > 0 else s - med

    d = df.dropna(subset=[col, 'p300']).copy()
    for c in (col, 'p300'):
        d[c+'_z'] = d.groupby('subject')[c].transform(rz)
    out = {}
    for subj, g in d.groupby('subject'):
        x = g[col+'_z'].values; y = g['p300_z'].values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < n_min or np.std(x) < 1e-10:
            continue
        out[subj] = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return pd.Series(out)

def figure_7_entropy_heterogeneity():
    prim_path = os.path.join(RESULTS_DIR, FILES['heterogeneity_primary'])
    ds_path   = os.path.join(RESULTS_DIR, FILES['heterogeneity_ds006018'])

    datasets = []
    if os.path.exists(prim_path):
        datasets.append(('Primary (N = 27)', pd.read_csv(prim_path)))
    else:
        print(f"  [warn] fig7: {prim_path} not found")
    if os.path.exists(ds_path):
        datasets.append(('ds006018 (N = 90)', pd.read_csv(ds_path)))
    else:
        print(f"  [warn] fig7: {ds_path} not found")

    if not datasets:
        print("  fig7 skipped: no heterogeneity data found")
        return

    measures = [('pe', 'Permutation entropy'),
                ('se', 'Sample entropy'),
                ('lz', 'Lempel-Ziv')]

    fig, axes = plt.subplots(len(datasets), 3,
                             figsize=(13, 3.8 * len(datasets)),
                             squeeze=False)

    for i, (dname, df) in enumerate(datasets):
        for j, (col, mlabel) in enumerate(measures):
            ax = axes[i][j]
            s = _slopes(df, col)
            if len(s) == 0:
                ax.set_axis_off(); continue
            pos = int((s > 0).sum()); n = len(s)
            ax.hist(s.values, bins=15, color=STANDARD_COLOR,
                    edgecolor='white', alpha=0.85)
            ax.axvline(0, color='k', lw=1)
            ax.axvline(s.mean(), color=TARGET_COLOR, lw=1.5, ls='--',
                       label=f'mean {s.mean():+.3f}')
            ax.set_title(f'{dname} — {mlabel}\n{pos}/{n} positive',
                         fontsize=9.5)
            ax.set_xlabel('Per-subject slope')
            if j == 0:
                ax.set_ylabel('Participants')
            ax.legend(loc='upper right', frameon=False, fontsize=8)

    fig.suptitle(
        'Figure 7.  Per-subject entropy–P300 coupling slopes',
        fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig7_entropy_heterogeneity.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"  fig7 -> {out}")

if __name__ == '__main__':
    banner("13_make_figures.py — all manuscript figures (1–7)")
    figure_1_erp_and_topo()
    figure_2_rectified_fz()
    figure_3_pseudotrial()
    figure_4_cross_channel()
    figure_5_same_channel()
    figure_6_complexity_topo()
    figure_7_entropy_heterogeneity()
    print()
    banner("Done — check results/figures/")
