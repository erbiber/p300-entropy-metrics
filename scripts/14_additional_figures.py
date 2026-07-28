"""
14_additional_figures.py  —  Figure 8 of the manuscript.

Rewritten from 14_additinal_figures.py (filename typo corrected).

Change from the previous version: the forest plot defaulted to
phase013_dbeta_config4.csv, which holds only the four amplitude/energy models,
while the Figure 8 caption describes "all nine models". The default is now
phase013_dbeta_config4_K1000.csv.

Figure 8 conventions taken from the caption:
  * models grouped by family - cross-channel (M1, M4a), same-channel (M8, M9a),
    complexity (M13-M15, M22, M23)
  * filled markers = BCa interval excludes zero; open markers = includes zero
  * vertical line at dbeta = 0
  * an interval extending past the axis is marked with an arrow (M15, primary)
  * numeric dbeta and surrogate p printed at right
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('DOF_RESULTS_DIR', os.path.join(HERE, '..', 'results'))
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)

PRIMARY_COLOR = '#20808D'
CROSSVAL_COLOR = '#A13544'

ORDER = [('M1_RMS_Fz_0_150', 'M1  cross-channel RMS'),
         ('M4a_mean_Fz', 'M4a  cross-channel mean'),
         ('M8_RMS_Pz_0_150', 'M8  same-channel energy'),
         ('M9a_mean_Pz', 'M9a  same-channel mean'),
         ('M13_PE_Fz_0_150', 'M13  permutation entropy'),
         ('M14_SE_Fz_0_150', 'M14  sample entropy'),
         ('M15_LZ_Fz_0_150', 'M15  Lempel-Ziv'),
         ('M22_HJORTHMOB_Fz_0_150', 'M22  Hjorth mobility'),
         ('M23_HJORTHCPLX_Fz_0_150', 'M23  Hjorth complexity')]
FAMILY_BREAKS = {2: 'Same-channel continuity', 4: 'Signal complexity'}
XLIM = (-0.28, 0.28)


def _save(fig, stem):
    png = os.path.join(FIG, stem + '.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {png}')


def figure_08_dbeta_forest(csv='phase013_dbeta_config4_K1000.csv'):
    path = os.path.join(RES, csv)
    if not os.path.exists(path):
        print(f'  fig08 skipped: {path} not found')
        return
    d = pd.read_csv(path)
    present = [m for m, _ in ORDER if m in set(d.model)]
    if len(present) < 9:
        print(f'  [warn] fig08: {csv} carries {len(present)} of the 9 models the caption describes')

    n_prim = int(d.query("dataset=='erp_core'").n_used.iloc[0])
    n_cv = int(d.query("dataset=='ds006018'").n_used.iloc[0])

    rows = [(m, lab) for m, lab in ORDER if m in present]
    fig, ax = plt.subplots(figsize=(11.5, 1.05 * len(rows) + 2.0))
    yticks, ylabels = [], []

    for i, (model, label) in enumerate(rows):
        base = len(rows) - i
        for off, tag, colour, nname in [(+0.19, 'erp_core', PRIMARY_COLOR, 'primary'),
                                        (-0.19, 'ds006018', CROSSVAL_COLOR, 'cross-validation')]:
            r = d[(d.model == model) & (d.dataset == tag)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = base + off
            lo, hi = r.bca_lo, r.bca_hi
            excludes_zero = (lo > 0) or (hi < 0)
            clipped_hi = min(hi, XLIM[1] - 0.006)
            clipped_lo = max(lo, XLIM[0] + 0.006)
            ax.plot([clipped_lo, clipped_hi], [y, y], color=colour, lw=1.9,
                    solid_capstyle='butt', zorder=2)
            if hi > XLIM[1] - 0.006:
                ax.annotate('', xy=(XLIM[1] - 0.001, y), xytext=(clipped_hi, y),
                            arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.9))
            if lo < XLIM[0] + 0.006:
                ax.annotate('', xy=(XLIM[0] + 0.001, y), xytext=(clipped_lo, y),
                            arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.9))
            ax.plot([r.dbeta], [y], marker='o', ms=7.5, zorder=3,
                    color=colour if excludes_zero else 'white',
                    markeredgecolor=colour, markeredgewidth=1.7)
            p = r.surrogate_p
            ptxt = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            ax.text(XLIM[1] + 0.012, y, f'{r.dbeta:+.3f}   {ptxt}', va='center',
                    fontsize=7.8, color=colour, family='monospace')
        yticks.append(base)
        ylabels.append(label)

    for idx, name in FAMILY_BREAKS.items():
        yb = len(rows) - idx + 0.5
        ax.axhline(yb, color='#bbbbbb', lw=0.8, ls=':')
        ax.text(XLIM[0] + 0.006, yb - 0.12, name, fontsize=7.5, style='italic', color='#666666')

    ax.axvline(0, color='k', lw=1.0)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(*XLIM); ax.set_ylim(0.3, len(rows) + 0.9)
    ax.set_xlabel(r'$\Delta\beta$  =  $\beta_{real}$ - $\beta_{pseudo}$   '
                  '(negative = diluted;  zero = carried by background;  positive = stimulus-locked)',
                  fontsize=9.5)
    ax.spines[['top', 'right']].set_visible(False)

    handles = [plt.Line2D([], [], color=PRIMARY_COLOR, marker='o', lw=1.9, ms=7,
                          label=f'Primary, ERP CORE (n = {n_prim})'),
               plt.Line2D([], [], color=CROSSVAL_COLOR, marker='o', lw=1.9, ms=7,
                          label=f'Cross-validation, ds006018 (n = {n_cv})'),
               plt.Line2D([], [], color='#555555', marker='o', lw=0, ms=7,
                          markerfacecolor='white', markeredgewidth=1.7,
                          label='Open marker: interval includes zero')]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc='lower left',
              bbox_to_anchor=(0.0, -0.20), ncol=3)

    fig.suptitle('Figure 8. Difference in standardised coupling between real and pseudotrials, '
                 'with 95% subject-cluster BCa intervals', fontsize=11.5, y=0.99)
    _save(fig, 'fig08_dbeta_forest')


def figure_04b_persubject_slopes(csv='phase013_persubject_config4_K1000.csv'):
    """Per-subject real vs pseudo slopes; supporting panel for Figure 4."""
    path = os.path.join(RES, csv)
    if not os.path.exists(path):
        print(f'  fig04b skipped: {path} not found')
        return
    d = pd.read_csv(path)
    d = d[(d.dataset == 'erp_core') & d.used]
    models = [m for m, _ in ORDER if m in set(d.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(2.05 * len(models), 3.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        s = d[d.model == m]
        ax.scatter(s.pseudo_mean, s.real_slope, s=20, alpha=.65,
                   color=PRIMARY_COLOR, edgecolor='none')
        lim = np.nanmax(np.abs(np.r_[s.pseudo_mean, s.real_slope])) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color='#999999', lw=.9, ls='--')
        ax.axhline(0, color='k', lw=.5); ax.axvline(0, color='k', lw=.5)
        ax.set_title(m.split('_')[0], fontsize=9, fontweight='bold')
        ax.set_xlabel('pseudo', fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel('real slope', fontsize=8)
    fig.suptitle('Per-participant real vs mean surrogate slope (primary dataset, n = %d)'
                 % d.subject.nunique(), fontsize=10.5)
    fig.tight_layout()
    _save(fig, 'fig04b_persubject_real_vs_pseudo')


def main():
    print('Figure 8 (and the Figure 4 support panel) ->', FIG)
    figure_08_dbeta_forest()
    figure_04b_persubject_slopes()


if __name__ == '__main__':
    main()
