import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
MODELS = ['M9a_mean_Pz', 'M8_RMS_Pz_0_150', 'M1_RMS_Fz_0_150', 'M4a_mean_Fz']
LABELS = {'M9a_mean_Pz': 'same-channel mean', 'M8_RMS_Pz_0_150': 'same-channel RMS',
          'M1_RMS_Fz_0_150': 'cross-channel RMS', 'M4a_mean_Fz': 'cross-channel mean'}
COLORS = {'erp_core': '#2166ac', 'ds006018': '#b2182b'}

def figure_3_dbeta_forest(dbeta_csv='phase013_dbeta_config4.csv'):
    db = pd.read_csv(os.path.join(RESULTS_DIR, dbeta_csv))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = 0; yt = []; yl = []
    for m in MODELS:
        for ds, dsl in [('ds006018', 'cross-validation'), ('erp_core', 'primary')]:
            r = db[(db.dataset == ds) & (db.model == m)].iloc[0]
            ax.plot([r['bca_lo'], r['bca_hi']], [y, y], color=COLORS[ds], lw=2.2, solid_capstyle='round')
            ax.plot(r['dbeta'], y, 'o', color=COLORS[ds], ms=7, zorder=3)
            yt.append(y); yl.append('%s - %s' % (LABELS[m], dsl)); y += 1
        y += 0.6
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7)
    ax.axvspan(-0.02, 0.02, color='grey', alpha=0.12)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9)
    ax.set_xlabel('dbeta = beta(real) - beta(pseudotrial)  (95% CI)', fontsize=11)
    ax.legend([Line2D([0], [0], color=COLORS['erp_core'], lw=2.2), Line2D([0], [0], color=COLORS['ds006018'], lw=2.2)],
              ['Primary (ERP CORE)', 'Cross-validation (ds006018)'], loc='upper left', fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure3_dbeta_forest.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)

def figure_7_persubject(persub_csv='phase013_persubject_config4_K1000.csv'):
    ps = pd.read_csv(os.path.join(RESULTS_DIR, persub_csv))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(0)
    for ds, c, off in [('erp_core', COLORS['erp_core'], -0.15), ('ds006018', COLORS['ds006018'], 0.15)]:
        d = ps[(ps.dataset == ds) & (ps.model == 'M9a_mean_Pz') & ps.used]['d'].values
        axes[0].scatter(rng.uniform(off - 0.09, off + 0.09, len(d)), d, color=c, alpha=0.5, s=18)
        axes[0].plot([off - 0.13, off + 0.13], [np.mean(d)] * 2, color=c, lw=2.5)
    axes[0].axhline(0, color='k', lw=0.8, ls='--')
    axes[0].set_xticks([-0.15, 0.15]); axes[0].set_xticklabels(['Primary', 'Cross-val'])
    axes[0].set_ylabel('per-participant delta (real - pseudo)')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[1].bar([0, 1], [0.87, -0.09], color=['#4d9221', '#c51b7d'], width=0.5)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Same trials', 'Disjoint halves'], fontsize=9)
    axes[1].set_ylabel('cross-measure slope r')
    axes[1].set_ylim(-0.3, 1); axes[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure7_persubject.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    figure_3_dbeta_forest()
    figure_7_persubject()
