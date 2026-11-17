import json, pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F = json.load(open(sys.argv[1])); OUT = sys.argv[2]
INK = '#0b0b0b'; MUT = '#52514e'; GRID = '#e4e3df'; BLUE = '#2a78d6'; DARK = '#1c3f6e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})
D = pd.DataFrame(F['reffree_dbeta']); A = pd.DataFrame(F['reffree_agreement'])
SAME = ('M8-D', 'M9a-D', 'M10-D', 'M24-D'); LOC = {'M4a-D': 'D', 'M18-D': 'D', 'M21-D': '^'}
R = D[(D.quantity == 'real_slope') & (~D.model_id.isin(SAME))]
models = sorted(R.model_id.unique())
def prof(ds, ref):
    g = R[(R.dataset == ds) & (R.reference == ref)].set_index('model_id')['mean']
    return np.array([g[m] for m in models])
mn = lambda s: s.replace('-', '−')
fig, axs = plt.subplots(1, 4, figsize=(8.6, 2.55))
lim = 0.31
panels = [('mastoid', 'Linked mastoids'), ('average', 'Common average'), ('csd', 'No reference (surface Laplacian)')]
for j, (ref, lab) in enumerate(panels):
    ax = axs[j]; x = prof('erpcore', ref); y = prof('ds006018', ref)
    ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=0); ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    oth = [k for k, m in enumerate(models) if m not in LOC]
    ax.scatter(x[oth], y[oth], s=16, color=BLUE, edgecolor='white', linewidth=0.5, zorder=2, label='other 14 statistics')
    for mk, l2 in (('D', 'mean, median Fz 0–150 ms'), ('^', 'slope Fz 0–150 ms')):
        kk = [models.index(m) for m, v in LOC.items() if v == mk]
        ax.scatter(x[kk], y[kk], s=28, marker=mk, facecolor='white', edgecolor=BLUE, linewidth=1.2, zorder=3, label=l2)
    t = A[(A.analysis == 'cross_dataset') & (A.quantity == 'real_slope') & (A.contrast == ref)].iloc[0]
    ax.text(0.97, 0.04, mn(f"r = {t.value:+.2f}\n[{t.lo:+.2f}, {t.hi:+.2f}]"), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.tick_params(labelsize=7)
    ax.set_title(f"({'abc'[j]}) {lab}", fontsize=7.6, color=INK, loc='left')
    ax.set_xlabel('ERP CORE β', fontsize=7.5)
    if j == 0:
        ax.set_ylabel('ds006018 β', fontsize=7.5); ax.legend(loc='upper left', frameon=False, fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)
ax = axs[3]
ax.plot([-lim, lim], [lim, -lim], color=GRID, lw=1, zorder=0); ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.axvline(0, color=GRID, lw=0.6, zorder=0)
for ds, mk, fc, lab in (('erpcore', 'o', BLUE, 'ERP CORE'), ('ds006018', 's', 'white', 'ds006018')):
    ax.scatter(prof(ds, 'mastoid'), prof(ds, 'average'), s=18, marker=mk, facecolor=fc, edgecolor=BLUE if fc == 'white' else 'white', linewidth=0.9, zorder=2, label=lab)
w = A[A.analysis.str.startswith('within_') & A.analysis.str.endswith('between_references') & (A.quantity == 'real_slope') & (A.contrast == 'mastoid~average')]
e = w[w.analysis.str.contains('erpcore')].iloc[0]; d = w[w.analysis.str.contains('ds006018')].iloc[0]
ax.text(0.97, 0.96, mn(f"ERP CORE r = {e.value:+.2f}\nds006018 r = {d.value:+.2f}"), transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.tick_params(labelsize=7)
ax.set_title('(d) Same data, two references', fontsize=7.6, color=INK, loc='left')
ax.set_xlabel('β, linked mastoids', fontsize=7.5); ax.set_ylabel('β, common average', fontsize=7.5)
ax.legend(loc='lower left', frameon=False, fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(); fig.savefig(OUT, dpi=300); print('ok')
