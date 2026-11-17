import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
R = sys.argv[1]; OUT = sys.argv[2]
INK = '#0b0b0b'; MUT = '#52514e'; GRID = '#e4e3df'; CT = '#6250d6'; CS = '#1baf7a'; SH = '#f1f0ec'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': MUT, 'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})
T = {'dist': 'r150_g05_ic20', 'cz': 'r150_g05_cz_ic20', 'mast': 'r150_g05_mastoid_ic20', 'aver': 'r150_g05_average_ic20'}
COLS = [('dist', 'As distributed'), ('cz', 'Cz'), ('mast', 'Linked mastoids'), ('aver', 'Common average')]
ROWS = [('erpcore', 'Fz'), ('erpcore', 'Pz'), ('ds006018', 'Fz'), ('ds006018', 'Pz')]
DN = {'erpcore': 'ERP CORE', 'ds006018': 'ds006018'}
cache = {}
def ga(ds, ref, ch):
    key = (ds, ref)
    if key not in cache:
        cache[key] = pd.read_csv(f'{R}/erp_{ds}_{T[ref]}.csv')
    e = cache[key]
    tc = [c for c in e.columns if c not in ('subject', 'condition', 'channel', 'n_trials', 'sfreq')]
    tt = np.array([float(c) for c in tc])
    out = {}
    for cond in ('target', 'standard'):
        x = e[(e.condition == cond) & (e.channel == ch)][tc].values * 1e6
        out[cond] = (x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)), len(x))
    return tt, out
fig, axes = plt.subplots(4, 4, figsize=(7.2, 6.4), sharex=True)
ylims = {}
for i, (ds, ch) in enumerate(ROWS):
    for j, (ref, lab) in enumerate(COLS):
        ax = axes[i, j]
        src = ref
        if ds == 'ds006018' and ref == 'dist':
            ax.axis('off')
            ax.text(0.5, 0.5, 'as distributed\n= Cz', transform=ax.transAxes, ha='center', va='center', color=MUT, fontsize=7.5)
            continue
        if ds == 'ds006018' and ref == 'cz':
            src = 'dist'
        tt, g = ga(ds, src, ch)
        ax.axvspan(0, 150, color=SH, zorder=0)
        ax.axvspan(300, 600, color=SH, zorder=0)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        for cond, col in (('standard', CS), ('target', CT)):
            m, se, n = g[cond]
            ax.fill_between(tt, m - se, m + se, color=col, alpha=0.18, lw=0)
            ax.plot(tt, m, color=col, lw=1.2, label=cond.capitalize() if (i, j) == (0, 0) else None)
        if i == 0:
            ax.set_title(lab, fontsize=8, color=INK)
        if (ds == 'erpcore' and j == 0) or (ds == 'ds006018' and j == 1):
            ax.set_ylabel(f'{DN[ds]} {ch} (µV)')
        if i == 3:
            ax.set_xlabel('Time (ms)')
        ax.set_xlim(-200, 800)
        ax.set_xticks([0, 300, 600])
lims = {}
for i in range(4):
    vals = [a.get_ylim() for a in axes[i] if a.axison]
    lo = min(v[0] for v in vals); hi = max(v[1] for v in vals)
    for a in axes[i]:
        if a.axison:
            a.set_ylim(lo, hi)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=300)
print('ok')
