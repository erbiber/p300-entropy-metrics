import json, pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F = json.load(open(sys.argv[1])); OUT = sys.argv[2]
INK = '#0b0b0b'; MUT = '#52514e'; GRID = '#e4e3df'; COL = {'real_slope': '#2a78d6', 'pseudo_mean': '#eb6834', 'd': '#1baf7a'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})
PR = F['common_profiles_free']; TR = pd.DataFrame(F['common_transport_free'])
CROSS = [m for m in PR['erpcore_distributed'] if m not in ('M8-D', 'M9a-D', 'M10-D', 'M24-D')]
COLS = [('distributed', 'distributed', 'As distributed'), ('cz', 'distributed', 'Both on Cz'), ('mastoid', 'mastoid', 'Linked mastoids'), ('average', 'average', 'Common average')]
ROW = {'real_slope': 'Real trials: slope', 'pseudo_mean': 'Event-free pseudotrials: slope', 'd': 'Δβ = real − pseudotrial'}
LOCM = {'M4a-D': 'D', 'M18-D': 'D', 'M21-D': '^'}
fig, axs = plt.subplots(3, 4, figsize=(8.6, 6.6))
for i, q in enumerate(ROW):
    vals = [PR[f'{ds}_{r}'][m][q] for ds, refs in (('erpcore', ('distributed', 'cz', 'mastoid', 'average')), ('ds006018', ('distributed', 'mastoid', 'average'))) for r in refs for m in CROSS]
    lim = max(abs(min(vals)), abs(max(vals))) * 1.12
    for j, (re_, rd_, lab) in enumerate(COLS):
        ax = axs[i, j]
        x = [PR[f'erpcore_{re_}'][m][q] for m in CROSS]
        y = [PR[f'ds006018_{rd_}'][m][q] for m in CROSS]
        ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=0); ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        oth = [k for k, m in enumerate(CROSS) if m not in LOCM]
        ax.scatter([x[k] for k in oth], [y[k] for k in oth], s=16, color=COL[q], edgecolor='white', linewidth=0.5, zorder=2, label='other 14 statistics')
        for mk, l2 in (('D', 'mean, median Fz 0–150 ms'), ('^', 'slope Fz 0–150 ms')):
            kk = [CROSS.index(m) for m, v in LOCM.items() if v == mk]
            ax.scatter([x[k] for k in kk], [y[k] for k in kk], s=28, marker=mk, facecolor='white', edgecolor=COL[q], linewidth=1.2, zorder=3, label=l2)
        if i == 0 and j == 0:
            ax.legend(loc='upper left', frameon=False, fontsize=6.3, handletextpad=0.3, borderaxespad=0.2)
        t = TR[(TR.quantity == q) & (TR.contrast == re_)].iloc[0]
        dg = 3 if max(abs(t['T']), abs(t.lo), abs(t.hi)) >= 0.995 else 2
        ax.text(0.97, 0.04, (f"r = {t['T']:+.{dg}f}\n[{t.lo:+.{dg}f}, {t.hi:+.{dg}f}]").replace("-", "−"), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=INK)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=7)
        if i == 0: ax.set_title(lab, fontsize=8, color=INK)
        if j == 0: ax.set_ylabel(f'{ROW[q]}\nds006018', fontsize=7)
        if i == 2: ax.set_xlabel('ERP CORE', fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT, dpi=300); print('ok')
