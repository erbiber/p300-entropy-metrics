import json, pandas as pd, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F = json.load(open(sys.argv[1])); OUT = sys.argv[2]
INK = '#0b0b0b'; MUT = '#52514e'; C = {'erpcore': '#2a78d6', 'ds006018': '#eb6834'}; DN = {'erpcore': 'ERP CORE', 'ds006018': 'ds006018'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})
eq = pd.DataFrame(F['common_dbeta_free']); eq = eq[eq.contrast == 'd']
NM = {'M13-D': 'PE', 'M14-D': 'SE', 'M15-D': 'LZ', 'M22-D': 'Hjorth mob.', 'M23-D': 'Hjorth cplx.'}
REFS = ['distributed', 'mastoid', 'average']; LAB = {'distributed': 'As distributed', 'mastoid': 'Linked mastoids', 'average': 'Common average'}
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
for ax, ref in zip(axs, REFS):
    ax.axvspan(-0.05, 0.05, color='#eef3fb', zorder=0); ax.axvline(0, color=MUT, lw=0.6)
    for j, ds in enumerate(['erpcore', 'ds006018']):
        g = eq[(eq.reference == ref) & (eq.dataset == ds)].set_index('model_id')
        for i, m in enumerate(NM):
            yy = i + (0.18 if j == 0 else -0.18); r = g.loc[m]
            ax.plot([r.bca90_lo, r.bca90_hi], [yy, yy], color=C[ds], lw=1.4, solid_capstyle='round')
            ax.plot(r.dbeta, yy, 'o' if j == 0 else 's', ms=3.5, color=C[ds], mec='white', mew=0.5, label=DN[ds] if i == 0 else None)
    ax.set_title(LAB[ref], fontsize=8, color=INK); ax.set_xlim(-0.12, 0.12); ax.set_xlabel('Δβ with 90% BCa interval')
axs[0].set_yticks(range(5)); axs[0].set_yticklabels(list(NM.values())); axs[0].invert_yaxis(); axs[2].legend(frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout(); fig.savefig(OUT, dpi=300); print('ok')
