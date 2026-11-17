import json,sys,pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
F=json.load(open(sys.argv[1]));OUT=sys.argv[2]
C={'erpcore':'#2a78d6','ds006018':'#eb6834'};INK='#0b0b0b';MUT='#52514e';GRID='#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':MUT,'axes.labelcolor':INK,'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False})
REFS=['distributed','mastoid','average'];LAB={'distributed':'As distributed','mastoid':'Linked mastoids','average':'Common average'};DN={'erpcore':'ERP CORE','ds006018':'ds006018'}
L=F['lags_trials'];lags=[1,2,3,5,10,20]
fig,axs=plt.subplots(1,4,figsize=(7.0,2.6),sharey=True)
FT={'rms_base_fz':('RMS Fz, pre-stimulus','#2a78d6','o'),'sd_early_fz':('SD Fz, 0–150 ms','#eb6834','s'),'pe_early_fz':('PE Fz, 0–150 ms','#1baf7a','^')}
for ax,key in zip(axs,['erpcore_mastoid','erpcore_average','ds006018_mastoid','ds006018_average']):
    ax.axhline(0,color=MUT,lw=0.6)
    for f,(lab,col,mk) in FT.items():
        ax.plot(lags,[L[key][f][str(k)]['mean'] for k in lags],marker=mk,ms=3.2,lw=1.4,color=col,label=lab,mec='white',mew=0.4)
    ds,ref=key.split('_');ax.set_title(f"{DN[ds]}, {LAB[ref].lower()}",fontsize=7.5,color=INK);ax.set_xscale('log');ax.set_xticks(lags);ax.set_xticklabels(lags);ax.set_xlabel('Lag (retained trials)')
    ax.text(0.97,0.95,f"within-trial r = {L[key]['within_r']:+.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=6,color=INK)
axs[0].set_ylabel('Mean within-subject r');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,frameon=False,fontsize=6.5,loc='lower center',ncol=3)
fig.tight_layout(rect=(0,0.08,1,1));fig.savefig(OUT,dpi=300);plt.close();print('ok')
