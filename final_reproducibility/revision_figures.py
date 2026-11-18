#!/usr/bin/env python3
"""Regenerate the revised figures. Uses saved impulse responses and anonymised data."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy import stats
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'figures';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'original'));import plots as old
old.configure_style();plt.rcParams.update({'font.size':9,'axes.titlesize':10})
BLUE,ORANGE='#147CE5','#E77900';COL={'individual':BLUE,'optimised':ORANGE}
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
from figure1 import render as render_figure1
render_figure1()
# Reuse unchanged behaviour figures, maintaining published values and aesthetics.
fb,srm=old.load_trials();old.figure_front_back(fb,OUT);old.figure_srm(srm,OUT)
# Fig 4 separated correlations, never pooled across configurations.
d=pd.read_csv(ROOT/'results/prediction_points.csv');c=pd.read_csv(ROOT/'results/within_configuration_correlations.csv')
fig,axs=plt.subplots(2,2,figsize=(7.1,5.4),sharey=True,layout='constrained')
for row,ses in enumerate(['before','after']):
 for col,cfg in enumerate(['T30 M150','T150 M30']):
  ax=axs[row,col];q=d[(d.session==ses)&(d.hrtf=='individual')&(d.configuration==cfg)];x=q.predicted_release_db;y=q.srm_pp
  ax.scatter(x,y,color=BLUE,s=24,edgecolor='white',lw=.4);b=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),100);ax.plot(xx,np.polyval(b,xx),color=BLUE,lw=1)
  rr=c[(c.session==ses)&(c.configuration==cfg)].iloc[0];ax.text(.04,.96,f'r = {rr.r:.2f}; Holm p = {rr.p_holm:.3f}',transform=ax.transAxes,va='top',fontsize=9)
  ax.set_title(f'{chr(65+row*2+col)}  {cfg.replace(" ","")} · {"Pre" if ses=="before" else "Post"}-training')
  ax.axhline(0,color='.65',lw=.6);ax.grid(color='.93',lw=.5);ax.set_ylim(-22,48)
  if col==0:ax.set_ylabel('Behavioural SRM (pp)')
  if row==1:ax.set_xlabel('Predicted release (dB)')
save(fig,'figure4_predictions')
# Figure 5: pre vs post, identity reference; no coupled change-score inference.
fbc=old.means(fb,['session','hrtf']);sc=old.means(srm[srm.configuration.isin(old.SEPARATED)],['session','hrtf'])
# Equal weighting of configuration participant means, matching analysis.py.
sc0=old.means(srm[srm.configuration.isin(old.SEPARATED)],['session','hrtf','configuration']);sc=sc0.groupby(['participant','session','hrtf']).accuracy_pct.mean().reset_index()
fig,axs=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
for ax,dat,title in zip(axs,[fbc,sc],['A  Front-back discrimination','B  Separated speech recognition']):
 for h in ['individual','optimised']:
  q=dat[dat.hrtf==h].pivot(index='participant',columns='session',values='accuracy_pct');ax.scatter(q.before,q.after,color=COL[h],label=h.capitalize(),s=28,edgecolor='white',lw=.4,alpha=.9)
 ax.plot([0,100],[0,100],ls='--',color='.5',lw=.8);ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Pre-training accuracy (%)',title=title);ax.set_aspect('equal');ax.grid(color='.92',lw=.5)
axs[0].set_ylabel('Post-training accuracy (%)');axs[1].legend(frameon=False,fontsize=8,loc='upper left');save(fig,'figure5_prepost')
# Appendix progress: disjoint bins of ten trials; final bin may be shorter.
t=pd.read_csv(ROOT/'inputs/training_trials.csv');summary=pd.read_csv(ROOT/'inputs/training_summary.csv').set_index('participant')
fig,axs=plt.subplots(6,3,figsize=(7.1,8.3),sharex=True,sharey=True,layout='constrained')
for ax,(pid,g) in zip(axs.flat,t.groupby('participant')):
 b=g.assign(bin=(g.trial-1)//10).groupby('bin').agg(trial=('trial','max'),accuracy=('correct','mean'))
 cap=summary.loc[pid,'stop']=='trial_cap';ax.plot(b.trial,100*b.accuracy,marker='.',ms=3,color=ORANGE if cap else BLUE,lw=.9);ax.axhline(50,color='.75',ls=':',lw=.6)
 ax.set_title(f'{pid} · {len(g)} trials',fontsize=8);ax.set(xlim=(0,215),ylim=(0,105),xticks=[0,100,200],yticks=[0,50,100]);ax.tick_params(labelsize=7);ax.grid(axis='y',color='.93',lw=.4)
fig.supxlabel('Training trial');fig.supylabel('Correct responses per ten-trial bin (%)');save(fig,'appendix_training')
print('Revised figures written to',OUT)
