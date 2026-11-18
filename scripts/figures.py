"""Publication figures from prespecified aggregate outputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
sources=['mimic','eicu','sicdb'];names={'mimic':'MIMIC-IV','eicu':'eICU-CRD','sicdb':'SICdb'}
models=['phosphate_only','phosphate_creatinine','six_variable']
labels={'phosphate_only':'Phosphate','phosphate_creatinine':'Phosphate + creatinine','six_variable':'Six variables'}
colors={'phosphate_only':'#7B8794','phosphate_creatinine':'#238B8B','six_variable':'#24466B'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelsize':11,'axes.titlesize':12,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight')
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(F/(name+'.tiff'),dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig)

flow={s:json.loads((ROOT/'audit'/(s+'_extraction.json')).read_text())['flow'] for s in sources}
fig,axes=plt.subplots(1,3,figsize=(10.8,7.2));fig.subplots_adjust(wspace=.18)
stages=['one_adult_admission_per_person','positive_feed_first_7_icu_days','remaining_in_icu_at_index','feed_documented_by_index',
 'feeding_tube_documented_if_required','valid_available_baseline_phosphate','no_known_phosphate_below_2_in_previous_24h','followup_phosphate_observed']
captions=['One adult admission\nper person','Recorded enteral feed\nin first 7 ICU days','Remaining in ICU/case\nat feeding + 24 h','Feed record entered\nby the index time','Feeding tube confirmed\n(SICdb requirement)','Available phosphate\nin previous 24 h','No known phosphate\n<2.0 mg/dL in that window','Subsequent phosphate\nwithin 48 h']
for ax,s in zip(axes,sources):
 ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(names[s],fontweight='bold',pad=12)
 yy=np.linspace(.93,.11,len(stages))
 for i,(st,cap,y) in enumerate(zip(stages,captions,yy)):
  if st=='feeding_tube_documented_if_required' and s!='sicdb':cap='Explicit enteral\nfeed documentation'
  ax.add_patch(FancyBboxPatch((.04,y-.044),.92,.089,boxstyle='round,pad=0.012',linewidth=.7,edgecolor='#CCD3DB',facecolor='#F3F6F9' if i<7 else '#E4EDF5'))
  ax.text(.5,y+.005,cap,ha='center',va='center',fontsize=9)
  ax.text(.5,y-.029,f'n = {flow[s][st]:,}',ha='center',va='center',fontsize=9,fontweight='bold')
  if i<len(stages)-1:ax.annotate('',xy=(.5,yy[i+1]+.053),xytext=(.5,y-.055),arrowprops={'arrowstyle':'->','color':'#7B8794'})
 ax.text(.5,.016,f"Events: {flow[s]['primary_events']:,} ({flow[s]['primary_events']/flow[s]['followup_phosphate_observed']:.1%})",ha='center',fontsize=10,fontweight='bold')
save(fig,'Figure_1_cohort_flow')

perf=pd.read_csv(R/'performance.csv');cals=pd.read_csv(R/'calibration_groups.csv')
fig,axes=plt.subplots(2,3,figsize=(11,7.4));fig.subplots_adjust(hspace=.40,wspace=.30,bottom=.13)
for j,s in enumerate(sources):
 pr=pd.read_parquet(ROOT/'private'/(s+'_predictions.parquet'))
 ax=axes[0,j];ax.plot([0,1],[0,1],ls='--',c='#CED4DA',lw=1)
 for m in models:
  x,y,_=roc_curve(pr.event,pr[m]);v=perf[(perf.source==s)&(perf.model==m)&(perf.metric=='auc')].iloc[0]
  ax.plot(x,y,c=colors[m],lw=1.8,label=f"{labels[m]}  {v.estimate:.3f}")
 ax.set(xlabel='False-positive fraction',ylabel='Sensitivity' if j==0 else '',xlim=(0,1),ylim=(0,1),title=names[s])
 ax.legend(loc='lower right',fontsize=7.7,frameon=False)
 ax.text(-.13,1.07,chr(65+j),transform=ax.transAxes,fontweight='bold',fontsize=13)
 ax=axes[1,j];ax.plot([0,.55],[0,.55],ls='--',c='#8D959D',lw=1)
 for m in models:
  z=cals[(cals.source==s)&(cals.model==m)]
  ax.plot(z.predicted,z.observed,'o-',c=colors[m],lw=1.3,ms=3.3,label=labels[m])
  if m=='six_variable':
   lo,hi=proportion_confint(z.events.to_numpy(),z.n.to_numpy(),method='wilson')
   ax.errorbar(z.predicted,z.observed,yerr=[z.observed-lo,hi-z.observed],fmt='none',c=colors[m],lw=.8,alpha=.55,capsize=2)
 ax.set(xlabel='Mean predicted risk',ylabel='Observed proportion' if j==0 else '',xlim=(0,.55),ylim=(0,.55))
 ax.text(-.13,1.07,chr(68+j),transform=ax.transAxes,fontweight='bold',fontsize=13)
 handles,leg=axes[1,0].get_legend_handles_labels()
fig.legend(handles,leg,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
save(fig,'Figure_2_discrimination_calibration')

dif=pd.read_csv(R/'model_differences.csv')
fig,axes=plt.subplots(1,2,figsize=(10.7,4.3));fig.subplots_adjust(wspace=.45,left=.14,right=.98,bottom=.17)
for ax,cmp,title in zip(axes,['six_variable_minus_phosphate_only','six_variable_minus_phosphate_creatinine'],['Adding five variables to phosphate','Adding four variables to phosphate + creatinine']):
 z=dif[(dif.comparison==cmp)&dif.metric.eq('auc')].set_index('source').loc[sources]
 ys=np.arange(3)[::-1]
 ax.axvline(0,color='#91989F',lw=1,ls='--')
 ax.errorbar(z.estimate,ys,xerr=[z.estimate-z.ci_low,z.ci_high-z.estimate],fmt='o',c='#24466B',capsize=4,lw=1.7)
 ax.set(yticks=ys,yticklabels=[names[s] for s in sources],xlim=(-.028,.060),ylim=(-.6,2.6),xlabel='Difference in AUROC',title=title)
 ax.tick_params(axis='y',length=0)
 for y,(_,r) in zip(ys,z.iterrows()):ax.text(r.estimate,y+.21,f'{r.estimate:+.3f} ({r.ci_low:+.3f}, {r.ci_high:+.3f})',ha='center',fontsize=8.5)
save(fig,'Figure_3_incremental_value')

dca=pd.read_csv(R/'decision_curves.csv')
fig,axes=plt.subplots(1,3,figsize=(11,3.8));fig.subplots_adjust(wspace=.30,bottom=.22)
for j,(ax,s) in enumerate(zip(axes,sources)):
 z=dca[dca.source.eq(s)]
 for m in models:
  q=z[z.model.eq(m)];ax.plot(q.threshold,q.net_benefit,label=labels[m],c=colors[m],lw=1.8)
 q=z[z.model.eq('six_variable')];ax.plot(q.threshold,q.monitor_all,c='#B68158',ls='--',label='Flag all');ax.axhline(0,c='#777777',ls=':',label='Flag none')
 ax.set(title=names[s],xlabel='Risk threshold',ylabel='Net benefit' if j==0 else '',xlim=(.05,.3),ylim=(-.04,.16))
handles,leg=axes[0].get_legend_handles_labels();fig.legend(handles,leg,ncol=5,loc='lower center',frameon=False,fontsize=9)
save(fig,'Figure_S1_exploratory_decision_curves')
print('4 figures written in PNG, PDF and 600-dpi TIFF',flush=True)
