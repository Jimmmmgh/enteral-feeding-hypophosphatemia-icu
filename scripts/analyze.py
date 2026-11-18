"""Prespecified nested models, sealed development, independent external evaluation."""
from pathlib import Path
import argparse,json,hashlib,datetime,platform,warnings
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import sklearn

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'private';O=ROOT/'results';O.mkdir(exist_ok=True)
A=ROOT/'audit';SEED=20260928
FEATURES=['phosphate','creatinine','magnesium','potassium','age','bmi']
SPECS={'phosphate_only':['phosphate'],'phosphate_creatinine':['phosphate','creatinine'],'six_variable':FEATURES}

def rcs(x,k):
 a,b,c=k
 return (np.maximum(x-a,0)**3-np.maximum(x-b,0)**3*(c-a)/(c-b)+np.maximum(x-c,0)**3*(b-a)/(c-b))/(c-a)**2

def design(df,spec):
 out={}
 for f in spec['features']:
  raw=df[f].to_numpy(dtype=float);x=np.where(np.isfinite(raw),raw,spec['medians'][f])
  if f=='phosphate':out[f]=x;out['phosphate_rcs']=rcs(x,spec['knots'])
  elif f=='creatinine':out['log_creatinine']=np.log(x)
  else:out[f]=x
  if f in spec['missing_flags']:out[f+'_missing']=(~np.isfinite(raw)).astype(float)
 return pd.DataFrame(out,index=df.index)

def fit(df,y,features):
 spec={'features':features,'medians':{f:float(df[f].median()) for f in features},
  'missing_flags':[f for f in features if df[f].isna().any() and not df[f].isna().all()],
  'knots':np.quantile(df.phosphate,[.1,.5,.9]).tolist(),'penalty':'ridge','C':1.0}
 x=design(df,spec);spec['columns']=x.columns.tolist()
 spec['mean']=x.mean().tolist();spec['scale']=x.std(ddof=0).replace(0,1).tolist()
 z=(x.to_numpy()-np.array(spec['mean']))/np.array(spec['scale'])
 model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-9).fit(z,y)
 spec['coef']=model.coef_[0].tolist();spec['intercept']=float(model.intercept_[0]);spec['iterations']=int(model.n_iter_[0])
 return spec

def predict(df,spec):
 x=design(df,spec)[spec['columns']].to_numpy()
 z=(x-np.array(spec['mean']))/np.array(spec['scale'])
 return expit(z@np.array(spec['coef'])+spec['intercept'])

def calibration(y,p,w):
 z=logit(np.clip(p,1e-8,1-1e-8));X=np.column_stack([np.ones(len(z)),z])
 beta=np.array([0.,1.])
 for _ in range(30):
  q=expit(X@beta);H=X.T@((w*q*(1-q))[:,None]*X)+np.eye(2)*1e-10
  step=np.linalg.solve(H,X.T@(w*(y-q)));beta+=step
  if np.max(np.abs(step))<1e-8:break
 alpha=0.
 for _ in range(30):
  q=expit(z+alpha);step=np.sum(w*(y-q))/max(np.sum(w*q*(1-q)),1e-10);alpha+=step
  if abs(step)<1e-8:break
 return float(alpha),float(beta[1])

def metrics(y,p,w=None,cal=True):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
 if w is None:w=np.ones(len(y))
 if (w@y)<=0 or (w@(1-y))<=0:return {}
 out={'auc':float(roc_auc_score(y,p,sample_weight=w)),
  'average_precision':float(average_precision_score(y,p,sample_weight=w)),
  'brier':float(np.average((y-p)**2,weights=w)),
  'prevalence':float(np.average(y,weights=w)),
  'observed_expected':float(np.sum(w*y)/np.sum(w*p))}
 if cal:out['calibration_intercept'],out['calibration_slope']=calibration(y,p,w)
 return out

def resample_weights(df,rng,source):
 if source=='eicu':
  labs,inv=np.unique(df.hospital.to_numpy(),return_inverse=True)
  n=len(labs);return np.bincount(rng.integers(0,n,n),minlength=n)[inv].astype(float)
 n=len(df);return np.bincount(rng.integers(0,n,n),minlength=n).astype(float)

def threshold_stats(y,p,t,w=None):
 if w is None:w=np.ones(len(y))
 f=p>=t;tp=np.sum(w*y*f);fn=np.sum(w*y*~f);fp=np.sum(w*(1-y)*f);tn=np.sum(w*(1-y)*~f)
 div=lambda a,b:float(a/b) if b else np.nan
 return {'sensitivity':div(tp,tp+fn),'specificity':div(tn,tn+fp),'ppv':div(tp,tp+fp),'npv':div(tn,tn+fn),
  'fraction_flagged':div(tp+fp,np.sum(w)),'missed_events':float(fn),'low_risk_event_rate':div(fn,fn+tn),
  'net_benefit':float((tp-fp*t/(1-t))/np.sum(w))}

def develop():
 d=pd.read_parquet(P/'mimic_features.parquet');d=d[d.followup_observed].reset_index(drop=True)
 y=d.event.to_numpy(dtype=int)
 if y.sum()<200:raise RuntimeError('Locked development event-count gate failed')
 params={};preds=d[['event','hospital']].copy()
 splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(d,y))
 for name,features in SPECS.items():
  oof=np.full(len(d),np.nan)
  for tr,te in splits:oof[te]=predict(d.iloc[te],fit(d.iloc[tr],y[tr],features))
  preds[name]=oof;params[name]=fit(d,y,features)
  print('DEVELOPMENT_OOF',name,metrics(y,oof),flush=True)
 (O/'locked_models.json').write_text(json.dumps(params,indent=2),encoding='utf8')
 preds.to_parquet(P/'mimic_oof_predictions.parquet',index=False)
 lock={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'stage':'development_only_before_any_external_performance','n':len(d),'events':int(y.sum()),
  'model_sha256':hashlib.sha256((O/'locked_models.json').read_bytes()).hexdigest(),
  'development_features_sha256':hashlib.sha256((P/'mimic_features.parquet').read_bytes()).hexdigest(),
  'protocol_sha256':hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest(),
  'source_correction_sha256':hashlib.sha256((A/'source_corrections_before_model.md').read_bytes()).hexdigest(),
  'python':platform.python_version(),'sklearn':sklearn.__version__}
 (A/'model_freeze.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
 print('MODEL_FROZEN',lock['frozen_at_utc'],flush=True)

def validate():
 lock=json.loads((A/'model_freeze.json').read_text());assert lock['model_sha256']==hashlib.sha256((O/'locked_models.json').read_bytes()).hexdigest()
 params=json.loads((O/'locked_models.json').read_text())
 result=[];diffs=[];thr=[];cals=[];dca=[];sens=[];base=[];flow=[];selection=[];site=[]
 for source in ['mimic','eicu','sicdb']:
  all_d=pd.read_parquet(P/(source+'_features.parquet'))
  qa=json.loads((A/(source+'_extraction.json')).read_text())
  for stage,n in qa['flow'].items():flow.append({'source':source,'stage':stage,'n':n})
  d=all_d[all_d.followup_observed].reset_index(drop=True);y=d.event.to_numpy(dtype=int)
  scores={n:predict(d,s) for n,s in params.items()}
  if source=='mimic':
   pp=pd.read_parquet(P/'mimic_oof_predictions.parquet');assert np.array_equal(pp.event,y)
   scores={n:pp[n].to_numpy() for n in params}
  pd.DataFrame({'event':y,**scores}).to_parquet(P/(source+'_predictions.parquet'),index=False)
  for var in FEATURES+['feed_hours','index_hours','phosphate_sample_hours','followup_tests','first_followup_hours','last_followup_hours','observed_followup_hours']:
   a=d[var].dropna();base.append({'source':source,'variable':var,'n':len(d),'available':len(a),'missing':int(d[var].isna().sum()),
    'median':float(a.median()),'q25':float(a.quantile(.25)),'q75':float(a.quantile(.75))})
  for var in ['female','event','severe_event','full_48h_observation','daily_surveillance','named_formula','state_zero','potassium_from_arterial']:
   a=d.sex.eq('F') if var=='female' else d[var] if var in d else pd.Series(False,index=d.index)
   base.append({'source':source,'variable':var,'n':len(d),'available':len(d),'count':int(a.sum()),'proportion':float(a.mean())})
  for f in FEATURES:
   for seen in [True,False]:
    a=all_d.loc[all_d.followup_observed.eq(seen),f]
    selection.append({'source':source,'variable':f,'followup_observed':seen,'n':len(a),'available':int(a.notna().sum()),
     'median':float(a.median()),'q25':float(a.quantile(.25)),'q75':float(a.quantile(.75))})
  point={n:metrics(y,p) for n,p in scores.items()};boot={n:[] for n in scores}
  tboot={t:[] for t in [.05,.1,.2]};rng=np.random.default_rng(SEED)
  for bi in range(500):
   w=resample_weights(d,rng,source)
   if w@y==0 or w@(1-y)==0:continue
   for n,p in scores.items():boot[n].append(metrics(y,p,w))
   for t in tboot:tboot[t].append(threshold_stats(y,scores['six_variable'],t,w))
  for n,p in scores.items():
   bb=pd.DataFrame(boot[n])
   for metric,v in point[n].items():
    result.append({'source':source,'model':n,'evaluation':'out_of_fold' if source=='mimic' else 'frozen_external',
     'n':len(d),'events':int(y.sum()),'metric':metric,'estimate':v,
     'ci_low':float(bb[metric].quantile(.025)),'ci_high':float(bb[metric].quantile(.975)),'bootstrap_valid':len(bb)})
   bins=pd.qcut(p,q=10,duplicates='drop')
   z=pd.DataFrame({'p':p,'y':y,'bin':bins})
   for group,g in z.groupby('bin',observed=True):cals.append({'source':source,'model':n,'n':len(g),'predicted':g.p.mean(),'observed':g.y.mean(),'events':int(g.y.sum())})
   for t in np.linspace(.05,.3,51):
    st=threshold_stats(y,p,t);dca.append({'source':source,'model':n,'threshold':float(t),'net_benefit':st['net_benefit'],
     'monitor_all':float(y.mean()-(1-y.mean())*t/(1-t)),'monitor_none':0.})
  for cmp in ['phosphate_only','phosphate_creatinine']:
   dfb=pd.DataFrame(boot['six_variable'])-pd.DataFrame(boot[cmp])
   for met in ['auc','brier','average_precision']:
    diffs.append({'source':source,'comparison':'six_variable_minus_'+cmp,'metric':met,
     'estimate':point['six_variable'][met]-point[cmp][met],
     'ci_low':float(dfb[met].quantile(.025)),'ci_high':float(dfb[met].quantile(.975))})
  for t in [.05,.1,.2]:
   st=threshold_stats(y,scores['six_variable'],t);bb=pd.DataFrame(tboot[t])
   st.update(source=source,threshold=t,n=len(d),events=int(y.sum()))
   for met in ['sensitivity','npv','fraction_flagged','net_benefit','low_risk_event_rate']:
    st[met+'_ci_low']=float(bb[met].quantile(.025));st[met+'_ci_high']=float(bb[met].quantile(.975))
   thr.append(st)
  masks={'complete_cases':d[FEATURES].notna().all(axis=1),'daily_surveillance':d.daily_surveillance}
  if source=='eicu':masks['named_formula_only']=d.named_formula
  if source=='sicdb':
   masks['source_state_zero']=d.state_zero;masks['continuous_feed']=d.continuous_feed
   masks['central_potassium_source']=~d.potassium_from_arterial&d.potassium.notna()
  for label,mask in masks.items():
   dd=d[mask];yy=y[mask]
   if len(np.unique(yy))<2:continue
   pp=scores['six_variable'][mask]
   row={'source':source,'analysis':label,'n':len(dd),'events':int(yy.sum()),**metrics(yy,pp)}
   sens.append(row)
  # The buffered analysis uses final frozen models, so development estimates here are apparent.
  dd=d.copy()
  for f in RISK_LABS:dd[f]=dd[f+'_buffer2h']
  dd=dd[dd.phosphate.notna()].copy();yy=dd.event.to_numpy(dtype=int)
  sens.append({'source':source,'analysis':'two_hour_predictor_buffer','n':len(dd),'events':int(yy.sum()),
   **metrics(yy,predict(dd,params['six_variable']))})
  for endpoint in ['event_below_1_5','severe_event','first_test_event']:
   yy=d[endpoint].to_numpy(dtype=int)
   sens.append({'source':source,'analysis':endpoint,'n':len(d),'events':int(yy.sum()),
    'auc':float(roc_auc_score(yy,scores['six_variable'])),'average_precision':float(average_precision_score(yy,scores['six_variable'])),
    'prevalence':float(yy.mean())})
  if source=='eicu':
   for h,g in d.assign(pred=scores['six_variable']).groupby('hospital'):
    if len(g)>=30 and g.event.sum()>=5 and g.event.sum()<len(g):
     site.append({'hospital':int(h),'n':len(g),'events':int(g.event.sum()),**metrics(g.event.to_numpy(),g.pred.to_numpy())})
  print('EVALUATED',source,'n',len(d),'events',int(y.sum()),'primary',point['six_variable'],flush=True)
 for name,rows in [('performance',result),('model_differences',diffs),('thresholds',thr),('calibration_groups',cals),('decision_curves',dca),('sensitivity',sens),('baseline',base),('flow',flow),('followup_selection',selection),('eicu_hospital_performance',site)]:
  pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
 print('ANALYSIS_FINISHED',flush=True)

RISK_LABS=['phosphate','creatinine','magnesium','potassium']
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['develop','validate']);args=a.parse_args()
 develop() if args.stage=='develop' else validate()
