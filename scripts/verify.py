"""Separate numerical implementation plus source-level aggregate reconstruction."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'audit';R=ROOT/'results';P=ROOT/'private'
sys.path.insert(0,str(ROOT.parent/'icu_nutrition_disposition_20260928'/'scripts'))
from db import connection,query
checks=[]
def check(name,ok,details=None):
 checks.append({'check':name,'passed':bool(ok),'details':details})
 if not ok:print('FAIL',name,details,flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def numeric():
 freeze=json.loads((A/'protocol_freeze.json').read_text())
 for f,h in freeze['files'].items():check('original_freeze_'+f,sha(ROOT/f)==h)
 lock=json.loads((A/'model_freeze.json').read_text())
 check('model_hash_frozen',sha(R/'locked_models.json')==lock['model_sha256'])
 check('development_file_unchanged',sha(P/'mimic_features.parquet')==lock['development_features_sha256'])
 params=json.loads((R/'locked_models.json').read_text());perf=pd.read_csv(R/'performance.csv')
 for src in ['mimic','eicu','sicdb']:
  a=pd.read_parquet(P/(src+'_features.parquet'));d=a[a.followup_observed].reset_index(drop=True)
  pr=pd.read_parquet(P/(src+'_predictions.parquet'));y=d.event.to_numpy()
  check(src+'_endpoint_equals_nadir',np.array_equal(y,d.phosphate_nadir.lt(2).astype(int)))
  check(src+'_baseline_no_low',d.pre_phosphate_min.ge(2).all())
  check(src+'_predictors_at_or_before_index',d.filter(regex='_sample_hours$').max().le(0).all())
  check(src+'_future_specimens_strictly_positive',d.first_followup_hours.gt(0).all())
  check(src+'_followup_not_after_48_or_discharge',(d.last_followup_hours<=d.observed_followup_hours+1e-8).all())
  check(src+'_no_patient_source_identifiers',not set(a.columns)&set(['sid','pid','subject_id','hadm_id','stay_id','CaseID','PatientID','patientunitstayid','uniquepid']))
  check(src+'_unknown_followup_not_negative',a.loc[~a.followup_observed,'event'].isna().all())
  check(src+'_positive_and_negative_events',y.sum()>=100 and (1-y).sum()>=100)
  for model in params:
   p=pr[model].to_numpy();n1=y.sum();n0=len(y)-n1
   auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
   row=perf[(perf.source==src)&(perf.model==model)].set_index('metric').estimate
   check(src+'_'+model+'_auc_independent_rank',abs(auc-row['auc'])<1e-12)
   check(src+'_'+model+'_brier_independent',abs(np.mean((y-p)**2)-row['brier'])<1e-12)
   cal=sm.GLM(y,sm.add_constant(logit(p)),family=sm.families.Binomial()).fit()
   check(src+'_'+model+'_calibration_independent',abs(cal.params[1]-row['calibration_slope'])<1e-7)
   check(src+'_'+model+'_valid_probabilities',np.isfinite(p).all() and ((p>0)&(p<1)).all())
   if src!='mimic':
    s=params[model];v={}
    for f in s['features']:
     x=d[f].fillna(s['medians'][f]).to_numpy(dtype=float)
     if f=='phosphate':
      a1,b,c=s['knots'];h=lambda knot:np.clip(x-knot,0,None)**3
      v['phosphate']=x;v['phosphate_rcs']=(h(a1)-h(b)*(c-a1)/(c-b)+h(c)*(b-a1)/(c-b))/(c-a1)**2
     elif f=='creatinine':v['log_creatinine']=np.log(x)
     else:v[f]=x
     if f in s['missing_flags']:v[f+'_missing']=d[f].isna().to_numpy(dtype=float)
    raw=np.column_stack([v[c] for c in s['columns']])
    coef=np.array(s['coef'])/np.array(s['scale'])
    intercept=s['intercept']-np.array(s['mean'])@coef
    check(src+'_'+model+'_raw_equation_prediction',np.max(np.abs(expit(raw@coef+intercept)-p))<1e-12)
 # Sparse severe outcomes are not represented as a developed severe-event model.
 check('severe_event_model_not_fitted',set(params)=={'phosphate_only','phosphate_creatinine','six_variable'})

def source_sql():
 c=connection('mimic')
 sql="""
 with person as (select distinct on(i.subject_id) i.*,p.anchor_age+extract(year from i.intime)-p.anchor_year age
 from mimiciv_icu.icustays i join mimiciv_hosp.patients p using(subject_id) order by i.subject_id,i.intime,i.stay_id),
 feeding as (select distinct on(i.stay_id) i.stay_id,i.starttime,i.storetime
 from mimiciv_icu.inputevents i join mimiciv_icu.d_items z using(itemid) join person a on a.stay_id=i.stay_id
 where a.age>=18 and z.category='Nutrition - Enteral' and i.ordercategoryname='13-Enteral Nutrition'
 and i.amount>0 and i.amountuom='mL' and i.statusdescription<>'Rewritten'
 and i.starttime between a.intime and a.intime+interval '7 days'
 order by i.stay_id,i.starttime,i.storetime,i.endtime,i.orderid),
 idx as materialized (select a.stay_id,a.hadm_id,a.outtime,f.starttime+interval '24 hours' t0
 from person a join feeding f using(stay_id) where a.outtime>f.starttime+interval '24 hours'
 and coalesce(f.storetime,f.starttime)<=f.starttime+interval '24 hours'),
 labs as materialized (select i.*,l.charttime,l.storetime,l.labevent_id,l.valuenum,
 row_number() over(partition by i.stay_id,l.charttime order by l.storetime desc,l.labevent_id desc) last_version
 from idx i join mimiciv_hosp.labevents l using(hadm_id)
 where l.itemid=50970 and lower(l.valueuom)='mg/dl' and l.valuenum between .1 and 20
 and l.charttime>i.t0-interval '24 hours' and l.charttime<=least(i.outtime,i.t0+interval '48 hours')),
 summary as (select stay_id,
 min(valuenum) filter(where charttime<=t0 and coalesce(storetime,charttime)<=t0) ppre,
 min(valuenum) filter(where charttime>t0 and last_version=1) ppost
 from labs group by stay_id)
 select count(*) n,count(*) filter(where ppost<2) events from summary where ppre>=2 and ppost is not null
 """
 q=query(c,sql).iloc[0];c.close()
 check('independent_mimic_sql_cohort_and_events',int(q['n'])==5622 and int(q['events'])==836,{k:int(v) for k,v in q.items()})
 c=connection('eicu');mp=pd.read_csv(A/'eicu_enteral_mapping_review.csv');paths=mp.loc[mp.eligible_feed_label,'cellpath'].tolist()
 sql="""
 with person as (select distinct on(uniquepid) * from eicu_crd.patient where uniquepid<>'' and uniquepid is not null order by uniquepid,patientunitstayid),
 feeding as (select distinct on(i.patientunitstayid) i.patientunitstayid,i.intakeoutputoffset f,i.intakeoutputentryoffset fe
 from eicu_crd.intakeoutput i join person a using(patientunitstayid)
 where (a.age='> 89' or (a.age ~ '^[0-9]+$' and a.age::integer>=18))
 and i.cellpath=any(%s) and i.cellvaluenumeric>0 and i.intakeoutputoffset between 0 and 10080
 order by i.patientunitstayid,i.intakeoutputoffset,i.intakeoutputentryoffset,i.intakeoutputid),
 idx as materialized (select a.patientunitstayid,a.unitdischargeoffset,f.f+1440 t0 from person a join feeding f using(patientunitstayid)
 where a.unitdischargeoffset>f.f+1440 and coalesce(f.fe,f.f)<=f.f+1440),
 labs as materialized (select i.*,l.labresultoffset t,l.labresultrevisedoffset avail,l.labresult p,
 row_number() over(partition by i.patientunitstayid,l.labresultoffset order by greatest(l.labresultoffset,coalesce(l.labresultrevisedoffset,l.labresultoffset)) desc,l.labid desc) version
 from idx i join eicu_crd.lab l using(patientunitstayid) where l.labname='phosphate' and lower(l.labmeasurenameinterface)='mg/dl'
 and l.labresult between .1 and 20 and l.labresultoffset>i.t0-1440 and l.labresultoffset<=least(i.unitdischargeoffset,i.t0+2880)),
 summary as (select patientunitstayid,min(p) filter(where t<=t0 and coalesce(avail,t)<=t0) pre,
 min(p) filter(where t>t0 and version=1) post from labs group by patientunitstayid)
 select count(*) n,count(*) filter(where post<2) events from summary where pre>=2 and post is not null
 """
 q=query(c,sql,(paths,)).iloc[0];c.close()
 check('independent_eicu_sql_cohort_and_events',int(q['n'])==879 and int(q['events'])==148,{k:int(v) for k,v in q.items()})

def main():
 numeric();source_sql()
 out={'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),
 'reviewer':'same-agent separate numerical/source implementation; not independent clinical validation'}
 (A/'scientific_qa.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 print('QA',out['passed'],'/',out['total'],flush=True)
 if out['passed']!=out['total']:raise SystemExit(1)

if __name__=='__main__':main()
