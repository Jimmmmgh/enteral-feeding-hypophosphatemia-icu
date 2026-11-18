"""Extract harmonized features in memory; persist no source participant identifiers."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'icu_nutrition_disposition_20260928'/'scripts'))
from db import connection,query
A=ROOT/'audit';P=ROOT/'private';P.mkdir(exist_ok=True)
LABS={50970:'phosphate',50912:'creatinine',50960:'magnesium',50971:'potassium'}
RANGES={'phosphate':(.1,20),'creatinine':(.1,30),'magnesium':(.3,10),'potassium':(1,10)}

MBASE="""
with ranked as (select i.*,row_number() over(partition by subject_id order by intime,stay_id) rn
 from mimiciv_icu.icustays i),
b as materialized (select r.*,p.anchor_age+extract(year from r.intime)-p.anchor_year age,
 p.gender sex,p.anchor_year_group era from ranked r join mimiciv_hosp.patients p using(subject_id)
 where rn=1 and p.anchor_age+extract(year from r.intime)-p.anchor_year>=18),
enrank as (select i.stay_id,i.starttime,i.storetime,i.rate,i.amount,i.endtime,
 row_number() over(partition by i.stay_id order by i.starttime,i.storetime,i.endtime,i.orderid) ern
 from mimiciv_icu.inputevents i join mimiciv_icu.d_items di using(itemid) join b on b.stay_id=i.stay_id
 where di.category='Nutrition - Enteral' and i.ordercategoryname='13-Enteral Nutrition'
 and i.amount>0 and i.amountuom='mL' and i.statusdescription<>'Rewritten'
 and i.starttime>=b.intime and i.starttime<=b.intime+interval '7 days'),
x as materialized (select b.*,e.starttime feedtime,e.storetime feedstore,e.rate feedrate,
 e.starttime+interval '24 hours' t0 from b left join enrank e on e.stay_id=b.stay_id and e.ern=1)
"""
EBASE="""
with ranked as (select p.*,row_number() over(partition by uniquepid order by patientunitstayid) rn,
 case when age='> 89' then 91 when age ~ '^[0-9]+$' then age::integer end age_num
 from eicu_crd.patient p where uniquepid is not null and uniquepid<>''),
b as materialized (select * from ranked where rn=1 and age_num>=18),
enrank as (select i.*,row_number() over(partition by i.patientunitstayid
 order by i.intakeoutputoffset,i.intakeoutputentryoffset,i.intakeoutputid) ern
 from eicu_crd.intakeoutput i join b using(patientunitstayid)
 where i.cellpath=ANY(%s) and i.cellvaluenumeric>0 and i.intakeoutputoffset between 0 and 10080),
x as materialized (select b.*,e.intakeoutputoffset feedtime,e.intakeoutputentryoffset feedstore,
 e.celllabel feedlabel,e.intakeoutputoffset+1440 t0 from b left join enrank e
 on e.patientunitstayid=b.patientunitstayid and e.ern=1)
"""

def source_mimic():
 c=connection('mimic')
 b=query(c,MBASE+"""select x.stay_id sid,x.age,x.sex,x.era,x.first_careunit unit,
 extract(epoch from (x.outtime-x.intime))/3600 end_hours,
 extract(epoch from (x.feedtime-x.intime))/3600 feed_hours,
 extract(epoch from (x.feedstore-x.intime))/3600 feed_entry_hours,
 h.height,w.weight_admit weight,x.feedrate from x
 left join mimiciv_derived.first_day_height h using(stay_id)
 left join mimiciv_derived.first_day_weight w using(stay_id)""")
 l=query(c,MBASE+"""select x.stay_id sid,l.labevent_id rid,l.itemid variable,l.valuenum value,
 extract(epoch from (l.charttime-x.t0))/3600 hours,
 extract(epoch from (coalesce(l.storetime,l.charttime)-x.t0))/3600 available_hours,
 l.storetime is null availability_fallback,l.valueuom unit
 from x join mimiciv_hosp.labevents l on l.hadm_id=x.hadm_id
 where x.feedtime is not null and x.outtime>x.t0
 and l.itemid in (50970,50912,50960,50971) and l.valuenum is not null
 and l.charttime>x.t0-interval '24 hours' and l.charttime<=least(x.outtime,x.t0+interval '48 hours')""")
 c.close();l['variable']=l.variable.map(LABS)
 b['hospital']=0;b['tube_confirmed']=True;b['named_formula']=True
 b['state_zero']=True;b['continuous_feed']=b.feedrate.gt(0)
 return b,l

def source_eicu():
 d=pd.read_csv(A/'eicu_enteral_mapping_review.csv')
 paths=d.loc[d.eligible_feed_label,'cellpath'].tolist()
 c=connection('eicu')
 b=query(c,EBASE+"""select patientunitstayid sid,age_num age,gender sex,
 hospitaldischargeyear::text era,unittype unit,hospitalid hospital,
 unitdischargeoffset/60.0 end_hours,feedtime/60.0 feed_hours,
 feedstore/60.0 feed_entry_hours,admissionheight height,admissionweight weight,feedlabel from x""",(paths,))
 l=query(c,EBASE+"""select x.patientunitstayid sid,l.labid rid,l.labname variable,l.labresult value,
 (l.labresultoffset-x.t0)/60.0 hours,
 (greatest(l.labresultoffset,coalesce(l.labresultrevisedoffset,l.labresultoffset))-x.t0)/60.0 available_hours,
 l.labresultrevisedoffset is null availability_fallback,l.labmeasurenameinterface unit
 from x join eicu_crd.lab l on l.patientunitstayid=x.patientunitstayid
 where x.feedtime is not null and x.unitdischargeoffset>x.t0
 and l.labname in ('phosphate','creatinine','magnesium','potassium') and l.labresult is not null
 and l.labresultoffset>x.t0-1440 and l.labresultoffset<=least(x.unitdischargeoffset,x.t0+2880)""",(paths,))
 c.close();b['sex']=b.sex.map({'Male':'M','Female':'F'}).fillna('Unknown')
 b['tube_confirmed']=True;b['state_zero']=True;b['continuous_feed']=False
 b['named_formula']=b.feedlabel.fillna('').str.contains('nepro|pivot|jevity|nutren|glucerna|perative|fibersource|vital|osmol|replete|novasource|twocal|two\\s*cal|2\\s*cal|peptamen|isosource|impact|pulmocare|promote|vivonex',case=False,regex=True)
 return b.drop(columns=['feedlabel']),l

def source_sicdb():
 p=Path(r'E:\Physionet\SICdb\1.0.8')
 r=pd.read_csv(p/'d_references.csv.gz').set_index('ReferenceGlobalID')
 b=pd.read_csv(p/'cases.csv.gz')
 b=b.sort_values(['PatientID','OffsetAfterFirstAdmission','CaseID']).drop_duplicates('PatientID')
 b=b[b.AgeOnAdmission.ge(18)]
 m=pd.read_csv(p/'medication.csv.gz',usecols=['id','CaseID','DrugID','Offset','OffsetDrugEnd','IsSingleDose','Amount','GivenState'])
 ids=[2067,2068,2069,2107,2108,2109,2110,2111,2112,2113,2114,2115,2116,2117,2120,2130,2150,2163]
 m=m[m.DrugID.isin(ids)&m.Amount.gt(0)&m.OffsetDrugEnd.gt(m.Offset)]
 m=m.merge(b[['CaseID','ICUOffset']],on='CaseID')
 m=m[m.Offset.ge(m.ICUOffset)&m.Offset.le(m.ICUOffset+7*86400)]
 first=m.sort_values(['CaseID','Offset','id']).drop_duplicates('CaseID')[['CaseID','Offset','GivenState','IsSingleDose']]
 first=first.rename(columns={'Offset':'feedtime'})
 b=b.merge(first,on='CaseID',how='left');b['t0']=b.feedtime+86400
 tubes=pd.read_csv(p/'data_range.csv.gz');tubes=tubes[tubes.DataID.isin([3050,3051,3052])]
 t=tubes.merge(b[['CaseID','feedtime']],on='CaseID')
 active=set(t.loc[t.Offset.le(t.feedtime)&t.OffsetEnd.ge(t.feedtime),'CaseID'])
 b['tube_confirmed']=b.CaseID.isin(active)
 l=pd.read_csv(p/'laboratory.csv.gz')
 l=l[l.LaboratoryID.isin([367,368,463,468,471,685])].merge(b[['CaseID','t0','TimeOfStay']],on='CaseID')
 l=l[l.Offset.gt(l.t0-86400)&l.Offset.le(np.minimum(l.TimeOfStay,l.t0+172800))]
 l['hours']=(l.Offset-l.t0)/3600;l['available_hours']=l.hours
 l['availability_fallback']=True
 l['unit']=l.LaboratoryID.map(r.ReferenceUnit)
 l['variable']=l.LaboratoryID.map({367:'creatinine',368:'creatinine',463:'potassium',468:'magnesium',471:'phosphate',685:'potassium'})
 l['potassium_arterial']=l.LaboratoryID.eq(685)
 l=l.rename(columns={'CaseID':'sid','id':'rid','LaboratoryValue':'value'})
 l=l[['sid','rid','variable','value','hours','available_hours','availability_fallback','unit','potassium_arterial']]
 b['end_hours']=(b.TimeOfStay-b.ICUOffset)/3600
 b['feed_hours']=(b.feedtime-b.ICUOffset)/3600
 b['feed_entry_hours']=b.feed_hours
 b['sex']=b.Sex.map({735:'M',736:'F'}).fillna('Unknown')
 b['unit']=b.HospitalUnit.map(r.ReferenceValue)
 b['hospital']=0;b['named_formula']=True
 b['state_zero']=b.GivenState.eq(2193);b['continuous_feed']=b.IsSingleDose.eq(0)
 b=b.rename(columns={'CaseID':'sid','AgeOnAdmission':'age','WeightOnAdmission':'weight','HeightOnAdmission':'height','AdmissionYear':'era'})
 b['weight']=b.weight/1000.0
 b=b[['sid','age','sex','era','unit','hospital','end_hours','feed_hours','feed_entry_hours','height','weight','tube_confirmed','named_formula','state_zero','continuous_feed']]
 return b,l

def process(name,b,l):
 b=b.copy();l=l.copy();qa={'source':name,'flow':{}}
 for col in ['age','end_hours','feed_hours','feed_entry_hours','height','weight']:
  b[col]=pd.to_numeric(b[col],errors='coerce')
 def step(label,mask=None):
  nonlocal b
  if mask is not None:b=b.loc[mask].copy()
  qa['flow'][label]=int(len(b))
 step('one_adult_admission_per_person')
 step('positive_feed_first_7_icu_days',b.feed_hours.notna())
 b['index_hours']=b.feed_hours+24
 step('remaining_in_icu_at_index',b.end_hours.gt(b.index_hours+1e-9))
 step('feed_documented_by_index',b.feed_entry_hours.fillna(b.feed_hours).le(b.index_hours+1e-9))
 step('feeding_tube_documented_if_required',b.tube_confirmed)
 l=l[l.sid.isin(b.sid)].copy()
 for col in ['value','hours','available_hours']:l[col]=pd.to_numeric(l[col],errors='coerce')
 l['normunit']=l.unit.fillna('').str.strip().str.lower()
 counts=l.groupby(['variable','normunit']).size().rename('rows').reset_index()
 counts.to_csv(A/(name+'_lab_units.csv'),index=False)
 units_ok=pd.Series(False,index=l.index)
 for var in RANGES:
  idx=l.variable.eq(var)
  if var in ['phosphate','creatinine','magnesium']:
   units_ok|=idx&l.normunit.eq('mg/dl')
  if var=='phosphate':
   m=idx&l.normunit.eq('mmol/l');l.loc[m,'value']*=3.097;units_ok|=m
  if var=='magnesium':
   m=idx&l.normunit.eq('mmol/l');l.loc[m,'value']*=2.4305;units_ok|=m
   m=idx&l.normunit.eq('meq/l');l.loc[m,'value']*=1.21525;units_ok|=m
  if var=='potassium':units_ok|=idx&l.normunit.isin(['mmol/l','meq/l','mmoll'])
 qa['invalid_units']=int((~units_ok).sum());l=l[units_ok].copy()
 valid=pd.Series(False,index=l.index)
 for var,(lo,hi) in RANGES.items():valid|=l.variable.eq(var)&l.value.between(lo,hi)
 qa['invalid_ranges']=l.loc[~valid].groupby('variable').size().to_dict();l=l[valid].copy()
 qa['source_lab_rows']=int(len(l));qa['availability_fallback_rows']=int(l.availability_fallback.sum())
 qa['negative_result_lag_rows']=int((l.available_hours<l.hours).sum())
 l['available_hours']=np.maximum(l.available_hours,l.hours)
 dup=l.groupby(['sid','variable','hours']).value.nunique()
 qa['conflicting_same_specimen_time_groups']=int(dup.gt(1).sum())
 l=l.sort_values(['sid','variable','hours','available_hours','rid'])
 baseline=l[l.hours.le(0)&l.available_hours.le(0)&l.hours.gt(-24)].copy()
 # Available results alone determine the baseline risk set.
 last=baseline.drop_duplicates(['sid','variable'],keep='last')
 if 'potassium_arterial' in last:
  ka=last[last.variable.eq('potassium')].set_index('sid').potassium_arterial
  b['potassium_from_arterial']=b.sid.map(ka).fillna(False).astype(bool)
 else:b['potassium_from_arterial']=False
 wide=last.pivot(index='sid',columns='variable',values='value')
 times=last.pivot(index='sid',columns='variable',values='hours').add_suffix('_sample_hours')
 b=b.merge(wide,on='sid',how='left').merge(times,on='sid',how='left')
 minphos=baseline[baseline.variable.eq('phosphate')].groupby('sid').value.min()
 b['pre_phosphate_min']=b.sid.map(minphos)
 for var in RANGES:
  if var not in b:b[var]=np.nan
 step('valid_available_baseline_phosphate',b.phosphate.notna())
 step('no_known_phosphate_below_2_in_previous_24h',b.pre_phosphate_min.ge(2))
 for col,lo,hi in [('height',120,220),('weight',25,300)]:
  b[col]=pd.to_numeric(b[col],errors='coerce');b.loc[~b[col].between(lo,hi),col]=np.nan
 b['bmi']=b.weight/(b.height/100)**2;b.loc[~b.bmi.between(10,70),'bmi']=np.nan
 buffered=baseline[baseline.hours.le(-2)].drop_duplicates(['sid','variable'],keep='last')
 bw=buffered.pivot(index='sid',columns='variable',values='value').add_suffix('_buffer2h')
 b=b.merge(bw,on='sid',how='left')
 post=l[l.hours.gt(0)&l.hours.le(48)&l.variable.eq('phosphate')&l.sid.isin(b.sid)].copy()
 # Collapse repeated interfaces at the same specimen time to one final revised value.
 post=post.drop_duplicates(['sid','hours'],keep='last')
 summary=post.groupby('sid').agg(phosphate_nadir=('value','min'),followup_tests=('value','size'),
  first_followup_hours=('hours','min'),last_followup_hours=('hours','max'))
 first=post.sort_values(['sid','hours']).drop_duplicates('sid').set_index('sid').value.rename('first_followup_phosphate')
 summary=summary.join(first)
 day1=set(post.loc[post.hours.le(24),'sid']);day2=set(post.loc[post.hours.gt(24),'sid'])
 b=b.merge(summary,on='sid',how='left');b['followup_observed']=b.followup_tests.notna()
 b['observed_followup_hours']=np.minimum(48,b.end_hours-b.index_hours)
 b['full_48h_observation']=b.observed_followup_hours.ge(48)
 b['daily_surveillance']=b.full_48h_observation&b.sid.isin(day1&day2)
 b['event']=np.where(b.followup_observed,b.phosphate_nadir.lt(2).astype(int),np.nan)
 b['event_below_1_5']=np.where(b.followup_observed,b.phosphate_nadir.lt(1.5).astype(int),np.nan)
 b['severe_event']=np.where(b.followup_observed,b.phosphate_nadir.lt(1).astype(int),np.nan)
 b['first_test_event']=np.where(b.followup_observed,b.first_followup_phosphate.lt(2).astype(int),np.nan)
 qa['flow']['followup_phosphate_observed']=int(b.followup_observed.sum())
 qa['flow']['primary_events']=int(b.event.sum())
 qa['flow']['severe_events']=int(b.severe_event.sum())
 qa['flow']['full_48h_daily_surveillance']=int(b.daily_surveillance.sum())
 qa['hospitals_observed']=int(b.loc[b.followup_observed,'hospital'].nunique())
 qa['predictor_missing_in_observed']=b.loc[b.followup_observed,['phosphate','creatinine','magnesium','potassium','age','bmi']].isna().mean().to_dict()
 qa['all_baseline_specimens_not_after_index']=bool((b.filter(regex='_sample_hours$').max().fillna(0)<=0).all())
 # No source IDs, absolute times, source row identifiers or free text enter the derived file.
 b=b.drop(columns=['sid','feedrate'],errors='ignore').reset_index(drop=True)
 assert not set(['sid','pid','subject_id','hadm_id','stay_id','patientunitstayid','uniquepid','PatientID','CaseID','rid'])&set(b.columns)
 b.to_parquet(P/(name+'_features.parquet'),index=False)
 qa['feature_file_sha256']=hashlib.sha256((P/(name+'_features.parquet')).read_bytes()).hexdigest()
 (A/(name+'_extraction.json')).write_text(json.dumps(qa,indent=2,default=str),encoding='utf8')
 print(name,json.dumps(qa,default=str),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('source',choices=['mimic','eicu','sicdb']);arg=parser.parse_args()
 b,l={'mimic':source_mimic,'eicu':source_eicu,'sicdb':source_sicdb}[arg.source]()
 process(arg.source,b,l)
