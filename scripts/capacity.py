"""Time-window capacity without reading phosphate concentrations or outcomes."""
from pathlib import Path
import sys,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'icu_nutrition_disposition_20260928'/'scripts'))
from db import connection,query
A=ROOT/'audit'

def selected_eicu():
    d=pd.read_csv(A/'eicu_enteral_dictionary.csv').fillna('')
    allowed=[]
    for row in d.itertuples():
        label=row.celllabel.strip().lower()
        yes=('|Intake (ml)|' in row.cellpath and not re.search(r'flush|water|h2o|h20|meds?|resid|parenter|tpn|total|drain|output',label)
            and (re.search(r'tube\s*feed|tubefeed|enteral formula volume',label)
            or (label.startswith('enteral ') and re.search(r'nepro|pivot|jevity|nutren|glucerna|perative|fibersource|vital|osmol|replete|novasource|twocal|two\s*cal|2\s*cal|peptamen|isosource|impact|pulmocare|promote|vivonex',label))))
        allowed.append(bool(yes))
    d['eligible_feed_label']=allowed
    d.to_csv(A/'eicu_enteral_mapping_review.csv',index=False)
    labels=d.loc[d.eligible_feed_label,'cellpath'].tolist()
    return labels

def mimic():
    c=connection('mimic')
    d=query(c,"""
    with ranked as (select i.*,row_number() over(partition by subject_id order by intime,stay_id) rn from mimiciv_icu.icustays i),
    b as materialized (select r.*,p.anchor_age+extract(year from r.intime)-p.anchor_year age
      from ranked r join mimiciv_hosp.patients p using(subject_id) where rn=1),
    en as (select i.stay_id,min(i.starttime) feedtime from mimiciv_icu.inputevents i
      join mimiciv_icu.d_items di using(itemid) join b on b.stay_id=i.stay_id
      where di.category='Nutrition - Enteral' and i.ordercategoryname='13-Enteral Nutrition'
      and i.amount>0 and i.amountuom='mL' and i.statusdescription<>'Rewritten'
      and i.starttime>=b.intime and i.starttime<=b.intime+interval '7 days'
      group by i.stay_id),
    x as materialized (select b.*,en.feedtime,en.feedtime+interval '24 hours' t0 from b join en using(stay_id)
      where b.age>=18 and b.outtime>en.feedtime+interval '24 hours'),
    flags as (select x.stay_id,
      bool_or(l.charttime>x.t0-interval '24 hours' and l.charttime<=x.t0 and coalesce(l.storetime,l.charttime)<=x.t0) baseline,
      bool_or(l.charttime>x.t0 and l.charttime<=least(x.outtime,x.t0+interval '48 hours')) followup
      from x left join mimiciv_hosp.labevents l on l.hadm_id=x.hadm_id and l.itemid=50970
      and l.charttime>x.t0-interval '24 hours' and l.charttime<=least(x.outtime,x.t0+interval '48 hours') group by x.stay_id)
    select count(*) as adult_feed_landmark, count(*) filter(where baseline) baseline_available,
    count(*) filter(where baseline and followup) baseline_and_followup from flags
    """)
    c.close();return d.to_dict('records')[0]

def eicu(paths):
    c=connection('eicu')
    d=query(c,"""
    with ranked as (select p.*,row_number() over(partition by uniquepid order by patientunitstayid) rn,
      case when age='> 89' then 91 when age ~ '^[0-9]+$' then age::integer end age_num
      from eicu_crd.patient p where uniquepid is not null and uniquepid<>''),
    b as materialized (select * from ranked where rn=1 and age_num>=18),
    en as (select i.patientunitstayid,min(i.intakeoutputoffset) feedtime
      from eicu_crd.intakeoutput i join b using(patientunitstayid)
      where i.cellpath=ANY(%s) and i.cellvaluenumeric>0 and i.intakeoutputoffset between 0 and 10080 group by 1),
    x as materialized (select b.*,en.feedtime,en.feedtime+1440 t0 from b join en using(patientunitstayid)
      where b.unitdischargeoffset>en.feedtime+1440),
    flags as (select x.patientunitstayid,x.hospitalid,
      bool_or(l.labresultoffset>x.t0-1440 and l.labresultoffset<=x.t0 and coalesce(l.labresultrevisedoffset,l.labresultoffset)<=x.t0) baseline,
      bool_or(l.labresultoffset>x.t0 and l.labresultoffset<=least(x.unitdischargeoffset,x.t0+2880)) followup
      from x left join eicu_crd.lab l on l.patientunitstayid=x.patientunitstayid and l.labname='phosphate'
      and l.labresultoffset>x.t0-1440 and l.labresultoffset<=least(x.unitdischargeoffset,x.t0+2880)
      group by x.patientunitstayid,x.hospitalid)
    select count(*) adult_feed_landmark,count(*) filter(where baseline) baseline_available,
      count(*) filter(where baseline and followup) baseline_and_followup,
      count(distinct hospitalid) filter(where baseline and followup) hospitals from flags
    """,(paths,))
    c.close();return d.to_dict('records')[0]

def sicdb():
    p=Path(r'E:\Physionet\SICdb\1.0.8')
    b=pd.read_csv(p/'cases.csv.gz')
    b=b.sort_values(['PatientID','OffsetAfterFirstAdmission','CaseID']).drop_duplicates('PatientID')
    b=b[b.AgeOnAdmission.ge(18)]
    m=pd.read_csv(p/'medication.csv.gz',usecols=['CaseID','DrugID','Offset','OffsetDrugEnd','IsSingleDose','Amount','GivenState'])
    ids=[2067,2068,2069,2107,2108,2109,2110,2111,2112,2113,2114,2115,2116,2117,2120,2130,2150,2163]
    m=m[m.DrugID.isin(ids)&m.Amount.gt(0)&m.OffsetDrugEnd.gt(m.Offset)]
    m=m.merge(b[['CaseID','ICUOffset']],on='CaseID')
    m=m[m.Offset.ge(m.ICUOffset)&m.Offset.le(m.ICUOffset+7*86400)]
    first=m.sort_values(['CaseID','Offset']).drop_duplicates('CaseID')[['CaseID','Offset','GivenState']]
    first=first.rename(columns={'Offset':'feedtime'})
    x=b.merge(first,on='CaseID');x['t0']=x.feedtime+86400
    x=x[x.TimeOfStay.gt(x.t0)]
    tubes=pd.read_csv(p/'data_range.csv.gz');tubes=tubes[tubes.DataID.isin([3050,3051,3052])]
    t=tubes.merge(x[['CaseID','feedtime']],on='CaseID')
    active=set(t.loc[t.Offset.le(t.feedtime)&t.OffsetEnd.ge(t.feedtime),'CaseID'])
    x['tube_confirmed']=x.CaseID.isin(active)
    l=pd.read_csv(p/'laboratory.csv.gz',usecols=['CaseID','LaboratoryID','Offset'])
    l=l[l.LaboratoryID.eq(471)].merge(x[['CaseID','t0','TimeOfStay']],on='CaseID')
    base=set(l.loc[l.Offset.gt(l.t0-86400)&l.Offset.le(l.t0),'CaseID'])
    follow=set(l.loc[l.Offset.gt(l.t0)&l.Offset.le(np.minimum(l.TimeOfStay,l.t0+172800)),'CaseID'])
    return {'adult_feed_landmark':len(x),'baseline_available':len(set(x.CaseID)&base),
     'baseline_and_followup':len(set(x.CaseID)&base&follow),
     'baseline_and_followup_with_tube':len(set(x.loc[x.tube_confirmed,'CaseID'])&base&follow),
     'given_state_not_semantically_resolved':True}

if __name__=='__main__':
    import concurrent.futures
    paths=selected_eicu()
    output={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut={ex.submit(mimic):'MIMIC-IV',ex.submit(eicu,paths):'eICU-CRD',ex.submit(sicdb):'SICdb'}
        for f in concurrent.futures.as_completed(fut):
            output[fut[f]]=f.result();print(fut[f],output[fut[f]],flush=True)
    (A/'time_window_capacity.json').write_text(json.dumps(output,indent=2),encoding='utf8')
