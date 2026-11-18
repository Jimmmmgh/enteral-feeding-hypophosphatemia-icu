-- Frozen SQL, exported from scripts/extract.py.
-- %s is a psycopg2 array parameter containing eligible cellpaths from audit/eicu_enteral_mapping_review.csv.
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
select x.patientunitstayid sid,l.labid rid,l.labname variable,l.labresult value,
 (l.labresultoffset-x.t0)/60.0 hours,
 (greatest(l.labresultoffset,coalesce(l.labresultrevisedoffset,l.labresultoffset))-x.t0)/60.0 available_hours,
 l.labresultrevisedoffset is null availability_fallback,l.labmeasurenameinterface unit
 from x join eicu_crd.lab l on l.patientunitstayid=x.patientunitstayid
 where x.feedtime is not null and x.unitdischargeoffset>x.t0
 and l.labname in ('phosphate','creatinine','magnesium','potassium') and l.labresult is not null
 and l.labresultoffset>x.t0-1440 and l.labresultoffset<=least(x.unitdischargeoffset,x.t0+2880);
