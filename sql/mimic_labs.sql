-- Frozen SQL, exported from scripts/extract.py.
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
select x.stay_id sid,l.labevent_id rid,l.itemid variable,l.valuenum value,
 extract(epoch from (l.charttime-x.t0))/3600 hours,
 extract(epoch from (coalesce(l.storetime,l.charttime)-x.t0))/3600 available_hours,
 l.storetime is null availability_fallback,l.valueuom unit
 from x join mimiciv_hosp.labevents l on l.hadm_id=x.hadm_id
 where x.feedtime is not null and x.outtime>x.t0
 and l.itemid in (50970,50912,50960,50971) and l.valuenum is not null
 and l.charttime>x.t0-interval '24 hours' and l.charttime<=least(x.outtime,x.t0+interval '48 hours');
