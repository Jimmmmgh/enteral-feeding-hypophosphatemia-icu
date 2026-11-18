"""Measure revised clinical fields without releasing linked records."""
import argparse
import json

import numpy as np
import pandas as pd

from stage4_link import OUT, SIC, extractor, replay, grouped_phosphate

SOURCES = ('mimic', 'eicu', 'sicdb')
PHOS = r'^(sodium phos|potassium (phos|and sodium phos|sodium phos|& sodium phos)|k[ -]phos|phos[ -]nak|phospha|phosphorus)'
IV = r'^iv(?:$|[ /-]|pb|push)|^intraven'
ENTERAL = r'^(?:po(?:$|[ /-])|oral|per (?:ng|g|peg)|g[ /-]?tube|peg|gastric|nasogastr|dobhoff|ng$|ft$|feeding|enteral)'
RRT = r'^renal\|dialysis\|(hemodialysis|C V V H|C A V H|SLED)'
TARGET = """
with z as (select * from unnest(%s::int[], %s::double precision[], %s::double precision[]) as a(sid, t0, tend))
"""
MIMIC = TARGET + """
, x as (select z.sid, i.hadm_id, i.intime, i.outtime, i.intime+z.t0*interval '1 hour' as landmark,
        i.intime+z.tend*interval '1 hour' as horizon
        from z join mimiciv_icu.icustays i on i.stay_id=z.sid)
"""


def query_target(module, conn, sql, cohort, extras=()):
    ids = cohort.sid.astype(int).tolist()
    index = cohort.index_hours.astype(float).tolist()
    end = np.minimum(cohort.end_hours, cohort.index_hours + 48).tolist()
    return module.query(conn, sql, (ids, index, end, *extras))


def coverage_columns(cohort, phos, rrt=None, severity=None, insulin=None, dead=None, stop=None):
    out = pd.DataFrame(phos, index=cohort.index)
    out['rrt_index_doc'] = rrt if rrt is not None else pd.NA
    out['severity'] = severity if severity is not None else np.nan
    out['insulin_pre_doc'] = insulin if insulin is not None else pd.NA
    out['death_by_end'] = dead if dead is not None else pd.NA
    out['end_status_known'] = stop if stop is not None else pd.NA
    return out


def routes(s):
    s = s.fillna('').str.strip()
    return np.select([s.str.contains(IV, case=False, regex=True), s.str.contains(ENTERAL, case=False, regex=True)],
                     ['iv', 'enteral'], default='unknown')


def mimic(module, cohort):
    conn = module.connection('mimic')
    try:
        supplements = query_target(module, conn, MIMIC + """
        select x.sid, extract(epoch from (i.starttime-x.intime))/3600.0 as time, 'iv' as route
        from x join mimiciv_icu.inputevents i on i.stay_id=x.sid
        where i.itemid in (225834,225835,225925) and i.amount>0
          and i.statusdescription is distinct from 'Rewritten' and i.statusdescription is distinct from 'Paused'
          and i.starttime>x.landmark-interval '24 hours' and i.starttime<=x.horizon
        union all
        select x.sid, extract(epoch from (e.charttime-x.intime))/3600.0, 'enteral'
        from x join mimiciv_hosp.emar e on e.hadm_id=x.hadm_id
        where e.charttime>x.landmark-interval '24 hours' and e.charttime<=x.horizon
          and e.medication ~* %s and e.event_txt in ('Administered','Delayed Administered')
          and exists (select 1 from mimiciv_hosp.emar_detail p
                      where p.subject_id=e.subject_id and p.emar_id=e.emar_id
                      and (p.route ~* %s or p.route ~* %s))
        """, cohort, (PHOS, r'^(PO$|ORAL|PER (NG|G|PEG)|G[ /-]?TUBE|PEG|GASTRIC|NASOGASTR|DOBHOFF|NG$|FT$|FEEDING|ENTERAL)', r'^by mouth|^oral gastric'))
        fields = query_target(module, conn, MIMIC + """
        , rrt_intervals as materialized (
          select distinct x.sid from x join mimiciv_icu.procedureevents p on p.stay_id=x.sid
          where p.itemid in (225441,225802,225803,225809,225955)
            and p.starttime<=x.landmark and p.endtime>x.landmark
            and p.statusdescription is distinct from 'Rewritten'
        ), rrt_notes as materialized (
          select distinct x.sid from x join mimiciv_derived.rrt r on r.stay_id=x.sid
          where r.charttime>x.landmark-interval '6 hours' and r.charttime<=x.landmark
            and r.dialysis_active=1 and r.dialysis_type not in ('None','Peritoneal')
        ), insulin_drips as materialized (
          select distinct x.sid from x join mimiciv_icu.inputevents i on i.stay_id=x.sid
          where i.itemid=223258 and i.ordercategoryname='01-Drips' and i.rate>0 and i.amount>0
            and i.statusdescription is distinct from 'Rewritten' and i.statusdescription is distinct from 'Paused'
            and i.endtime>x.landmark-interval '24 hours' and i.starttime<=x.landmark
        ), sepsis_pre as materialized (
          select distinct x.sid from x join mimiciv_derived.sepsis3 s on s.stay_id=x.sid
          where s.sepsis3 is true and s.sofa_time<=x.landmark and s.suspected_infection_time<=x.landmark
        ), dka_any as materialized (
          select distinct x.sid from x join mimiciv_hosp.diagnoses_icd dx on dx.hadm_id=x.hadm_id
          where (dx.icd_version=10 and dx.icd_code~'^E(08|09|10|11|13)1[01]'
                     or dx.icd_version=9 and dx.icd_code~'^2501[0-3]')
        ), scores as materialized (
          select distinct on (x.sid) x.sid, s.sofa_24hours
          from x join mimiciv_derived.sofa s on s.stay_id=x.sid
          where s.endtime<=x.landmark and s.endtime>x.landmark-interval '2 hours'
          order by x.sid, s.endtime desc
        )
        select x.sid,
          a.deathtime is not null and a.deathtime<=x.outtime as dead,
          a.hadm_id is not null and (a.hospital_expire_flag=0 or a.deathtime is not null) as status_known,
          sf.sofa_24hours as severity, treatment.sid is not null as rrt,
          rnotes.sid is not null as rrt_chart_pre6, i.sid is not null as insulin,
          s.sid is not null as sepsis_documented, d.sid is not null as dka_hospital_only
        from x left join mimiciv_hosp.admissions a on a.hadm_id=x.hadm_id
        left join scores sf using(sid)
        left join rrt_intervals treatment using(sid)
        left join rrt_notes rnotes using(sid)
        left join insulin_drips i using(sid)
        left join sepsis_pre s using(sid)
        left join dka_any d using(sid)
        """, cohort)
        fields = fields.set_index('sid').reindex(cohort.sid)
    finally:
        conn.close()
    flags = grouped_phosphate(supplements, cohort, 1)
    out = coverage_columns(cohort, flags, fields.rrt.to_numpy(), fields.severity.to_numpy(),
                           fields.insulin.to_numpy(), fields.dead.to_numpy(), fields.status_known.to_numpy())
    for c in ['rrt_chart_pre6','sepsis_documented','dka_hospital_only']:
        out[c] = fields[c].to_numpy()
    return out, None


def eicu_base(module, cohort, raw_base, conn):
    """Relink the exact 2,376 source-eligible stays before baseline phosphate exclusion."""
    raw_base = raw_base.copy()
    timing = ['feed_hours','feed_entry_hours','end_hours']
    raw_base[timing] = raw_base[timing].apply(pd.to_numeric, errors='coerce')
    sourcebase = raw_base.loc[raw_base.feed_hours.notna() &
                              raw_base.end_hours.gt(raw_base.feed_hours + 24 + 1e-9) &
                              raw_base.feed_entry_hours.fillna(raw_base.feed_hours).le(raw_base.feed_hours + 24 + 1e-9)].copy()
    sourcebase['index_hours'] = sourcebase.feed_hours + 24
    if len(sourcebase) != 2376:
        raise RuntimeError('eICU baseline-test source eligibility no longer reproduces 2376')
    cov = query_target(module, conn, TARGET + r"""
    , insulin_note as materialized (
        select distinct x.sid
        from z x join eicu_crd.infusiondrug i on i.patientunitstayid=x.sid
        where i.infusionoffset>round(x.t0*60)-1440 and i.infusionoffset<=round(x.t0*60)
          and i.drugname ~* '^insulin|^regular insulin'
          and case when trim(i.drugrate)~'^[0-9]+([.][0-9]+)?$' then trim(i.drugrate)::numeric > 0 else false end
    )
    select x.sid,
     exists (select 1 from eicu_crd.lab l where l.patientunitstayid=x.sid
              and l.labname='phosphate'
              and l.labresultoffset>round(x.t0*60)-1440 and l.labresultoffset<=round(x.t0*60)
              and greatest(l.labresultoffset,coalesce(l.labresultrevisedoffset,l.labresultoffset))<=round(x.t0*60)
              and (lower(trim(l.labmeasurenameinterface))='mg/dl' and l.labresult between 0.1 and 20
                 or lower(trim(l.labmeasurenameinterface))='mmol/l' and l.labresult*3.097 between 0.1 and 20)) as baseline_phosphate,
     target_score.severity,
     p.unitdischargestatus,
     target_score.severity is not null as severity_available,
     exists (select 1 from eicu_crd.admissiondx a where a.patientunitstayid=x.sid and a.admitdxenteredoffset<=round(x.t0*60)) as admission_dx_doc,
     exists (select 1 from eicu_crd.diagnosis d where d.patientunitstayid=x.sid and d.diagnosisoffset<=round(x.t0*60)) as dx_doc,
     exists (select 1 from eicu_crd.diagnosis d where d.patientunitstayid=x.sid and d.diagnosisoffset<=round(x.t0*60)
              and d.diagnosisstring ~* '\m(sepsis|septic shock)\M') as sepsis_documented,
     exists (select 1 from eicu_crd.diagnosis d where d.patientunitstayid=x.sid and d.diagnosisoffset<=round(x.t0*60)
              and d.diagnosisstring ~* '\m(diabetic ketoacidosis|dka)\M') as dka_documented,
     exists (select 1 from eicu_crd.admissiondx a where a.patientunitstayid=x.sid and a.admitdxenteredoffset<=round(x.t0*60)
              and a.admitdxpath ~* '(^|\|)operative(\||$)') as surgery_documented,
     exists (select 1 from eicu_crd.treatment t where t.patientunitstayid=x.sid and t.treatmentoffset>round(x.t0*60)-1440
              and t.treatmentoffset<=round(x.t0*60) and t.treatmentstring ~* %s) as rrt_chart_pre24,
     i.sid is not null as insulin
    from z x join eicu_crd.patient p on p.patientunitstayid=x.sid
    left join insulin_note i on i.sid=x.sid
    left join lateral (select max(a.apachescore) as severity from eicu_crd.apachepatientresult a
                       where a.patientunitstayid=p.patientunitstayid and a.apacheversion='IV' and a.apachescore>=0
                       having count(distinct a.apachescore)=1) target_score on true
    """, sourcebase, (RRT,))
    sourcebase = sourcebase.merge(cov, on='sid', how='inner', validate='one_to_one')
    if len(sourcebase) != 2376 or int(sourcebase.baseline_phosphate.sum()) != 1342:
        raise RuntimeError('eICU baseline phosphate selection does not reproduce 2376 to 1342')
    return sourcebase


def eicu(module, cohort, raw_base):
    conn = module.connection('eicu')
    try:
        prelab = eicu_base(module, cohort, raw_base, conn)
        meds = query_target(module, conn, TARGET + """
        select x.sid, m.drugstartoffset as time, m.routeadmin
        from z x join eicu_crd.medication m on m.patientunitstayid=x.sid
        where m.drugname ~* %s and m.drugordercancelled='No'
          and m.drugstartoffset>round(x.t0*60)-1440 and m.drugstartoffset<=round(x.tend*60)
        """, cohort, (PHOS,))
        meds['route'] = routes(meds.routeadmin)
        rec = query_target(module, conn, TARGET + """
        , orders as materialized (
          select x.sid, bool_or(m.drugstartoffset<=round(x.t0*60)) as preindex
          from z x join eicu_crd.medication m on m.patientunitstayid=x.sid
          where m.drugstartoffset>round(x.t0*60)-1440 and m.drugstartoffset<=round(x.tend*60)
          group by x.sid
        ), infusions as materialized (
          select x.sid, bool_or(f.drugname ~* %s) as has_phosphate
          from z x join eicu_crd.infusiondrug f on f.patientunitstayid=x.sid
          where f.infusionoffset>round(x.t0*60)-1440 and f.infusionoffset<=round(x.tend*60)
          group by x.sid
        )
        select x.sid, m.sid is not null as any_med_orders,
           coalesce(m.preindex,false) as any_pre_med_orders,
           i.sid is not null as any_infusion_chart,
           exists (select 1 from eicu_crd.treatment t where t.patientunitstayid=x.sid
                   and t.treatmentoffset>round(x.t0*60)-1440 and t.treatmentoffset<=round(x.t0*60)) as any_treatment_chart,
           coalesce(i.has_phosphate,false) as any_phos_infusion_chart
        from z x left join orders m on m.sid=x.sid left join infusions i on i.sid=x.sid
        """, cohort, (r'\m(sodium phosphate|potassium phosphate|na phosphate|k phosphate|k phos|na phos|phosphorus)\M',))
    finally:
        conn.close()
    fields = prelab.set_index('sid').reindex(cohort.sid)
    flags = grouped_phosphate(meds, cohort, 60)
    out = coverage_columns(cohort, flags, severity=fields.severity.to_numpy(), insulin=fields.insulin.to_numpy(),
                           dead=fields.unitdischargestatus.eq('Expired').to_numpy(), stop=fields.unitdischargestatus.isin(['Expired','Alive']).to_numpy())
    for c in ['rrt_chart_pre24','sepsis_documented','dka_documented','surgery_documented','admission_dx_doc','dx_doc']:
        out[c] = fields[c].to_numpy()
    for c in rec.columns.drop('sid'):
        out[c] = rec.set_index('sid').reindex(cohort.sid)[c].to_numpy()
    return out, prelab


def sicdb(cohort):
    cases = pd.read_csv(SIC / 'cases.csv.gz', usecols=['CaseID','ICUOffset','TimeOfStay','OffsetOfDeath','DischargeState',
                         'saps3','HoursOfCRRT','ICD10Main','AdmissionFormHasSepsis','SurgicalSite','WeightOnAdmission']).set_index('CaseID')
    s = cases.reindex(cohort.sid).reset_index(drop=True)
    join = cohort[['sid','index_hours','end_hours']].copy()
    join['offset0'] = s.ICUOffset.to_numpy()
    join['index_hours'] = join['index_hours'] + join.offset0/3600
    join['end_hours'] = join['end_hours'] + join.offset0/3600
    medications = pd.read_csv(SIC / 'medication.csv.gz', usecols=['CaseID','DrugID','Offset','OffsetDrugEnd','Amount','IsSingleDose','GivenState'])
    medications = medications[medications.CaseID.isin(cohort.sid) & medications.Amount.gt(0)]
    med = medications[medications.DrugID.isin([1760,2084,2138])].rename(columns={'CaseID':'sid','Offset':'time'}).copy()
    med['route'] = np.where(med.DrugID.isin([2084,2138]), 'iv', 'unknown')
    flags = grouped_phosphate(med, join, 3600)
    s_insulin = medications.loc[medications.DrugID.eq(1557) & medications.IsSingleDose.eq(0)].merge(join, left_on='CaseID', right_on='sid')
    i_active = s_insulin.OffsetDrugEnd.gt((s_insulin.index_hours-24)*3600) & s_insulin.Offset.le(s_insulin.index_hours*3600)
    insulin = cohort.sid.isin(s_insulin.loc[i_active,'sid']).to_numpy()
    death = s.OffsetOfDeath.notna() & s.OffsetOfDeath.le(s.TimeOfStay) | s.DischargeState.eq(2215)
    severity = pd.to_numeric(s.saps3, errors='coerce').where(lambda p: p.gt(0))
    out = coverage_columns(cohort, flags, severity=severity.to_numpy(), insulin=insulin, dead=death.to_numpy(),
                           stop=s.DischargeState.isin([2202,2215]).to_numpy())
    out['rrt_any_hours_available'] = s.HoursOfCRRT.notna().to_numpy()
    out['rrt_any_hours_positive'] = s.HoursOfCRRT.gt(0).to_numpy()
    out['sicdb_admit_sepsis_field_known'] = s.AdmissionFormHasSepsis.notna().to_numpy()
    out['sicdb_surgical_site_field_known'] = s.SurgicalSite.notna().to_numpy()
    out['sicdb_icd10main_known'] = s.ICD10Main.fillna('').str.strip().ne('').to_numpy()
    observed_src = set(cohort.loc[cohort.followup_observed,'sid'])
    state_counts = med.loc[med.sid.isin(observed_src),'GivenState'].value_counts(dropna=False).to_dict()
    meta = {'observed_candidate_phosphate_record_counts_by_state_whole_stay': {str(k): int(v) for k,v in state_counts.items()},
            'raw_weight_quantiles': s.loc[cohort.followup_observed,'WeightOnAdmission'].quantile([.25,.5,.75]).tolist()}
    return out, meta


def run(source):
    m = extractor()
    cohort, raw = replay(m, source)
    if source == 'mimic':
        flags, meta = mimic(m, cohort)
    elif source == 'eicu':
        flags, meta = eicu(m, cohort, raw)
    else:
        flags, meta = sicdb(cohort)
    return cohort, flags, meta


def write_coverage(source, cohort, flags, meta):
    analysis = flags.loc[cohort.followup_observed]
    summary = {'source': source, 'exact_frozen_linkage_verified': True, 'eligible_n': len(cohort),
               'analysed_n': len(analysis), 'fields': {}}
    for c in analysis.drop(columns='death_by_end'):
        d = analysis[c]
        field = {'n_observed': int(d.notna().sum())}
        if c != 'severity':
            field['n_documented_positive'] = int(d.fillna(False).astype(bool).sum()) if d.notna().any() else None
        summary['fields'][c] = field
    if source == 'eicu':
        summary['eicu_prebaseline'] = {
            'n': len(meta), 'n_eligible_phosphate': int(meta.baseline_phosphate.sum()),
            'age_known_n': int(meta.age.notna().sum()), 'sex_male_or_female_n': int(meta.sex.isin(['M','F']).sum()),
            'hospital_known_n': int(meta.hospital.notna().sum()), 'sites_n': int(meta.hospital.nunique()),
            'admission_diagnosis_n': int(meta.admission_dx_doc.sum()), 'preindex_diagnosis_n': int(meta.dx_doc.sum()),
            'apache_iv_valid_n': int(meta.severity_available.sum())}
    elif source == 'sicdb':
        summary['sicdb_source_checks'] = meta
    path = OUT / f'coverage_{source}.json'
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(source, 'coverage only; exact linked sample', len(analysis), 'fields', len(analysis.columns), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', choices=SOURCES)
    args = p.parse_args()
    write_coverage(args.source, *run(args.source))

