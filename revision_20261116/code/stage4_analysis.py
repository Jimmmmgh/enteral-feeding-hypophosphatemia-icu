"""Post-checkpoint 3 exploratory/sensitivity analyses on the exactly replayed cohort."""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from stage4_link import OUT, PROJECT
from stage4_coverage import run, SOURCES

MODELS = ('phosphate_only','phosphate_creatinine','six_variable')
METRICS = ('auroc','brier','oe','cal_intercept','cal_slope')
SEED = 20260928
BOOT = 500


def quantiles(x):
    return {'q1':x.quantile(.25), 'median':x.median(), 'q3':x.quantile(.75)}


def save(source, stem, rows):
    path = OUT / f'{stem}_{source}.csv'
    data = pd.DataFrame(rows)
    data.to_csv(path, index=False, encoding='utf-8')


def c5(prelab):
    groups = []
    for observed, sub in prelab.groupby('baseline_phosphate', sort=False):
        valid_sex = sub.sex.isin(['M','F'])
        row = {
            'baseline_phosphate_measured': bool(observed),
            'n': len(sub),
            'n_sites': sub.hospital.nunique(),
            'female_n': sub.sex.eq('F').sum(),
            'male_n': sub.sex.eq('M').sum(),
            'other_or_unknown_sex_n': (~valid_sex).sum(),
            'diagnosis_preindex_coverage_n': sub.dx_doc.sum(),
            'admission_diagnosis_preindex_coverage_n': sub.admission_dx_doc.sum(),
            'sepsis_documented_n': sub.sepsis_documented.sum(),
            'dka_documented_n': sub.dka_documented.sum(),
            'operative_admission_documented_n': sub.surgery_documented.sum(),
            'apache_iv_available_n': sub.severity_available.sum(),
        }
        for label, values in [('age',sub.age), ('apache_iv',sub.severity)]:
            row.update({f'{label}_{k}':v for k,v in quantiles(values).items()})
        row['small_cells'] = '; '.join(k for k,v in row.items() if k.endswith('_n') and 0 <= v < 5)
        groups.append(row)
    site = prelab.groupby('hospital').baseline_phosphate.agg(['size','sum']).rename(columns={'size':'n','sum':'with_baseline_phosphate'})
    site['without_baseline_phosphate'] = site.n - site.with_baseline_phosphate
    site['fraction_without_baseline_phosphate'] = site.without_baseline_phosphate.div(site.n)
    site['fraction_all_missing_cases'] = site.without_baseline_phosphate.div(site.without_baseline_phosphate.sum())
    site = site.sort_values(['without_baseline_phosphate','n','with_baseline_phosphate'],ascending=False).reset_index(drop=True)
    site.insert(0,'anonymous_site_rank', np.arange(1,len(site)+1))
    site['small_cell'] = site[['n','with_baseline_phosphate','without_baseline_phosphate']].lt(5).any(axis=1)
    site_summary = pd.DataFrame([{
        'n_sites': len(site), 'sites_with_any_missing': site.without_baseline_phosphate.gt(0).sum(),
        'fraction_missing_in_top_5_sites': site.fraction_all_missing_cases.head(5).sum(),
        'fraction_missing_in_top_10_sites': site.fraction_all_missing_cases.head(10).sum(),
        'fraction_missing_by_site_median': site.fraction_without_baseline_phosphate.median(),
        'fraction_missing_by_site_q1': site.fraction_without_baseline_phosphate.quantile(.25),
        'fraction_missing_by_site_q3': site.fraction_without_baseline_phosphate.quantile(.75),
    }])
    pd.DataFrame(groups).to_csv(OUT/'c5_eicu_baseline.csv',index=False)
    site.to_csv(OUT/'c5_eicu_hospital.csv',index=False)
    site_summary.to_csv(OUT/'c5_eicu_hospital_summary.csv',index=False)


def c6(source, cohort, flags):
    records = []
    for label, selected in (
        ('all_eligible', pd.Series(True,index=cohort.index)),
        ('followup_tested', cohort.followup_observed),
        ('followup_untested', ~cohort.followup_observed),
    ):
        elapsed = cohort.loc[selected,'observed_followup_hours']
        truncated = elapsed.lt(48)
        documented_death = flags.loc[selected,'death_by_end'].fillna(False).astype(bool)
        known = flags.loc[selected,'end_status_known'].fillna(False).astype(bool)
        reasons = {
            'truncated_death_documented': int((truncated & documented_death).sum()),
            'truncated_icu_exit_no_death_documented': int((truncated & ~documented_death & known).sum()) if source != 'sicdb' else 0,
            'truncated_metavision_case_closure_no_death_documented': int((truncated & ~documented_death).sum()) if source == 'sicdb' else 0,
            'truncated_status_unknown': int((truncated & ~documented_death & ~known).sum()) if source != 'sicdb' else 0,
            'case_closure_unknown_icu_discharge_status_subset': int((truncated & ~documented_death & ~known).sum()) if source == 'sicdb' else 0,
        }
        if sum(list(reasons.values())[:4]) != truncated.sum():
            raise RuntimeError(f'{source} mismatched observation causes')
        row = {'source':source,'cohort':label,'n':len(elapsed),'n_truncated':int(truncated.sum()),**reasons}
        row.update({f'elapsed_hours_{k}':v for k,v in quantiles(elapsed).items()})
        row['minimum_hours'] = elapsed.min()
        row['maximum_hours'] = elapsed.max()
        row['small_cells'] = '; '.join(k for k,v in reasons.items() if 0 <= v < 5)
        records.append(row)
    save(source,'c6_observation',records)


def calibration(y, z, w):
    """Fit logistic calibration by Newton iteration, returning NaN for separation or singularity."""
    x = np.column_stack([np.ones(len(z)),z])
    alpha = 0.0
    b = np.array([0.,1.])
    slope = np.nan
    intercept = np.nan
    for _ in range(30):
        q = expit(z + alpha)
        h = np.sum(w*q*(1-q))
        if h < 1e-10:
            break
        step = np.sum(w*(y-q))/h
        alpha += step
        if abs(step) < 1e-8:
            intercept = alpha
            break
    for _ in range(30):
        q = expit(x @ b)
        h = x.T @ ((w*q*(1-q))[:,None] * x)
        if np.linalg.det(h) < 1e-10:
            break
        bstep = np.linalg.solve(h, x.T @ (w*(y-q)))
        b += bstep
        if np.max(np.abs(bstep)) < 1e-8:
            slope = b[1] if abs(b[1]) < 100 else np.nan
            break
    return intercept, slope


def metrics(y, p, z, w, with_cal):
    count_events = np.dot(w,y)
    if count_events <= 0 or count_events >= w.sum():
        return np.repeat(np.nan,len(METRICS))
    vals = [roc_auc_score(y,p,sample_weight=w),np.average((y-p)**2,weights=w),count_events/np.dot(w,p)]
    return np.array(vals+list(calibration(y,z,w) if with_cal else [np.nan,np.nan]))


def evaluate(source, analysis, subgroup, cohort, predictions, models=MODELS):
    """Use a shared sequence of resamples across model pairs; retain every metric's valid count."""
    y = cohort.event.to_numpy(dtype=int)
    n = len(y)
    n_ev = int(y.sum())
    if n == 0 or min(n_ev,n-n_ev) == 0:
        return [{'source':source,'analysis':analysis,'subgroup':subgroup,'model':model,'n':n,
                 'events':n_ev,'small_cell':True,'status':'not_estimable_no_both_classes'} for model in models]
    p = predictions.loc[:, list(models)].to_numpy(dtype=float)
    z = logit(np.clip(p, 1e-8, 1-1e-8))
    n_mod = p.shape[1]
    units = np.unique(cohort.hospital,return_inverse=True)[1] if source == 'eicu' else np.arange(n)
    m = np.unique(units).size
    rng = np.random.default_rng(SEED)
    cal = min(n_ev,n-n_ev)>=5
    point = [metrics(y,p[:,k],z[:,k],np.ones(n),cal) for k in range(n_mod)]
    draws = []
    for _ in range(BOOT):
        wk = np.bincount(rng.integers(m,size=m),minlength=m)[units].astype(float)
        draws.append([metrics(y,p[:,k],z[:,k],wk,cal) for k in range(n_mod)])
    draws = np.array(draws)
    outputs = []
    for k,model in enumerate(models):
        row = {'source':source,'analysis':analysis,'subgroup':subgroup,'model':model,'n':n,'events':n_ev,
               'small_cell':min(n_ev,n-n_ev)<5,'status':'computed'}
        for j,met in enumerate(METRICS):
            valid = draws[:,k,j]
            valid = valid[np.isfinite(valid)]
            low, high = np.quantile(valid,[.025,.975]) if len(valid)>=100 else (np.nan,np.nan)
            if np.isfinite(point[k][j]):
                row.update({met:point[k][j],met+'_lo':low,met+'_hi':high,met+'_boot_n':len(valid)})
        outputs.append(row)
    if analysis == 'C11' and n_mod==2:
        delta = draws[:,1,:] - draws[:,0,:]
        row = {'source':source,'analysis':analysis,'subgroup':subgroup,'model':'expanded_minus_six','n':n,'events':n_ev,
               'small_cell':min(n_ev,n-n_ev)<5,'status':'computed'}
        for j,met in enumerate(METRICS[:2]):
            valid = delta[:,j][np.isfinite(delta[:,j])]
            low, high = np.quantile(valid,[.025,.975]) if len(valid)>=100 else (np.nan,np.nan)
            row.update({met:point[1][j]-point[0][j],met+'_lo':low,met+'_hi':high,met+'_boot_n':len(valid)})
        outputs.append(row)
    return outputs


def c7(source, cohort, flags, saved):
    labels = {'mimic':'documented_inputevents_or_eMAR_administration',
              'eicu':'noncancelled_medication_order_proxy','sicdb':'candidate_medication_record_state_unverified'}
    n = len(cohort)
    row = {'source':source,'n':n,'scope':labels[source]}
    for col in flags.columns[flags.columns.str.startswith('phos_')]:
        row[col+'_n'] = int(flags[col].sum())
        row[col+'_fraction'] = float(flags[col].mean())
    row['small_cells'] = '; '.join(k for k,v in row.items() if k.endswith('_n') and 0<=v<5)
    save(source,'c7_supplement',[row])
    if source=='sicdb':
        return [{'source':source,'analysis':'C7','subgroup':'not_evaluated','model':model,'n':len(cohort),
                 'status':'HOLD_no_identified_candidate_phosphate_records_and_state_unverified'}
                for model in MODELS]
    covered = flags.any_pre_med_orders if source=='eicu' else pd.Series(True,index=flags.index)
    if source=='eicu':
        out = evaluate(source,'C7','preindex_medication_order_interface_documented',cohort.loc[covered],saved.loc[covered])
    else:
        out = []
    scope = 'covered_no_documented_phosphate_order' if source=='eicu' else 'no_documented_preindex_administration'
    take = covered & ~flags.phos_pre_any
    return out + evaluate(source,'C7',scope,cohort.loc[take],saved.loc[take])


def c8(source, cohort, flags, saved):
    row = {'source':source,'n':len(cohort)}
    if source=='mimic':
        row.update(status='interval_based_documentation', n_at_index=int(flags.rrt_index_doc.sum()),
                   n_timed_status_unknown=np.nan, n_chart_active_prior_6h=int(flags.rrt_chart_pre6.sum()))
    elif source=='eicu':
        row.update(status='no_verifiable_index_state_from_treatment_entry', n_timed_status_unknown=len(cohort),
                   n_treatment_entries_prior_24h=int(flags.rrt_chart_pre24.sum()))
    else:
        row.update(status='no_verifiable_combined_index_state', n_timed_status_unknown=len(cohort),
                   n_positive_whole_stay_hours_crrt=int(flags.rrt_any_hours_positive.sum()))
    row['small_cells'] = '; '.join(k for k,v in row.items() if k.startswith('n_') and pd.notna(v) and 0<=v<5)
    save(source,'c8_rrt',[row])
    if source!='mimic':
        return [{'source':source,'analysis':'C8','subgroup':'not_evaluated','model':model,'n':len(cohort),
                 'status':'HOLD_no_verifiable_combined_index_RRT_state'}
                for model in MODELS]
    active = flags.rrt_index_doc.astype(bool)
    return (evaluate(source,'C8','interval_overlaps_index',cohort.loc[active],saved.loc[active]) +
            evaluate(source,'C8','without_documented_index_interval',cohort.loc[~active],saved.loc[~active]))


def run_stage4(source):
    cohort, flags, prelab = run(source)
    if source=='eicu':
        c5(prelab)
    c6(source,cohort,flags)
    primary = cohort.followup_observed
    included = cohort.loc[primary].reset_index(drop=True)
    these_flags = flags.loc[primary].reset_index(drop=True)
    predictions = pd.read_parquet(PROJECT/'private'/f'{source}_predictions.parquet')
    if len(predictions)!=len(included) or not np.array_equal(predictions.event,included.event):
        raise RuntimeError('Stored prediction outcome/order does not align')
    findings = c7(source,included,these_flags,predictions) + c8(source,included,these_flags,predictions)
    save(source,'c7_c8_model_metrics',findings)
    print(source,'C5/C6/C7/C8 scoped results complete',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source',choices=SOURCES)
    args = p.parse_args()
    run_stage4(args.source)
