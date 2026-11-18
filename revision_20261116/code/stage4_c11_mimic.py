"""Exploratory within-MIMIC-only expansion with strictly pre-index event windows."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stage4_link import OUT, PROJECT, extractor, replay
from stage4_coverage import MIMIC, query_target
from stage4_analysis import evaluate, SEED


def fields(module, cohort):
    c = module.connection('mimic')
    try:
        result = query_target(module,c,MIMIC + """
        , rrt as materialized (
          select distinct x.sid
          from x join mimiciv_icu.procedureevents p on p.stay_id=x.sid
          where p.itemid in (225441,225802,225803,225809,225955)
            and p.starttime<=x.landmark and p.endtime>x.landmark and p.storetime<=x.landmark
            and p.statusdescription is distinct from 'Rewritten'
        ), scores as materialized (
          select distinct on(x.sid) x.sid,s.sofa_24hours
          from x join mimiciv_derived.sofa s on s.stay_id=x.sid
          where s.endtime<=x.landmark and s.endtime>x.landmark-interval '2 hours'
          order by x.sid,s.endtime desc
        )
        select x.sid,scores.sofa_24hours as sofa, cast(rrt.sid is not null as int) as rrt_record_available_at_index
        from x left join scores using(sid) left join rrt using(sid)
        """,cohort)
        return result.set_index('sid').reindex(cohort.sid).reset_index(drop=True)
    finally:
        c.close()


def run():
    m = extractor()
    all_eligible,_ = replay(m,'mimic')
    d = all_eligible.loc[all_eligible.followup_observed].reset_index(drop=True)
    d = pd.concat([d,fields(m,d)],axis=1)
    y = d.event.to_numpy(dtype=int)
    take = d.sofa.notna().to_numpy()
    rrt = d.rrt_record_available_at_index.to_numpy(dtype=int)
    take_count = int(take.sum())
    active_count = int(rrt[take].sum())
    meta = {'attempted': True,'target':'mimic_internal_only','n':len(d),'n_verified_sofa':take_count,
            'n_documented_active_rrt_with_record_by_index':active_count,
            'small_cell':active_count<5,'external_status':'HOLD_nonexchangeable_severity_and_index_RRT'}
    (OUT/'c11_mimic_feasibility.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    if min(active_count,take_count-active_count)<5:
        print('C11 MIMIC HOLD: small or empty index-available RRT cell',flush=True)
        return
    original = runpy.run_path(str(PROJECT/'scripts'/'analyze.py'),run_name='frozen_analysis_pure_functions')
    cols = original['FEATURES']
    extended = cols+['sofa','rrt_record_available_at_index']
    models = {'six_variable_refit_same_sample':cols,'expanded_sofa_index_documented_rrt':extended}
    p = pd.DataFrame(np.nan,index=d.index,columns=models,dtype=float)
    split = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    cv = list(split.split(d,y))
    if any(np.unique(rrt[tr[take[tr]]]).size<2 for tr,_ in cv):
        print('C11 MIMIC HOLD: deficient RRT support in training fold',flush=True)
        return
    for tr,te in cv:
        tr,te=tr[take[tr]],te[take[te]]
        for model, spec in models.items():
            fit = original['fit'](d.iloc[tr],y[tr],spec)
            p.loc[te,model] = original['predict'](d.iloc[te],fit)
    fitted = {model:original['fit'](d.loc[take],y[take],spec) for model,spec in models.items()}
    path = OUT/'c11_mimic_fitted_models.json'
    path.write_text(json.dumps(fitted,indent=2)+'\n',encoding='utf-8')
    if p.loc[take].isna().any().any():
        raise RuntimeError('C11 incomplete out-of-fold predictions')
    result=evaluate('mimic','C11','verified_sofa_procedure_record_at_index_OOF_exploratory',
                    d.loc[take],p.loc[take],tuple(models))
    pd.DataFrame(result).to_csv(OUT/'c11_mimic_model_metrics.csv',index=False)
    print('C11 MIMIC-only exploratory complete n',take_count,'documented active',active_count,flush=True)


if __name__ == '__main__':
    run()
