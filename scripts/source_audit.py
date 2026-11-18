"""Aggregate source/timing audit. No post-index laboratory values or patient IDs saved."""
from pathlib import Path
import sys,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'icu_nutrition_disposition_20260928'/'scripts'))
from db import connection,query
A=ROOT/'audit'; A.mkdir(exist_ok=True,parents=True)

def save(name,d):
    d.to_csv(A/(name+'.csv'),index=False)
    print(name, 'rows',len(d),flush=True)

def main():
    c=connection('mimic')
    save('mimic_enteral_dictionary',query(c,"""
    select d.itemid,d.label,d.category,i.amountuom,i.ordercategoryname,i.secondaryordercategoryname,
    count(*) records,count(distinct i.stay_id) stays
    from mimiciv_icu.d_items d join mimiciv_icu.inputevents i using(itemid)
    where d.category in ('Nutrition - Enteral','Nutrition - Parenteral')
    and i.amount>0 and i.statusdescription<>'Rewritten'
    group by 1,2,3,4,5,6 order by records desc
    """));c.close()
    c=connection('eicu')
    save('eicu_enteral_dictionary',query(c,"""
    select cellpath,celllabel,count(*) records,count(distinct patientunitstayid) stays
    from eicu_crd.intakeoutput
    where lower(cellpath) ~ 'nutri|enteral|tube feed|jevity|nutren|glucerna|nepro|promote|replete|peptamen|isosource|osmolite|fibersource|pivot|novasource|pulmocare|perative|tpn|parenteral'
    group by 1,2 order by records desc
    """));
    save('eicu_relevant_columns',query(c,"""select table_name,column_name,data_type
    from information_schema.columns where table_schema='eicu_crd'
    and table_name in ('patient','lab','intakeoutput','infusiondrug') order by 1,ordinal_position"""));c.close()
    p=Path(r'E:\Physionet\SICdb\1.0.8')
    r=pd.read_csv(p/'d_references.csv.gz')
    save('sicdb_source_dictionary',r[r.ReferenceName.isin(['Fluid','Food','FoodKcal','GastricTube','DrugGivenState','Sex','HospitalUnit','DischargeState'])])
    m=pd.read_csv(p/'medication.csv.gz',usecols=['CaseID','DrugID','Offset','OffsetDrugEnd','IsSingleDose','Amount','GivenState'])
    enids=[2067,2068,2069,2107,2108,2109,2110,2111,2112,2113,2114,2115,2116,2117,2120,2130,2150,2163]
    z=m[m.DrugID.isin(enids)&m.Amount.gt(0)]
    save('sicdb_enteral_record_capacity',z.groupby(['DrugID','GivenState','IsSingleDose']).agg(records=('CaseID','size'),cases=('CaseID','nunique'),positive_duration=('OffsetDrugEnd',lambda s:int((s>z.loc[s.index,'Offset']).sum()))).reset_index())
    labs=pd.read_csv(p/'laboratory.csv.gz',usecols=['CaseID','LaboratoryID','Offset'])
    save('sicdb_lab_capacity',labs[labs.LaboratoryID.isin([367,368,463,468,471])].groupby('LaboratoryID').agg(records=('CaseID','size'),cases=('CaseID','nunique')).reset_index())
    ranges=pd.read_csv(p/'data_range.csv.gz')
    save('sicdb_tube_capacity',ranges[ranges.DataID.isin([3050,3051,3052])].groupby('DataID').agg(records=('CaseID','size'),cases=('CaseID','nunique')).reset_index())
    print('SOURCE_AUDIT_FINISHED',flush=True)

if __name__=='__main__': main()
