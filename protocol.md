# Locked analysis protocol: phosphate risk during enteral feeding

Version 1, 2026-09-28. This local protocol follows a source/timing feasibility audit, before outcome concentrations, outcome rates, or predictor–outcome associations are examined. This is not a prospective registry entry. Source counts are in audit/time_window_capacity.json. The source-only targets were met: MIMIC-IV 6635, eICU-CRD 1170 from 45 hospitals, SICdb 1390 with a contemporaneously recorded gastrointestinal feeding tube. Final counts will be lower after valid-unit, current-phosphate, and timing rules.

## Question and clinical scope

Among adults receiving recorded enteral feeding who have no marked hypophosphatemia during the preceding 24 hours, can a six-variable biochemical model at a 24-hour feeding reassessment predict subsequently detected marked hypophosphatemia over the next 48 hours? Does it add useful information beyond phosphate alone or phosphate plus creatinine when transported unchanged to independent US multicenter and Austrian cohorts?

This is a prognostic/benchmarking study. It does not estimate nutrition effects, establish refeeding syndrome, diagnose malnutrition, or test whether changing monitoring or phosphate supplementation improves outcomes. First documented feeding need not be the actual first exposure to calories. The external cohorts are independently collected datasets, not proof of prospective deployment.

## Cohorts and time

One admission per person is chosen before inspecting nutrition or outcomes. For MIMIC-IV, use the earliest ICU stay by intime/stay_id. For eICU, use the lowest patientunitstayid per uniquepid; do not imply exact cross-hospital chronology. For SICdb, use OffsetAfterFirstAdmission then CaseID. Adults are age >=18. eICU >89 is represented by 91; SICdb age, height and weight are released in bins and are used as supplied. Sex is descriptive, not an eligibility criterion or model predictor.

The qualifying feed is the earliest positive recorded administration in the first 7 ICU days. MIMIC-IV requires a mL record in both Nutrition - Enteral and 13-Enteral Nutrition, excludes Rewritten and isolated supplements not in mL. eICU requires the exact reviewed intake path in audit/eicu_enteral_mapping_review.csv, positive recorded volume, and explicit tube-feed/formula terminology; flushes, water, medicines, oral intake, totals, and unspecified tube intake are excluded. SICdb requires a prespecified enteral formula (including explicitly unspecified enteral solution), positive amount, a positive-duration interval, and a gastric/jejunal/PEG tube documented as present at that time. Oral supplements and parenteral nutrition are excluded as qualifying feeds.

Index time is 24 hours after that record. Patients must remain in the ICU/case at index. SICdb medication/lab offsets are seconds from the case origin; ICUOffset is added to establish the first seven ICU days. MIMIC timestamps use hours; eICU offsets use minutes. Feed documentation must be recorded by index where a separate entry/store time exists. Because SICdb's public medication table is described as administered treatment but GivenState values lack a semantic mapping beyond State_0 through State_3, no state is relabelled as completed/cancelled. The primary analysis uses positive recorded administrations; restrictive State_0 and continuous-feed sensitivities will be labelled source sensitivities, not clinically validated adjudication.

## Predictors and laboratory harmonization

Use the latest valid result collected in (index-24 h, index], provided the available store/revision timestamp is not later than index in MIMIC/eICU. Missing store/revision timestamps fall back to specimen/result offset and are counted. SICdb has no separate result-availability timestamp; this limitation is disclosed, with a two-hour lookback buffer sensitivity in all cohorts. In MIMIC use central laboratory item IDs, not duplicated bedside chart representations. If multiple valid results share a sampling time, choose the most recently stored/revised value available by index; conflicting same-time source values are counted.

Units: phosphate and creatinine mg/dL, magnesium mg/dL, potassium mmol/L. SICdb phosphate mmol/L is multiplied by 3.097; magnesium mmol/L by 2.4305. eICU magnesium mEq/L is multiplied by 1.21525. Potassium mEq/L equals mmol/L; ambiguous mEq/mL is excluded. Age is years and BMI kg/m2. Use admission weight/height for eICU and SICdb; MIMIC uses measurements recorded no later than index (first-day derived anthropometry is acceptable because index is >=24 h after ICU admission).

Broad valid ranges, specified before reading outcomes: phosphate 0.1–20 mg/dL, creatinine 0.1–30 mg/dL, magnesium 0.3–10 mg/dL, potassium 1–10 mmol/L, height 120–220 cm, weight 25–300 kg, BMI 10–70. Values outside these ranges become missing, with counts reported. Phosphate is mandatory and is never imputed. At least one valid contemporary phosphate and no valid phosphate <2.0 mg/dL in the preceding 24 hours are required. Other predictor missingness is reported by source. No albumin, lymphocyte count or complete nutritional-score panel is required.

Six predefined predictors: latest phosphate, latest creatinine, latest magnesium, latest potassium, age, BMI. Phosphate uses a restricted cubic spline with three development-cohort knots at the 10th/50th/90th percentiles (two degrees of freedom); creatinine is log-transformed; the other predictors enter linearly. No interaction or univariable p-value selection. Phosphate and creatinine are prespecified as nested comparator models; the six-variable model remains the primary candidate regardless of which has the highest performance.

## Outcome and monitoring

Primary outcome: at least one valid phosphate <2.0 mg/dL in (index, index+48 h], truncated at ICU/case end. This is detected marked hypophosphatemia under observed care. Severe hypophosphatemia <1.0 mg/dL is secondary and evaluated with the primary-model ranking rather than a separately optimized rare-event model. A patient with no follow-up phosphate is unascertained and excluded from performance calculations, never counted as a negative. Discharge/death before 48 h truncates ascertainment and is reported; this study does not infer that these patients would remain event-free if retained in ICU.

Report the complete source flow, testing frequency, follow-up span, and covariate differences between those with and without follow-up. Repeat performance in patients remaining in the ICU for the entire 48 h with at least one phosphate measurement in each 24 h block. This is a surveillance sensitivity, not a correction of informative testing. Also report the first scheduled/observed post-index phosphate result and time without selecting the most extreme result for that sensitivity.

## Development and external evaluation

MIMIC-IV develops all preprocessing and coefficients. Ridge logistic regression uses a fixed C=1 with standardized design columns. Other than phosphate, missing values are replaced by development medians with development-defined missingness indicators; constant indicators are omitted. Transformations, knots, medians, scaling, coefficients and intercept are frozen before any external model performance is calculated. External values and medians never train, tune or recalibrate the primary model.

Use five-fold stratified out-of-fold validation in MIMIC, seed 20260928, repeating preprocessing within each training fold. Fit the final model in the full development cohort. If development events are fewer than 200, do not claim definitive model development; deliver a feasibility report without using external outcomes to choose a new model. External cohorts with fewer than 100 events are labelled imprecise/exploratory; do not pool them to disguise that limitation.

Report AUROC, average precision (with outcome prevalence), Brier score, observed/expected events, calibration intercept and slope, and calibration plots. Confidence intervals use 500 bootstrap resamples: people in MIMIC/SICdb and hospitals in eICU; eICU samples preserve all patients within sampled hospitals. Paired resampling estimates differences between nested models. Bootstrap calibration intervals are diagnostic, not proof of calibration equivalence. The development out-of-fold evaluation reflects internal performance; fixed-model external bootstraps condition on the fitted development model.

At prespecified predicted-risk thresholds of 5%, 10%, and 20%, report sensitivity, specificity, PPV, NPV, fraction flagged, and missed events; do not optimize a cutoff on external data. A decision curve over 5–30% is exploratory and concerns hypothetical selection for additional surveillance, not proven health benefit or a recommendation to omit routine monitoring. If net benefit is absent or calibration poor, retain those findings in the main report. No model deployment or bedside app will be supplied as validated care.

## Prespecified sensitivity and subtraction checks

1. Phosphate alone and phosphate + creatinine versus the six-variable model (core subtraction test).
2. Complete predictor cases, keeping the frozen coefficients.
3. Full 48-hour ICU/case observation and phosphate measured in both daily blocks.
4. Predictor specimens limited to at least two hours before index, with the same 24-hour lookback and frozen coefficients; retain the original at-risk cohort and report secondary availability.
5. Outcome threshold <1.5 mg/dL and severe <1.0 mg/dL, reporting ranking and event capture, not treating primary predicted probabilities as calibrated probabilities of these different outcomes.
6. eICU named-formula-only versus generic tube-feed paths; hospital heterogeneity at sites with >=30 patients and >=5 events, purely descriptive.
7. SICdb first qualifying feeding record in State_0; and continuous feeds (IsSingleDose=0). These assess recording sensitivity without interpreting undocumented state semantics.
8. Endpoint based on the first post-index phosphate rather than the 48-hour minimum, retaining frozen-model discrimination and documenting changed estimand.

No post hoc mortality, causal treatment, machine-learning algorithm competition, score-derived-label prediction, or biomarker ratio is appended to manufacture novelty. Additional analyses required by a discovered source error must be dated and separated from this protocol.

## Outputs and reporting

TRIPOD-compatible transparent manuscript and supplement if estimable; otherwise a clear study report describing the failed gates. Include reproducible extraction/model code, exact model parameters, source mapping, sample flow, all comparisons, figures, aggregate tables, and technical QA. No source patient identifiers, protected clinical records or authentication material are included in deliverables. Journal interest and acceptance remain unverified outcomes, not products of technical QA.
