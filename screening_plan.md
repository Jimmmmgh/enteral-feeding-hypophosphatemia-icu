# Nutrition study redesign: source-first feasibility

Created 2026-09-28. The prior PNI/discharge manuscript is retained as a backup, not the main publication route.

## Candidate selected for source feasibility

Can routine biochemistry at a fixed reassessment point after first documented enteral feeding stratify the risk of subsequently detected marked hypophosphatemia across independent ICU databases?

Provisional index: 24 hours after the first documented positive enteral-feed administration in the first seven ICU days. This is a reassessment during documented feeding, not a verified first-ever exposure to calories. MIMIC-IV will develop the model; eICU-CRD and SICdb will independently evaluate it if source semantics and follow-up capacity permit. No hospital outcome or outcome-predictor associations are used to select the question.

Source gates before model development: actual positive feeding records (not orders or care plans); consistent time units; admission and within-person deduplication; a contemporary phosphate result and subsequent testing; interpretable units; usable independent sites. Inspect lab timing and counts before outcome values. Target at least 1000 eligible development patients and 500 patients in each external source. These are feasibility targets, not a formal sample-size calculation. Event counts and predictor completeness will subsequently determine model complexity; no p-value-based variable selection.

## Alternatives screened out

- PNI/GNRI with mortality or disposition: close precedents, limited incremental information, and extensive selection from missing biomarkers; previous PNI result is preserved in its own directory.
- Joint nutritional-risk/fluid-balance trajectories: directly overlapping 2026 Frontiers in Nutrition dual-database article, doi:10.3389/fnut.2026.1881675.
- Phosphate-based refeeding syndrome definition comparison: directly overlapping MIMIC-IV/eICU dual-cohort study and code deposit, doi:10.5281/zenodo.20948017.
- Non-nutritional calorie/protein accounting: clinically relevant but product composition and delivered protein are not yet harmonized; not selected merely because administration labels exist.

## Scope boundaries

Hypophosphatemia does not establish refeeding syndrome, malnutrition, a treatment effect, or preventable harm. A risk model is not a proven monitoring strategy. Do not claim that an unrecorded feed was absent, or that first charted volume is actual feeding initiation. No external upload, registration, submission, publication, or author transmission is authorized by this local redesign.

## Next gate

After source capacity audit, lock a complete protocol and model comparisons before opening the outcome values. Retain failed source gates and all prespecified comparisons. Do not promote a model or a threshold merely because it yields a more favorable result.
