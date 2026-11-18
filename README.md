# Parsimonious versus Expanded Routine-Variable Models for Hypophosphatemia During Enteral Feeding

Public reproducibility repository (`https://github.com/Jimmmmgh/enteral-feeding-hypophosphatemia-icu`, also archived as `Supplementary_Reproducibility_Package.zip`) containing extraction SQL specifications, preprocessing and modeling code, dated protocol/freeze records, and aggregate results for:

> *Parsimonious versus expanded routine-variable models for hypophosphatemia in critically ill adults receiving enteral feeding: development and external evaluation in three databases* (Frontiers in Medicine, Manuscript ID: 2026682)

## Scope and Data Governance

- **No patient-level data or credentials:** This package contains zero row-level clinical records, patient identifiers, derived individual features, row-level predictions, or database credentials.
- **Controlled source access:** Authorized researchers must obtain independent credentialed access through PhysioNet to **MIMIC-IV v3.1**, **eICU-CRD v2.0**, and **SICdb v1.0.8** under each database's data-use agreement.
- **Primary vs. revision analyses:** Primary results use the frozen development models (`results/locked_models.json`, `results/raw_equations.json`) locked on 28 September 2026 (`05:57:17 UTC`) following the local protocol freeze (`05:49:06 UTC`). All post-review subset, penalty, missingness, recalibration, and source-linked analyses under `revision_20261116/` are explicitly secondary or exploratory; same-cohort external intercept updates (`B9_recalibration.csv`) represent apparent recalibration only.

## Repository Structure

- `protocol.md`, `screening_plan.md`, `REPRODUCIBILITY.md` — Prespecified local protocol, source screening plan, and environment/execution guide.
- `audit/` — Dated protocol and model freeze manifests (`protocol_freeze.json`, `model_freeze.json`), pre-fitting harmonization disclosure (`source_corrections_before_model.md`), floating-point boundary note, extraction receipts, and manually reviewed eICU feeding paths (`eicu_enteral_mapping_review.csv`).
- `sql/` — Standalone PostgreSQL cohort and laboratory queries exported from `scripts/extract.py` for MIMIC-IV and eICU-CRD.
- `scripts/` — Primary extraction (`extract.py`), ridge model development and unchanged external validation (`analyze.py`), independent SQL/arithmetic verification (`verify.py`), and figure generation (`figures.py`).
- `results/` — Locked model parameters, unstandardized equations, cohort flow, baseline summaries, primary discrimination/calibration metrics, paired differences (`model_differences.csv`), thresholds, and decision-curve coordinates.
- `revision_20261116/` — Revision sensitivity scripts (`code/`), aggregate sensitivity outputs with 95% bootstrap intervals (`results/stage2/`, `results/stage4/`), freeze verification receipt, and stage reports.
- `TRIPOD_AI_checklist.docx` — Complete 52-subitem TRIPOD+AI reporting checklist.
- `requirements.txt` & `SHA256SUMS.txt` — Exact Python package versions and SHA-256 checksums for every file in this archive.

## Reproducing the Analyses

1. **Work in a separate copy:** Never run write-capable scripts inside the archived reference directory, as `analyze.py` and `verify.py` overwrite output files and timestamps.
2. **Configure database connections:** Set standard `libpq` environment variables (`PGHOST` or `PGSERVICE`) for PostgreSQL schemas (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`, `eicu_crd`) and point the SICdb directory path in your working copy to licensed SICdb v1.0.8 tables (see `REPRODUCIBILITY.md`).
3. **Execute in order:**
   - Extract cohorts and features via `scripts/extract.py` (`mimic`, `eicu`, `sicdb`).
   - Fit and lock MIMIC-IV models via `scripts/analyze.py develop`.
   - Score external cohorts unchanged via `scripts/analyze.py validate` (or score directly against `results/locked_models.json`).
   - Run `scripts/verify.py` and `revision_20261116/code/` scripts to reproduce verification checks and revision sensitivity tables (`B2_thresholds.csv`, `B1_metrics.csv`, etc.).

These research equations and scripts are provided solely for scientific transparency and independent methodological evaluation; they have not been validated or approved for clinical bedside deployment.
