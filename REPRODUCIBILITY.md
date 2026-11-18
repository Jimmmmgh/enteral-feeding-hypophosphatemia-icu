# Local reproducibility guide

The manuscript reports a retrospective, locally specified research analysis, not a deployed clinical tool. The package is for authorized local author review. It contains aggregate results and code, not patient-level data or credentials. Source-data access and any distribution of derived parameters remain subject to source agreements. SICdb study-specific authorization has not been confirmed in this task.

## Inputs and environments

The analysis used local PostgreSQL databases named `mimic` and `eicu`, with schemas `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived` and `eicu_crd`. The installed first-day height and weight concepts are required. SICdb files were read from `E:\Physionet\SICdb\1.0.8`. The source releases were MIMIC-IV 3.1 and eICU-CRD 2.0 as confirmed for the preceding study, and SICdb 1.0.8 as supported by the local verified download provenance. The two PostgreSQL installation manifests were not independently regenerated in this study.

The scientific runtime was Python 3.13.0 with scikit-learn 1.8.0, pandas, numpy, scipy, statsmodels, psycopg2, pyarrow and matplotlib. Documents were built with the bundled document runtime and python-docx, rendered in Microsoft Word and rasterized with pypdfium2. Exact model medians, knots, means, scales, coefficients and intercepts are in `results/locked_models.json`; unstandardized transformed-term equations are in `results/raw_equations.json`.

## Repeating the scientific work

Use a separate copy of the project for a new run so that the original protocol, frozen model and audit hashes remain intact. Obtain the approved database access before re-extraction. Configure PostgreSQL through the user's libpq environment or service configuration; do not place passwords in shared scripts. The included `db.py` is a credential-free connector. Its local fallback imports an existing approved connection helper on the original workstation; that helper and its secrets are not distributed. Outside that workstation, set `PGHOST` or `PGSERVICE` and the appropriate libpq authentication configuration. The package's scripts also include a historical search path to the preceding local study; if that path does not exist, Python falls back to the included `db.py`.

The SICdb directory is explicit in `source_audit.py`, `capacity.py` and `extract.py`; update that input path in a new copy if needed. These scripts preserve the original extraction specification. The original timing-feasibility script and selected eICU path mapping were hashed before outcome inspection. Do not reinterpret a rerun's timestamp as a new prospective protocol registration.

Run the extraction for `mimic`, `eicu` and `sicdb`. Then run `analyze.py develop` using only MIMIC data. That command creates the development predictions and model-freeze record. Only afterward run `analyze.py validate`, which evaluates both external datasets without fitting to them. `verify.py` checks arithmetic, feature and model hashes, time boundaries and independent MIMIC/eICU aggregate SQL reconstruction. `figures.py` produces all four figures. `build_documents.py` reads only aggregate outputs, references and the authorized author metadata. Native Word rendering and `check_documents.py` validate the generated documents.

## Interpretation and provenance

The primary outcome is a detected phosphate value below 2.0 mg/dL within the post-index observation window. Absence of testing is an unknown outcome, not a negative. The risk index is 24 hours after first qualifying documented enteral feeding, not verified first calorie exposure. The model does not diagnose refeeding syndrome, establish a nutritional cause or justify less frequent monitoring.

Read `protocol.md`, `audit/source_corrections_before_model.md` and `audit/floating_point_boundary_note.md` together. The original protocol was frozen before outcome concentrations were inspected. Decimal times, SICdb admission-weight scaling and arterial potassium mapping were corrected after aggregate event counts were known but before any model fitting. A later eICU floating-point boundary correction changed intermediate flow counts only; final cohort membership and predictions did not change.

The source-level numerical and document checks were performed by the same analysis agent. They are reproducibility checks, not independent clinical review. The 500-resample uncertainty intervals condition on existing fitted predictions. Development sensitivity results use out-of-fold predictions except the two-hour specimen buffer, which uses apparent predictions from the fully fitted model. Alternative-outcome AUROCs do not validate the original probabilities for those different outcomes.

The document builder contains two bibliography formatting overrides verified against publisher pages: the SICdb 1.0.8 dataset author list and the named authors preceding the STRATOS consortium in the calibration reference. Raw Crossref metadata are preserved in the audit folder. Human-readable supplementary coefficients are rounded to eight decimal places; the document QA verifies that this rounding changes illustrative predicted probabilities by less than 0.000001.

The local draft is not uploaded, submitted, accepted or published. Do not distribute the entire internal audit package as a journal supplement without checking author approval, source permissions and the journal's requirements.
