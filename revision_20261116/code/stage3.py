"""Revision-only sensitivities. All scoring is local, and outputs are aggregate."""
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import chi2
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
OUT = REV / "results" / "stage3"
OUT.mkdir(exist_ok=True, parents=True)
ANALYZE = ROOT / "scripts" / "analyze.py"
sys.dont_write_bytecode = True
loader = importlib.util.spec_from_file_location("frozen_analyze", ANALYZE)
original = importlib.util.module_from_spec(loader)
loader.loader.exec_module(original)
SEED = original.SEED
FEATURES = original.FEATURES
MODELS = original.SPECS
SOURCES = ("mimic", "eicu", "sicdb")
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LAB_FLAGS = {"creatinine", "potassium", "magnesium"}
METRICS = ("auc", "brier", "observed_expected", "average_precision",
           "calibration_intercept", "calibration_slope", "log_loss")
MI_M = 20
MI_ROUNDS = 10
MI_LO = np.array([0.1, np.log(0.1), 0.3, 1.0, 18.0, 10.0])
MI_HI = np.array([20.0, np.log(30.0), 10.0, 10.0, np.inf, 70.0])


def load_cohort(source):
    columns = [*FEATURES, "hospital", "followup_observed", "event"]
    frame = pd.read_parquet(ROOT / "private" / f"{source}_features.parquet", columns=columns)
    return frame.loc[frame["followup_observed"]].reset_index(drop=True)


def cv(y):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)


def prepare(frame, features, exclude=()):
    model = {
        "features": list(features),
        "medians": {f: float(frame[f].median()) for f in features},
        "missing_flags": [f for f in features if f not in exclude and frame[f].isna().any()
                          and frame[f].notna().any()],
        "knots": np.quantile(frame["phosphate"], [.1, .5, .9]).tolist(),
    }
    x = original.design(frame, model)
    model.update(columns=x.columns.tolist(), mean=x.mean().tolist(),
                 scale=x.std(ddof=0).replace(0, 1).tolist())
    return model


def transform(frame, model):
    x = original.design(frame, model)[model["columns"]].to_numpy()
    return (x - np.array(model["mean"])) / np.array(model["scale"])


def fit_prepared(model, z, y, c=1.0):
    solver = LogisticRegression(C=np.inf if c is None else c, l1_ratio=0,
                                solver="lbfgs", max_iter=2000, tol=1e-10)
    solver.fit(z, y)
    if solver.n_iter_[0] >= solver.max_iter:
        raise RuntimeError("Logistic optimizer did not converge")
    return {**model, "C": c, "coef": solver.coef_[0].tolist(),
            "intercept": float(solver.intercept_[0])}


def fit(frame, y, features, *, c=1.0, exclude=()):
    model = prepare(frame, features, exclude)
    return fit_prepared(model, transform(frame, model), y, c)


def tune(frame, y, features):
    predictions = np.empty((len(GRID), len(frame)))
    for train, test in cv(y):
        model = prepare(frame.iloc[train], features)
        z_train = transform(frame.iloc[train], model)
        z_test = transform(frame.iloc[test], model)
        for k, c in enumerate(GRID):
            fit_model = fit_prepared(model, z_train, y[train], c)
            predictions[k, test] = expit(z_test @ fit_model["coef"] + fit_model["intercept"])
    scores = [float(log_loss(y, p)) for p in predictions]
    return GRID[int(np.argmin(scores))], scores


def design_separation(z, y):
    # If a direction has strictly positive margin for some observations and
    # nonnegative margin for all, likelihood increases along it indefinitely.
    matrix = np.column_stack([np.ones(len(z)), z])
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise RuntimeError("Rank-deficient unpenalized model design")
    margins = (2 * y[:, None] - 1) * matrix
    result = linprog(-np.mean(margins, axis=0), A_ub=-margins,
                     b_ub=np.zeros(len(y)), bounds=(-1, 1), method="highs")
    if not result.success:
        raise RuntimeError("Unpenalized linear-program diagnostic did not converge")
    return result.fun < -1e-7


def indicator_boundary(frame, y, active_flags):
    cells = []
    for f in active_flags:
        flagged = frame[f].isna().to_numpy()
        if flagged.any() and len(np.unique(y[flagged])) == 1:
            cells.append((f, int(flagged.sum()), int(y[flagged].sum())))
    return cells


def separate(frame, y, features):
    model = prepare(frame, features)
    z = transform(frame, model)
    known = indicator_boundary(frame, y, model["missing_flags"])
    return model, z, known, bool(known) or design_separation(z, y)


def fit_mle(frame, y, features):
    model, z, known, separated = separate(frame, y, features)
    if separated:
        return None, known
    return fit_prepared(model, z, y, None), known


def likelihood_limit(frame, y, features):
    """Calculate the same full-design likelihood at a pure-indicator boundary."""
    full_model = prepare(frame, features)
    z = transform(frame, full_model)
    keep = np.ones(len(frame), dtype=bool)
    active = full_model["missing_flags"].copy()
    exclusions = []
    while cells := indicator_boundary(frame.loc[keep], y[keep], active):
        for feature, n, events in cells:
            exclusions.append({"feature": feature, "n": n, "events": events,
                               "small_cell": n < 5 or events < 5})
            keep &= frame[feature].notna().to_numpy()
            active.remove(feature)
    # Removing a pure indicator's rows lets their log-likelihood tend to 0.
    # Keep the original medians, spline knots and transforms for retained terms.
    active_columns = [not (col.endswith("_missing") and col[:-8] not in active)
                      for col in full_model["columns"]]
    x = z[keep][:, active_columns]
    if design_separation(x, y[keep]):
        raise RuntimeError("Non-indicator separation prevents boundary likelihood calculation")
    spec = fit_prepared({}, x, y[keep], None)
    probability = expit(x @ spec["coef"] + spec["intercept"])
    return (-float(log_loss(y[keep], probability, normalize=False)),
            x.shape[1] + 1, exclusions)


def mi_input(frame):
    x = frame[FEATURES].to_numpy(dtype=float, copy=True)
    x[:, FEATURES.index("creatinine")] = np.log(x[:, FEATURES.index("creatinine")])
    return x


def mi(seed, y_included):
    lo = np.r_[MI_LO, 0.0] if y_included else MI_LO
    hi = np.r_[MI_HI, 1.0] if y_included else MI_HI
    return IterativeImputer(BayesianRidge(), sample_posterior=True, max_iter=MI_ROUNDS,
                            min_value=lo, max_value=hi, random_state=seed)


def mi_frame(x):
    result = pd.DataFrame(x[:, :len(FEATURES)], columns=FEATURES)
    result["creatinine"] = np.exp(result["creatinine"])
    return result


def mi_train(frame, y, index):
    x = mi_input(frame)
    xy = np.column_stack([x, y])
    fitted = []
    for m in range(MI_M):
        seed = SEED + 3000 * index + 2 * m
        train_fill = mi(seed, True).fit_transform(xy)
        prediction_imputer = mi(seed + 1, False).fit(x)
        fitted.append((fit(mi_frame(train_fill), y, FEATURES), prediction_imputer))
    return fitted


def mi_predict(frame, ensemble):
    x = mi_input(frame)
    draws = []
    for model, imputer in ensemble:
        # Fix independent, deterministic prediction RNG starting states per evaluation.
        rng = imputer.random_state_.get_state()
        draws.append(original.predict(mi_frame(imputer.transform(x)), model))
        imputer.random_state_.set_state(rng)
    return np.mean(draws, axis=0)


def freeze_mi(ensemble):
    """Summarize the complete frozen conditional parameters, without rows."""
    described = []
    for model, imputer in ensemble:
        steps = []
        for feature, predictors, est in imputer.imputation_sequence_:
            steps.append({"feature": int(feature), "conditioning": predictors.astype(int).tolist(),
                          "coef": est.coef_.tolist(), "intercept": float(est.intercept_),
                          "posterior_covariance": est.sigma_.tolist(), "noise_precision": float(est.alpha_)})
        described.append({"outcome_model": model, "initial_fill": imputer.initial_imputer_.statistics_.tolist(),
                          "conditional_models": steps, "rng_seed": imputer.random_state})
    return described


def summarize(y, p, w):
    result = original.metrics(y, p, w)
    result["log_loss"] = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    return np.array([result[k] for k in METRICS])


def quantiles(values):
    v = np.asarray(values)
    return tuple(np.percentile(v, [2.5, 97.5]).tolist())


def evaluated(source, frame, y, alternatives, base):
    estimators = {**{"frozen_" + name: values for name, values in base.items()}, **alternatives}
    rng = np.random.default_rng(SEED)
    point = {k: summarize(y, p, np.ones(len(y))) for k, p in estimators.items()}
    boot = {k: [] for k in estimators}
    for _ in range(500):
        w = original.resample_weights(frame, rng, source)
        if 0 < w @ y < w.sum():
            for k, p in estimators.items():
                boot[k].append(summarize(y, p, w))
    records = []
    contrasts = []
    for k, arr in boot.items():
        if k.startswith("frozen_"):
            continue
        arr = np.asarray(arr)
        baseline = "frozen_" + k.split("|", 1)[1]
        diffs = arr - np.asarray(boot[baseline])
        for j, metric in enumerate(METRICS):
            cilo, cihi = quantiles(arr[:, j])
            row = {"source": source, "analysis": k.split("|", 1)[0], "model": k.split("|", 1)[1],
                   "n": len(y), "events": int(y.sum()), "metric": metric, "estimate": point[k][j],
                   "ci_low": cilo, "ci_high": cihi, "bootstrap_valid": len(arr),
                   "evaluation": "OOF" if source == "mimic" else "frozen_external",
                   "analysis_class": "revision sensitivity"}
            records.append(row)
            low, high = quantiles(diffs[:, j])
            contrasts.append({**row, "estimate": point[k][j] - point[baseline][j],
                              "ci_low": low, "ci_high": high, "comparison": "alternative_minus_original"})
    for analysis in ("B4a_unpenalized", "B4b_tuned", "B6"):
        for smaller, larger in (("phosphate_only", "phosphate_creatinine"),
                                ("phosphate_only", "six_variable"),
                                ("phosphate_creatinine", "six_variable")):
            left_key, right_key = f"{analysis}|{smaller}", f"{analysis}|{larger}"
            if left_key not in alternatives or right_key not in alternatives:
                continue
            arr = np.asarray(boot[right_key]) - np.asarray(boot[left_key])
            for j, metric in enumerate(METRICS):
                low, high = quantiles(arr[:, j])
                contrasts.append({"source": source, "analysis": analysis, "model": larger,
                                  "n": len(y), "events": int(y.sum()), "metric": metric,
                                  "estimate": point[right_key][j] - point[left_key][j],
                                  "ci_low": low, "ci_high": high, "bootstrap_valid": len(arr),
                                  "evaluation": "OOF" if source == "mimic" else "frozen_external",
                                  "analysis_class": "revision sensitivity",
                                  "comparison": f"{larger}_minus_{smaller}"})
    return records, contrasts


def changes(source, f, y, alt, base):
    rows = []
    group = f[["creatinine", "magnesium", "potassium"]].isna().any(axis=1).to_numpy()
    for model in ("six_variable", "phosphate_creatinine"):
        subsets = {"any_Cr_Mg_K_missing": group}
        if model == "phosphate_creatinine":
            subsets["Cr_missing_model_specific"] = f["creatinine"].isna().to_numpy()
        for subset, g in subsets.items():
            p0 = base[model][g]
            p1 = alt["B6|" + model][g]
            d = p1 - p0
            for cutoff in (0.05, 0.10, 0.20):
                before = p0 >= cutoff
                after = p1 >= cutoff
                row = {"source": source, "model": model, "subset": subset, "cutoff": cutoff,
                       "n": int(g.sum()), "events": int(y[g].sum()), "changed_up": int((~before & after).sum()),
                       "changed_down": int((before & ~after).sum()), "flagged_original": int(before.sum()),
                       "flagged_no_indicator": int(after.sum()), "mean_delta": float(d.mean()) if len(d) else np.nan,
                       "median_delta": float(np.median(d)) if len(d) else np.nan,
                       "q25_delta": float(np.quantile(d, .25)) if len(d) else np.nan,
                       "q75_delta": float(np.quantile(d, .75)) if len(d) else np.nan}
                for f_name in ("n", "events", "changed_up", "changed_down", "flagged_original", "flagged_no_indicator"):
                    row[f_name + "_lt5"] = 0 <= row[f_name] < 5
                rows.append(row)
    return rows


def coef_table(final, locked):
    rows = []
    for k, spec in final.items():
        if not isinstance(spec, dict):
            continue
        analysis, name = k.split("|", 1)
        reference = locked[name]
        for tag, m in ((analysis, spec), ("original_C1", reference)):
            b_std = np.asarray(m["coef"])
            b_raw = b_std / np.asarray(m["scale"])
            intercept_raw = m["intercept"] - b_raw @ np.asarray(m["mean"])
            for i, term in enumerate(["intercept", *m["columns"]]):
                rows.append({"analysis": analysis, "model": name, "fit": tag, "term": term,
                             "C": m.get("C"), "standardized": ([m["intercept"]] + b_std.tolist())[i],
                             "original_transform": ([float(intercept_raw)] + b_raw.tolist())[i]})
    return rows


def main():
    frozen_models = json.loads((ROOT / "results/locked_models.json").read_text(encoding="utf-8"))
    mimic = load_cohort("mimic")
    y = mimic["event"].to_numpy(dtype=int)
    assert (len(y), int(y.sum())) == (5622, 836)
    folds = list(cv(y))
    oof = {}
    final = {}
    mle_diagnostics = []
    tuning = []
    for name, features in MODELS.items():
        key = "B4a_unpenalized|" + name
        predictions = np.full(len(mimic), np.nan)
        for i, (train, test) in enumerate(folds, 1):
            spec, known = fit_mle(mimic.iloc[train], y[train], features)
            mle_diagnostics.append({"model": name, "training_split": f"outer_{i}",
                                    "finite_mle": spec is not None, "one_class_indicators": known})
            if spec:
                predictions[test] = original.predict(mimic.iloc[test], spec)
        spec, known = fit_mle(mimic, y, features)
        mle_diagnostics.append({"model": name, "training_split": "full_MIMIC",
                                "finite_mle": spec is not None, "one_class_indicators": known})
        if spec:
            final[key] = spec
        if np.isfinite(predictions).all():
            oof[key] = predictions

        key = "B4b_tuned|" + name
        predictions = np.empty(len(mimic))
        for i, (train, test) in enumerate(folds, 1):
            training = mimic.iloc[train]
            c, loss = tune(training, y[train], features)
            tuning.extend({"model": name, "selection": f"outer_{i}", "C": v, "log_loss": l,
                           "selected": v == c} for v, l in zip(GRID, loss))
            predictions[test] = original.predict(mimic.iloc[test], fit(training, y[train], features, c=c))
        oof[key] = predictions
        c, loss = tune(mimic, y, features)
        tuning.extend({"model": name, "selection": "full_MIMIC", "C": v, "log_loss": l,
                       "selected": v == c} for v, l in zip(GRID, loss))
        final[key] = fit(mimic, y, features, c=c)

    variants = {
        "B6|phosphate_creatinine": (MODELS["phosphate_creatinine"], LAB_FLAGS),
        "B6|six_variable": (FEATURES, LAB_FLAGS),
        "B7a_no_bmi|six_variable": ([f for f in FEATURES if f != "bmi"], ()),
    }
    for key, (feature, exclude) in variants.items():
        predictions = np.empty(len(mimic))
        for train, test in folds:
            fitted = fit(mimic.iloc[train], y[train], feature, exclude=exclude)
            predictions[test] = original.predict(mimic.iloc[test], fitted)
        oof[key] = predictions
        final[key] = fit(mimic, y, feature, exclude=exclude)

    lls = {name: likelihood_limit(mimic, y, features) for name, features in MODELS.items()}
    lr = []
    for sm, lg in (("phosphate_only", "phosphate_creatinine"),
                   ("phosphate_only", "six_variable"),
                   ("phosphate_creatinine", "six_variable")):
        smaller_ll, smaller_df, smaller_boundary = lls[sm]
        larger_ll, larger_df, larger_boundary = lls[lg]
        boundary = bool(smaller_boundary or larger_boundary)
        statistic = 2 * (larger_ll - smaller_ll)
        df = larger_df - smaller_df if not boundary else None
        lr.append({"smaller": sm, "larger": lg, "deviance_gain": statistic,
                   "degrees_of_freedom": df, "p_value": float(chi2.sf(statistic, df)) if df else None,
                   "regular_finite_mle": not boundary, "smaller_loglike": smaller_ll,
                   "larger_loglike": larger_ll, "larger_boundary_cells": larger_boundary})

    print("Nested tuning, MLE diagnostics, B6 and B7a complete; running MIMIC-only chained imputation.", flush=True)
    predictions = np.empty(len(mimic))
    for i, (train, test) in enumerate(folds, 1):
        ensemble = mi_train(mimic.iloc[train], y[train], i)
        predictions[test] = mi_predict(mimic.iloc[test], ensemble)
        print(f"MI outer fold {i} complete.", flush=True)
    oof["B7b_MI|six_variable"] = predictions
    ensemble = mi_train(mimic, y, 6)
    freeze = {
        "frozen_before_external_evaluation_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "MIMIC only", "specs": final, "MI_m": MI_M, "MI_iterations": MI_ROUNDS,
        "MI_parameters": freeze_mi(ensemble),
    }
    model_file = OUT / "mimic_sensitivity_fits.json"
    model_file.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    freeze_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    pd.DataFrame(tuning).to_csv(OUT / "b4_tuning.csv", index=False)
    pd.DataFrame(coef_table(final, frozen_models)).to_csv(OUT / "coefficients.csv", index=False)
    (OUT / "b3b_lrt.json").write_text(json.dumps(lr, indent=2), encoding="utf-8")
    (OUT / "b4_mle_diagnostics.json").write_text(json.dumps(mle_diagnostics, indent=2), encoding="utf-8")
    print("MIMIC-only sensitivity parameters frozen; evaluating.", flush=True)
    all_perf = []
    all_diffs = []
    all_shifts = []
    for source in SOURCES:
        frame = mimic if source == "mimic" else load_cohort(source)
        outcome = frame["event"].to_numpy(dtype=int)
        predictions = pd.read_parquet(ROOT / "private" / f"{source}_predictions.parquet")
        assert np.array_equal(predictions["event"].to_numpy(dtype=int), outcome)
        baseline = {name: predictions[name].to_numpy() for name in MODELS}
        if source == "mimic":
            alternative = oof
        else:
            scoring = frame[FEATURES]
            alternative = {key: original.predict(scoring, spec) for key, spec in final.items()}
            alternative["B7b_MI|six_variable"] = mi_predict(scoring, ensemble)
            for name, spec in frozen_models.items():
                np.testing.assert_allclose(original.predict(scoring, spec), baseline[name], atol=1e-12, rtol=1e-10)
        records, diffs = evaluated(source, frame, outcome, alternative, baseline)
        all_perf += records
        all_diffs += diffs
        all_shifts += changes(source, frame, outcome, alternative, baseline)
        print(f"Completed aggregate evaluation: {source}.", flush=True)
    pd.DataFrame(all_perf).to_csv(OUT / "performance.csv", index=False)
    pd.DataFrame(all_diffs).to_csv(OUT / "paired_differences.csv", index=False)
    pd.DataFrame(all_shifts).to_csv(OUT / "b6_missing_subgroup_threshold_changes.csv", index=False)
    (OUT / "receipt.json").write_text(json.dumps({"model_freeze_hash": freeze_hash, "patients": 7734,
         "development_events": int(y.sum()), "bootstrap_seed": SEED, "bootstrap_attempts": 500,
         "evaluation_rows": len(all_perf), "paired_rows": len(all_diffs), "missing_shift_rows": len(all_shifts),
         "row_level_outputs": 0}, indent=2), encoding="utf-8")
    print("Stage 3 complete: aggregate tables and MIMIC-only parameter manifest written.", flush=True)


if __name__ == "__main__":
    main()

