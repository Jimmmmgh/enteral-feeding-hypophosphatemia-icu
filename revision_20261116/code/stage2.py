"""Revision-only sensitivity analyses from frozen predictions and derived features."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_20261116"
OUT = REVISION / "results" / "stage2"
SOURCES = ("mimic", "eicu", "sicdb")
MODELS = ("phosphate_only", "phosphate_creatinine", "six_variable")
COMPARISONS = ((2, 0), (2, 1), (1, 0))
FEATURES = ("phosphate", "creatinine", "magnesium", "potassium", "age", "bmi")
LABS = FEATURES[:4]
METRICS = ("auc", "brier", "calibration_intercept", "calibration_slope", "observed_expected")
THRESHOLD_METRICS = ("fraction_flagged", "sensitivity", "specificity", "ppv", "npv", "missed_events")
SEED = 20260928
B = 500
THRESHOLDS = np.array([.05, .10, .20])
DCA_THRESHOLDS = np.array([.05, .10, .15, .20, .25, .30])
RULES = (2.5, 3.0)


def read_data(source, specs):
    features = pd.read_parquet(ROOT / "private" / f"{source}_features.parquet")
    saved = pd.read_parquet(ROOT / "private" / f"{source}_predictions.parquet")
    measured = features.loc[features.followup_observed].reset_index(drop=True)
    outcomes = measured.event.to_numpy(dtype=int)
    assert np.array_equal(outcomes, saved.event)
    scores = saved[list(MODELS)].to_numpy()
    if source == "mimic":
        oof = pd.read_parquet(ROOT / "private" / "mimic_oof_predictions.parquet")
        np.testing.assert_array_equal(scores, oof[list(MODELS)].to_numpy())
        np.testing.assert_array_equal(outcomes, oof.event)
        np.testing.assert_array_equal(measured.hospital, oof.hospital)
    else:
        np.testing.assert_allclose(scores, frozen_scores(measured, specs), atol=1e-12, rtol=0)
    return features, measured, outcomes, scores


def frozen_terms(d, spec):
    cols = {}
    for lab in spec["features"]:
        x = d[lab].to_numpy(dtype=float)
        available = np.isfinite(x)
        x = np.where(available, x, spec["medians"][lab])
        if lab == "phosphate":
            a, b, c = spec["knots"]
            cube = lambda knot: np.maximum(x - knot, 0) ** 3
            cols["phosphate"] = x
            cols["phosphate_rcs"] = (
                cube(a) - cube(b) * (c - a) / (c - b) + cube(c) * (b - a) / (c - b)
            ) / (c - a) ** 2
        elif lab == "creatinine":
            cols["log_creatinine"] = np.log(x)
        else:
            cols[lab] = x
        if lab in spec["missing_flags"]:
            cols[f"{lab}_missing"] = ~available
    design = np.column_stack([cols[name] for name in spec["columns"]])
    return ((design - spec["mean"]) / spec["scale"]) * spec["coef"]


def frozen_scores(d, specs):
    return np.column_stack([
        expit(specs[m]["intercept"] + frozen_terms(d, specs[m]).sum(axis=1)) for m in MODELS
    ])


def bootstrap_weights(d, source):
    rng = np.random.default_rng(SEED)
    if source == "eicu":
        assert d.hospital.notna().all()
        _, cluster = np.unique(d.hospital, return_inverse=True)
    else:
        cluster = np.arange(len(d))
    n_units = np.max(cluster) + 1
    weights = np.empty((B + 1, len(d)))
    weights[0] = 1
    for i in range(1, B + 1):
        weights[i] = np.bincount(rng.integers(n_units, size=n_units), minlength=n_units)[cluster]
    return weights


def percentile_summary(values):
    valid = np.isfinite(values[1:])
    ci = np.quantile(values[1:][valid], [.025, .975]) if valid.any() else (np.nan, np.nan)
    return dict(estimate=values[0], ci_low=ci[0], ci_high=ci[1], bootstrap_valid=int(valid.sum()))


def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan),
                     where=np.asarray(b) != 0)


def rank_metrics(y, p, weights):
    order = np.argsort(-p, kind="stable")
    ties = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    y_sorted = y[order]
    w = weights[:, order]
    pos = np.add.reduceat(w * y_sorted, ties, axis=1)
    neg = np.add.reduceat(w * (1 - y_sorted), ties, axis=1)
    cp = pos.cumsum(axis=1)
    cn = neg.cumsum(axis=1)
    total_events = cp[:, -1]
    total_neg = cn[:, -1]
    auc = divide((pos * (total_neg[:, None] - cn + .5 * neg)).sum(axis=1), total_events * total_neg)
    precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=cp + cn > 0)
    ap = divide((pos * precision).sum(axis=1), total_events)
    return auc, ap


def calibration(y, p, weights):
    """Offset intercept and free-intercept logistic recalibration slope."""
    z = logit(np.clip(p, 1e-8, 1 - 1e-8))
    out = np.full((len(weights), 2), np.nan)
    x = np.column_stack([np.ones(len(y)), z])
    for i, w in enumerate(weights):
        if w @ y == 0 or w @ (1 - y) == 0:
            continue
        beta = np.array([0., 1.])
        for _ in range(50):
            q = expit(x @ beta)
            info = x.T @ ((w * q * (1 - q))[:, None] * x)
            step = np.linalg.solve(info + np.eye(2) * 1e-10, x.T @ (w * (y - q)))
            beta += np.clip(step, -2, 2)
            if np.max(np.abs(step)) < 1e-8:
                out[i, 1] = beta[1]
                break
        alpha = 0.
        for _ in range(50):
            q = expit(z + alpha)
            step = (w @ (y - q)) / max(w @ (q * (1 - q)), 1e-12)
            alpha += np.clip(step, -2, 2)
            if abs(step) < 1e-8:
                out[i, 0] = alpha
                break
    return out


def metrics(y, p, w):
    auc, _ = rank_metrics(y, p, w)
    intercept, slope = calibration(y, p, w).T
    return dict(auc=auc, brier=(w @ (y - p) ** 2) / w.sum(axis=1), calibration_intercept=intercept,
                calibration_slope=slope, observed_expected=divide(w @ y, w @ p))


def threshold_arrays(y, flags, w):
    tp = w @ (y[:, None] * flags)
    tn = w @ ((1 - y[:, None]) * ~flags)
    fp = w @ ((1 - y[:, None]) * flags)
    fn = w @ (y[:, None] * ~flags)
    total = w.sum(axis=1)[:, None]
    return dict(fraction_flagged=(tp + fp) / total, sensitivity=divide(tp, tp + fn),
                specificity=divide(tn, tn + fp), ppv=divide(tp, tp + fp), npv=divide(tn, tn + fn),
                missed_events=fn, flagged=tp + fp, unflagged=tn + fn,
                tp=tp, fp=fp, total=total)


def described(n, events):
    return dict(n=n, events=events, sparse_n_lt_5=n < 5, sparse_events_lt_5=events < 5)


def metric_columns(values, i, metric_names):
    row = {}
    for name in metric_names:
        interval = percentile_summary(values[name][:, i])
        row[name] = interval.pop("estimate")
        row.update({f"{name}_{key}": value for key, value in interval.items()})
    row["sparse_missed_events_lt_5"] = row["missed_events"] < 5
    row["sparse_flagged_lt_5"] = values["flagged"][0, i] < 5
    row["sparse_unflagged_lt_5"] = values["unflagged"][0, i] < 5
    return row


def comparison(i, j):
    return f"{MODELS[i]}_minus_{MODELS[j]}"


def check_frozen_files():
    reference = json.loads((REVISION / "checkpoint1_freeze_verification.json").read_text(encoding="utf-8"))
    observed = {
        path: sha256((ROOT / path).read_bytes()).hexdigest()
        for path in reference["protected_input_sha256"]
    }
    assert observed == reference["protected_input_sha256"]
    return len(observed)


def b1(source, d, y, scores, specs):
    buffered = d.copy()
    for col in LABS:
        buffered[col] = buffered[f"{col}_buffer2h"]
    buffered = buffered.loc[buffered.phosphate.notna()]
    subsets = [
        ("complete_cases", d[list(FEATURES)].notna().all(axis=1)),
        ("daily_surveillance", d.daily_surveillance),
    ]
    evaluation = "out_of_fold" if source == "mimic" else "frozen_external"
    datasets = [(label, d.loc[mask], y[mask], scores[mask], evaluation) for label, mask in subsets]
    datasets.append(("two_hour_predictor_buffer", buffered, buffered.event.to_numpy(dtype=int),
                     frozen_scores(buffered, specs), "apparent" if source == "mimic" else evaluation))
    output, diffs = [], []
    for label, sub, outcomes, p, ev in datasets:
        w = bootstrap_weights(sub, source)
        raw = [metrics(outcomes, p[:, i], w) for i in range(len(MODELS))]
        common = dict(source=source, analysis=label, evaluation=ev,
                      **described(len(sub), int(outcomes.sum())))
        for model, m in zip(MODELS, raw):
            for metric, values in m.items():
                output.append(dict(**common, model=model, metric=metric, **percentile_summary(values)))
        for i, j in COMPARISONS[:2]:
            for metric in ("auc", "brier"):
                diffs.append(dict(**common, comparison=comparison(i, j), metric=metric,
                                  **percentile_summary(raw[i][metric] - raw[j][metric])))
    return output, diffs


def b2_b3_e2(source, d, y, scores, w, evaluation):
    thresh, ap, dnb, rules, rule_nb = [], [], [], [], []
    common = dict(source=source, evaluation=evaluation, **described(len(d), int(y.sum())))
    aps, net = [], []
    for i, model in enumerate(MODELS):
        p = scores[:, i]
        tab = threshold_arrays(y, p[:, None] >= THRESHOLDS, w)
        for j, threshold in enumerate(THRESHOLDS):
            thresh.append(dict(**common, model=model, threshold=threshold,
                               **metric_columns(tab, j, THRESHOLD_METRICS)))
        _, ap_values = rank_metrics(y, p, w)
        aps.append(ap_values)
        ap.append(dict(**common, model=model, comparison="", **percentile_summary(ap_values)))
        tab = threshold_arrays(y, p[:, None] >= DCA_THRESHOLDS, w)
        net.append((tab["tp"] - tab["fp"] * DCA_THRESHOLDS / (1 - DCA_THRESHOLDS)) / tab["total"])
    for i, j in COMPARISONS:
        cm = comparison(i, j)
        ap.append(dict(**common, model="", comparison=cm, **percentile_summary(aps[i] - aps[j])))
        for k, threshold in enumerate(DCA_THRESHOLDS):
            dnb.append(dict(**common, threshold=threshold, comparison=cm,
                            **percentile_summary(net[i][:, k] - net[j][:, k])))
    for cutoff in RULES:
        tab = threshold_arrays(y, (d.phosphate.to_numpy() < cutoff)[:, None], w)
        rules.append(dict(**common, baseline_phosphate_lt_mg_dl=cutoff,
                          **metric_columns(tab, 0, ("fraction_flagged", "sensitivity", "npv", "missed_events"))))
        for threshold in THRESHOLDS:
            nb = (tab["tp"][:, 0] - tab["fp"][:, 0] * threshold / (1 - threshold)) / tab["total"][:, 0]
            rule_nb.append(dict(**common, baseline_phosphate_lt_mg_dl=cutoff, threshold=threshold,
                                **percentile_summary(nb)))
    return thresh, ap, dnb, rules, rule_nb


def b8(source, d, specs):
    terms, distributions = [], []
    for model in MODELS[1:]:
        spec = specs[model]
        contribution = pd.DataFrame(frozen_terms(d, spec), columns=spec["columns"])
        contribution["phosphate"] += contribution.pop("phosphate_rcs")
        mean = contribution.mean()
        mean["intercept"] = spec["intercept"]
        mean["total_logit"] = mean.sum()
        np.testing.assert_allclose(mean["total_logit"],
                                   spec["intercept"] + frozen_terms(d, spec).sum(axis=1).mean(), atol=1e-12)
        for term, value in mean.items():
            terms.append(dict(source=source, model=model, term=term, mean_logit_contribution=value, n=len(d),
                              evaluation="full_development_frozen_fit"))
    for variable in FEATURES:
        x = d[variable]
        q1, med, q3 = x.quantile([.25, .5, .75])
        missing = int(x.isna().sum())
        distributions.append(dict(source=source, variable=variable, n=len(x), available=int(x.notna().sum()),
                                  missing=missing, missing_fraction=missing / len(x), mean=x.mean(),
                                  sd=x.std(ddof=1), q25=q1, median=med, q75=q3, iqr=q3-q1,
                                  sparse_missing_lt_5=missing < 5))
    return terms, distributions


def b9(source, y, scores, specs):
    if source == "mimic":
        return []
    out = []
    for i, model in enumerate(MODELS):
        p = scores[:, i]
        z = logit(np.clip(p, 1e-8, 1 - 1e-8))
        w = np.ones((1, len(y)))
        alpha, slope = calibration(y, p, w)[0]
        q = expit(z + alpha)
        oe = y.sum() / q.sum()
        slope_after = calibration(y, q, w)[0, 1]
        assert abs(oe - 1) < 1e-10
        np.testing.assert_allclose(slope, slope_after, rtol=0, atol=1e-7)
        out.append(dict(source=source, model=model, evaluation="apparent_exploratory", **described(len(y), int(y.sum())),
                        intercept_update=alpha, frozen_intercept=specs[model]["intercept"],
                        updated_model_intercept=specs[model]["intercept"] + alpha, original_brier=np.mean((y-p)**2),
                        brier=np.mean((y-q)**2), original_observed_expected=y.sum()/p.sum(), observed_expected=oe,
                        calibration_slope=slope_after, original_calibration_slope=slope))
    return out


def c1(source, d, y, scores, evaluation):
    hours = d.first_followup_hours.to_numpy(dtype=float)
    assert np.isfinite(hours).all() and (hours > 0).all() and (hours <= 48).all()
    masks = (hours <= 12, (hours > 12) & (hours <= 24), hours > 24)
    q1, med, q3 = np.quantile(hours, [.25, .5, .75])
    dist = dict(source=source, n=len(hours), median_hours=med, q25_hours=q1, q75_hours=q3,
                min_hours=hours.min(), max_hours=hours.max(), n_le_12=int(masks[0].sum()),
                n_12_to_24=int(masks[1].sum()), n_gt_24=int(masks[2].sum()))
    output = []
    for stratum, mask in zip(("le_12h", "gt_12h_le_24h", "gt_24h"), masks):
        yy = y[mask]
        for i, model in enumerate(MODELS):
            p = scores[mask, i]
            auc = roc_auc_score(yy, p) if np.unique(yy).size == 2 else np.nan
            output.append(dict(source=source, stratum=stratum, evaluation=evaluation, model=model,
                               **described(len(yy), int(yy.sum())), auc=auc, observed_expected=divide(yy.sum(), p.sum()).item()))
    return output, [dist]


def c2(source, features, d, y, scores, specs):
    missing = features.loc[~features.followup_observed]
    assert len(missing) == dict(mimic=405, eicu=207, sicdb=82)[source]
    assert missing.event.isna().all()
    frozen_missing = frozen_scores(missing, specs)
    if source == "mimic":
        versions = [("mixed_oof_and_locked", scores),
                    ("all_locked_apparent", frozen_scores(d, specs))]
    else:
        versions = [("frozen_external", scores)]
    rows = []
    for status, observed_scores in versions:
        for assumed_event in (0, 1):
            yy = np.r_[y, np.full(len(missing), assumed_event)]
            for i, model in enumerate(MODELS):
                p = np.r_[observed_scores[:, i], frozen_missing[:, i]]
                rows.append(dict(source=source, model=model, evaluation=status, assumed_event_for_untested=assumed_event,
                                 n_without_followup=len(missing), **described(len(yy), int(yy.sum())),
                                 auc=roc_auc_score(yy, p), brier=np.mean((yy-p)**2),
                                 observed_expected=yy.sum()/p.sum()))
    return rows


def e3(source, d, scores, evaluation):
    baseline = d.phosphate.to_numpy()
    nadir = d.phosphate_nadir.to_numpy()
    assert np.isfinite(nadir).all() and (baseline > 0).all()
    # Since the lab values have variable recorded precision, retain equality at
    # the numerical boundary (e.g. 2.1 from 3.0) within floating-point tolerance.
    relative_decline = (baseline - nadir) / baseline
    y = (relative_decline >= .30 - 1e-12).astype(int)
    return [dict(source=source, model=model, evaluation=evaluation, **described(len(y), int(y.sum())),
                 auc=roc_auc_score(y, scores[:, i])) for i, model in enumerate(MODELS)]


def f5(d):
    site = d.groupby("hospital").event.agg(["size", "sum"])
    assert len(site) == 41
    rows = []
    for variable, field in (("included_patients", "size"), ("events", "sum")):
        x = site[field]
        q1, med, q3 = x.quantile([.25, .5, .75])
        rows.append(dict(source="eicu", variable=variable, n_hospitals=len(site),
                         q25=q1, median=med, q75=q3, iqr=q3-q1, minimum=int(x.min()), maximum=int(x.max()),
                         minimum_lt_5=x.min()<5, hospitals_with_fewer_than_10_patients=int(site["size"].lt(10).sum())))
    return rows


def loess_bootstrap(y, p, w):
    """Degree-one tricube LOESS with bandwidths fixed at the observed 35% span."""
    low, high = np.quantile(p, [.025, .975])
    grid = np.linspace(low, high, 101)
    distance = p[:, None] - grid
    k = min(len(p)-1, int(np.ceil(.35 * len(p)))-1)
    radius = np.partition(np.abs(distance), k, axis=0)[k]
    kernel = np.maximum(1 - (np.abs(distance) / radius) ** 3, 0) ** 3
    dx = kernel * distance
    mom0, mom1, mom2 = w @ kernel, w @ dx, w @ (dx * distance)
    m_y, m_xy = (w * y) @ kernel, (w * y) @ dx
    fitted = divide(mom2 * m_y - mom1 * m_xy, mom0 * mom2 - mom1**2)
    clipped = np.clip(fitted, 0, 1)
    return [dict(predicted=predicted, **percentile_summary(clipped[:, i])) for i, predicted in enumerate(grid)]


def check_reference(out):
    frozen_sensitivity = pd.read_csv(ROOT / "results" / "sensitivity.csv")
    b1_table = out["B1_metrics"]
    calc = b1_table.loc[b1_table.model.eq("six_variable")].merge(frozen_sensitivity, on=["source","analysis"])
    assert len(calc) == 45
    for row in calc.to_dict("records"):
        assert row["n_x"] == row["n_y"] and row["events_x"] == row["events_y"]
        np.testing.assert_allclose(row["estimate"], row[row["metric"]], atol=1e-10, rtol=0)
    threshold = out["B2_thresholds"].loc[lambda d: d.model.eq("six_variable")]
    frozen_threshold = pd.read_csv(ROOT / "results" / "thresholds.csv")
    paired = threshold.merge(frozen_threshold, on=["source", "threshold"], suffixes=("_calc", "_frozen"))
    assert len(paired) == 9
    overlap = ("n", "events", *THRESHOLD_METRICS, "sensitivity_ci_low", "sensitivity_ci_high",
               "npv_ci_low", "npv_ci_high", "fraction_flagged_ci_low", "fraction_flagged_ci_high")
    for name in overlap:
        np.testing.assert_allclose(paired[f"{name}_calc"], paired[f"{name}_frozen"], atol=1e-10, rtol=0)
    ap = out["B2_average_precision"]
    frozen_ap = pd.read_csv(ROOT / "results" / "performance.csv").query("metric == 'average_precision'")
    paired = ap.merge(frozen_ap, on=["source","model"], suffixes=("_calc", "_frozen"))
    assert len(paired) == 9
    ap_diffs = pd.read_csv(ROOT / "results" / "model_differences.csv").query("metric == 'average_precision'")
    ap_checks = ap.merge(ap_diffs, on=["source","comparison"], suffixes=("_calc", "_frozen"))
    assert len(ap_checks) == 6
    for name in ("estimate", "ci_low", "ci_high"):
        np.testing.assert_allclose(paired[f"{name}_calc"], paired[f"{name}_frozen"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(ap_checks[f"{name}_calc"], ap_checks[f"{name}_frozen"], atol=1e-10, rtol=0)


def run():
    verified_files = check_frozen_files()
    specs = json.loads((ROOT / "results" / "locked_models.json").read_text(encoding="utf-8"))
    names = ("B1_metrics", "B1_differences", "B2_thresholds", "B2_average_precision", "B3a_net_benefit",
             "B8_contributions", "B8_predictor_distribution", "B9_recalibration", "B10_loess", "B10_ROC_display",
             "C1_time_strata", "C1_time_distribution", "C2_bounds", "E2_rules", "E2_net_benefit",
             "E3_relative_decline", "F5_hospital_distribution")
    out = {name: [] for name in names}
    for source in SOURCES:
        all_d, d, y, scores = read_data(source, specs)
        print(f"Computing {source} aggregate revision sensitivity results", flush=True)
        evaluation = "out_of_fold" if source == "mimic" else "frozen_external"
        w = bootstrap_weights(d, source)
        result, diffs = b1(source, d, y, scores, specs)
        out["B1_metrics"] += result
        out["B1_differences"] += diffs
        result_names = ("B2_thresholds", "B2_average_precision", "B3a_net_benefit", "E2_rules", "E2_net_benefit")
        for name, rows in zip(result_names, b2_b3_e2(source, d, y, scores, w, evaluation)):
            out[name] += rows
        terms, distribution = b8(source, d, specs)
        out["B8_contributions"] += terms
        out["B8_predictor_distribution"] += distribution
        out["B9_recalibration"] += b9(source, y, scores, specs)
        for i, model in enumerate(MODELS):
            p = scores[:, i]
            out["B10_loess"] += [dict(source=source, model=model, **row) for row in loess_bootstrap(y, p, w)]
            fpr, tpr, _ = roc_curve(y, p)
            # Fixed plotting grid makes the ROC display artifact aggregate-only.
            grid = np.linspace(0, 1, 201)
            sens = np.interp(grid, fpr, tpr)
            out["B10_ROC_display"] += [
                dict(source=source, model=model, false_positive_fraction=x, sensitivity=yy)
                for x, yy in zip(grid, sens)
            ]
        strata, distribution = c1(source, d, y, scores, evaluation)
        out["C1_time_strata"] += strata
        out["C1_time_distribution"] += distribution
        out["C2_bounds"] += c2(source, all_d, d, y, scores, specs)
        out["E3_relative_decline"] += e3(source, d, scores, evaluation)
        if source == "eicu":
            out["F5_hospital_distribution"] += f5(d)
    out = {name: pd.DataFrame(rows) for name, rows in out.items()}
    contributions = out["B8_contributions"]
    reference = contributions.loc[contributions.source.eq("mimic")].set_index(["model", "term"])
    contributions["difference_vs_mimic"] = contributions["mean_logit_contribution"].to_numpy() - (
        reference.loc[pd.MultiIndex.from_frame(contributions[["model", "term"]]), "mean_logit_contribution"].to_numpy()
    )
    check_reference(out)
    assert check_frozen_files() == verified_files
    validation = dict(technical_status="PASS", frozen_input_files_verified=verified_files, bootstrap_seed=SEED,
                      bootstrap_repetitions=B, no_database_reads=True, new_external_fit="B9 intercept only",
                      original_B1_6_variable_points_matched=45, original_B2_threshold_rows_matched=9,
                      original_AP_points_and_intervals_matched=9, original_AP_difference_points_and_intervals_matched=6,
                      rows_by_output={name: len(d) for name, d in out.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        if name == "B10_ROC_display":
            frame["analysis_scope"] = "frozen_main_analysis_display_only"
        else:
            frame["analysis_scope"] = "exploratory" if name in {"B3a_net_benefit", "B9_recalibration", "E2_rules", "E2_net_benefit"} else "revision_sensitivity"
        frame.to_csv(OUT / (name + ".csv"), index=False)
    (OUT / "stage2_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Directory containing frozen inputs and revision_20261116")
    args = parser.parse_args()
    ROOT = args.project_root.resolve()
    REVISION = ROOT / "revision_20261116"
    OUT = REVISION / "results" / "stage2"
    run()
