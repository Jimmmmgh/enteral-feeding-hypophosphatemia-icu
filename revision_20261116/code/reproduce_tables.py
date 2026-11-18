"""Reproduce checkpoint 1 tables using only frozen local artifacts."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REVISION = Path(__file__).resolve().parents[1]
ROOT = REVISION.parent
SOURCE_LABELS = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "sicdb": "SICdb"}
MODEL_LABELS = {
    "phosphate_only": "Phosphate",
    "phosphate_creatinine": "Phosphate + creatinine",
    "six_variable": "Six variables",
}
METRICS = ("auc", "brier", "observed_expected")
THRESHOLDS = (.05, .10, .20)
SEED = 20260928
BOOTSTRAPS = 500


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p, w):
    return np.array([
        roc_auc_score(y, p, sample_weight=w),
        np.average((y - p) ** 2, weights=w),
        np.sum(w * y) / np.sum(w * p),
    ])


def threshold_counts(y, p, threshold, w):
    flagged = p >= threshold
    tp = np.sum(w * y * flagged)
    fp = np.sum(w * (1 - y) * flagged)
    fn = np.sum(w * y * ~flagged)
    tn = np.sum(w * (1 - y) * ~flagged)
    return {
        "n_flagged": int(tp + fp), "n_unflagged": int(fn + tn),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp), "npv": tn / (tn + fn),
        "fraction_flagged": (tp + fp) / np.sum(w), "missed_events": int(fn),
        "low_risk_event_rate": fn / (fn + tn),
        "net_benefit": (tp - fp * threshold / (1 - threshold)) / np.sum(w),
    }


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def classify_values(text):
    return [token.replace(",", "") for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("−", "-"))]


def matches_display(actual, displayed):
    decimals = len(displayed.split(".")[1]) if "." in displayed else 0
    return f"{actual:.{decimals}f}" == displayed


def read_word_table(path, table_name, header):
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    matched = []
    for table in document.find(namespace + "body").findall(namespace + "tbl"):
        rows = [
            ["".join(node.text or "" for node in cell.iter(namespace + "t")).strip()
             for cell in row.findall(namespace + "tc")]
            for row in table.findall(namespace + "tr")
        ]
        if rows and rows[0][:len(header)] == header:
            matched.append(rows[1:])
    ensure(len(matched) == 1 and matched[0], f"Expected one {table_name} in {path.name}")
    return matched[0]


def compare_word(report, manuscript, supplement):
    for table_name, path, key, header, display in (
        ("Table 2", manuscript, "table_2", ["Cohort", "Model", "AUROC (95% CI)"], lambda r: [
            [r["auc"][k] for k in ("estimate", "ci_low", "ci_high")],
            [r["brier"]["estimate"]],
            [r["observed_expected"][k] for k in ("estimate", "ci_low", "ci_high")],
        ]),
        ("Table 3", manuscript, "table_3", ["Cohort", "Risk threshold"], lambda r: [
            [100 * r[k]] for k in ("fraction_flagged", "sensitivity", "npv")
        ] + [[r["missed_events"]]]),
        ("Table S3", supplement, "table_s3", ["Cohort", "Comparator"], lambda r: [
            [r[metric][k] for k in ("estimate", "ci_low", "ci_high")]
            for metric in ("auc", "brier")
        ]),
    ):
        rows = read_word_table(path, table_name, header)
        ensure(len(rows) == len(report[key]), f"{table_name} row count differs in {path.name}")
        for row, expected in zip(rows, report[key], strict=True):
            ensure(row[0] == SOURCE_LABELS[expected["source"]], f"{table_name} cohort label differs")
            comparison = (
                f'{expected["threshold"]:.0%}' if key == "table_3"
                else MODEL_LABELS[expected["model"]]
            )
            ensure(row[1].lower() == comparison.lower(), f"{table_name} model or threshold label differs")
            for values, cell in zip(display(expected), row[2:], strict=True):
                tokens = classify_values(cell)
                ensure(len(values) == len(tokens), f"{table_name} display count differs")
                ensure(
                    all(matches_display(v, tok) for v, tok in zip(values, tokens, strict=True)),
                    f"{table_name} {row[:2]} discrepancy: computed {values}, draft {cell}",
                )
    rows = read_word_table(supplement, "Table S1", ["Selection or outcome"])
    counts = {source: list(report["table_s1"][source].values()) for source in SOURCE_LABELS}
    ensure(len(rows) == len(counts["mimic"]), f"Table S1 row count differs in {supplement.name}")
    for i, row in enumerate(rows):
        for source, cell in zip(SOURCE_LABELS, row[1:], strict=True):
            count = counts[source][i]
            ensure(cell.replace(",", "") == str(count), f"Table S1 {source} {row[0]} differs: {count} vs {cell}")


def entries(point, distribution):
    low, high = np.quantile(distribution, [.025, .975], axis=0)
    return {
        metric: {"estimate": float(estimate), "ci_low": float(lo), "ci_high": float(hi)}
        for metric, estimate, lo, hi in zip(METRICS, point, low, high, strict=True)
    }


def close(actual, expected, label):
    ensure(
        abs(actual - expected) < 1e-11,
        f"{label} differs: computed {actual:.12g}, frozen {expected:.12g}",
    )


def verify_metrics(row, reference, metrics_subset):
    for metric in metrics_subset:
        ref = reference.loc[metric]
        for key, value in row[metric].items():
            close(value, ref[key], f'{row["source"]} {row["model"]} {metric} {key}')


def reproduce():
    frozen_performance = pd.read_csv(ROOT / "results/performance.csv")
    frozen_differences = pd.read_csv(ROOT / "results/model_differences.csv")
    frozen_thresholds = pd.read_csv(ROOT / "results/thresholds.csv")
    frozen_flow = pd.read_csv(ROOT / "results/flow.csv")
    report = {
        "table_2": [], "table_3": [], "table_s3": [], "table_s1": {},
        "table_s1_provenance": {
            "row_1_to_6": "Frozen extraction audit counts reconciled against flow.csv. Excluded rows are unavailable in retained features.",
            "row_7_to_11": "Recounted using baseline nadir, follow-up test count, follow-up nadir and surveillance timing in frozen features. Reconciled against extraction audit and flow.csv.",
        },
        "table_s1_small_cells": [],
        "table_3_small_cells": [],
        "bootstrap": {"seed": SEED, "requested": BOOTSTRAPS, "valid_by_source": {}},
    }
    for source in SOURCE_LABELS:
        private = ROOT / "private"
        feature_file = private / f"{source}_features.parquet"
        extraction = json.loads((ROOT / f"audit/{source}_extraction.json").read_text())
        ensure(sha256(feature_file) == extraction["feature_file_sha256"], f"{source} feature hash differs")
        columns = [
            "followup_observed", "event", "hospital", "phosphate_nadir", "daily_surveillance",
            "pre_phosphate_min", "followup_tests", "first_followup_hours", "last_followup_hours",
            "end_hours", "index_hours",
        ]
        features = pd.read_parquet(feature_file, columns=columns)
        eligible_baseline = features["pre_phosphate_min"].ge(2)
        has_followup = features["followup_tests"].gt(0)
        ensure(eligible_baseline.all(), f"{source} contains patients with ineligible baseline")
        ensure(has_followup.equals(features["followup_observed"]), f"{source} follow-up flag differs")
        observed = features.loc[has_followup]
        daily_surveillance = (
            (observed["end_hours"] - observed["index_hours"]).ge(48)
            & observed["first_followup_hours"].le(24)
            & observed["last_followup_hours"].gt(24)
        )
        ensure(daily_surveillance.equals(observed["daily_surveillance"]), f"{source} daily surveillance flag differs")
        y = observed["event"].to_numpy(dtype=int)
        ensure(np.array_equal(y, observed["phosphate_nadir"].lt(2)), f"{source} primary outcome differs")
        source_counts = frozen_flow.loc[frozen_flow.source.eq(source)].set_index("stage")["n"].to_dict()
        ensure(source_counts == extraction["flow"], f"{source} extraction and frozen flow differ")
        checked_counts = {
            "no_known_phosphate_below_2_in_previous_24h": int(eligible_baseline.sum()),
            "followup_phosphate_observed": len(y),
            "primary_events": int(y.sum()),
            "severe_events": int(observed["phosphate_nadir"].lt(1).sum()),
            "full_48h_daily_surveillance": int(daily_surveillance.sum()),
        }
        for key, value in checked_counts.items():
            ensure(source_counts[key] == value, f"{source} S1 {key}: {value} vs {source_counts[key]}")
        report["table_s1"][source] = source_counts
        for stage, count in source_counts.items():
            if count < 5:
                report["table_s1_small_cells"].append({"source": source, "stage": stage, "count": count})
        scores = pd.read_parquet(private / f"{source}_predictions.parquet")
        ensure(np.array_equal(scores["event"], y), f"{source} stored outcome alignment differs")
        if source == "mimic":
            oof = pd.read_parquet(private / "mimic_oof_predictions.parquet")
            ensure(
                np.array_equal(oof[["event", *MODEL_LABELS]], scores[["event", *MODEL_LABELS]]),
                "MIMIC analysis predictions differ from frozen out-of-fold predictions",
            )
            ensure(np.array_equal(oof["hospital"], observed["hospital"]), "MIMIC OOF hospital alignment differs")
        ensure(
            np.isfinite(scores[list(MODEL_LABELS)].to_numpy()).all()
            and ((scores[list(MODEL_LABELS)] > 0) & (scores[list(MODEL_LABELS)] < 1)).all().all(),
            f"{source} stored scores are invalid",
        )
        if source == "eicu":
            _, groups = np.unique(observed["hospital"], return_inverse=True)
            ensure(groups.max() + 1 == 41, "eICU cluster count differs")
        else:
            groups = np.arange(len(y))
        n_units = int(groups.max() + 1)
        predicts = [scores[m].to_numpy() for m in MODEL_LABELS]
        rng = np.random.default_rng(SEED)
        unit_weights = np.ones(len(y))
        point = np.stack([metrics(y, p, unit_weights) for p in predicts])
        boot = []
        for _ in range(BOOTSTRAPS):
            w = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)[groups].astype(float)
            if w @ y == 0 or w @ (1 - y) == 0:
                continue
            boot.append([metrics(y, p, w) for p in predicts])
        boot = np.array(boot)
        report["bootstrap"]["valid_by_source"][source] = len(boot)
        for model_i, model in enumerate(MODEL_LABELS):
            ref = frozen_performance.query("source == @source and model == @model").set_index("metric")
            ensure(
                (ref.n == len(y)).all() and (ref.events == int(y.sum())).all()
                and (ref.bootstrap_valid == len(boot)).all(),
                f"{source} {model} cohort or bootstrap count differs",
            )
            row = {"source": source, "model": model, **entries(point[model_i], boot[:, model_i])}
            verify_metrics(row, ref, METRICS)
            report["table_2"].append(row)
        for model_i, model in enumerate(tuple(MODEL_LABELS)[:2]):
            comparison = "six_variable_minus_" + model
            ref = frozen_differences.query("source == @source and comparison == @comparison").set_index("metric")
            results = entries(point[2] - point[model_i], boot[:, 2] - boot[:, model_i])
            row = {"source": source, "model": model, **{k: results[k] for k in ("auc", "brier")}}
            verify_metrics(row, ref, ("auc", "brier"))
            report["table_s3"].append(row)
        for threshold in THRESHOLDS:
            ref = frozen_thresholds.query("source == @source and threshold == @threshold").iloc[0]
            counts = threshold_counts(y, predicts[2], threshold, unit_weights)
            ensure(ref.n == len(y) and ref.events == y.sum(), f"{source} threshold cohort count differs")
            for key in counts.keys() & set(ref.index):
                close(counts[key], ref[key], f"{source} Table 3 {threshold} {key}")
            report["table_3"].append({"source": source, "threshold": threshold, **counts})
            if counts["missed_events"] < 5:
                report["table_3_small_cells"].append({
                    "source": source, "threshold": threshold, "missed_events": counts["missed_events"],
                })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True, help="Local comparator DOCX")
    parser.add_argument("--supplement", type=Path, required=True, help="Local comparator DOCX")
    args = parser.parse_args()
    report = reproduce()
    compare_word(report, args.manuscript, args.supplement)
    report["reference"] = {
        "status": "Provisional local 12 October comparator. The exact 2026682 files were unavailable.",
        "manuscript": str(args.manuscript),
        "supplement": str(args.supplement),
        "verified": ["Table 2", "Table 3", "Table S1", "Table S3"],
        "comparison_precision": "Frozen CSV at 1e-11 absolute tolerance. Word at printed precision with full computed O:E retained.",
    }
    output = REVISION / "checkpoint1_stage0_tables.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified the frozen aggregate files and four provisional local tables. Output: {output}")


if __name__ == "__main__":
    main()

