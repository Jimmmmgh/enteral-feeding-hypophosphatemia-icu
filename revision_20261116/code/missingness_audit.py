"""Read-only audit of the saved low-phosphate predictions for checkpoint 1."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedKFold

REVISION = Path(__file__).resolve().parents[1]
ROOT = REVISION.parent
SEED = 20260928
MODEL = "six_variable"
SOURCES = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "sicdb": "SICdb"}
LABS = ("potassium", "magnesium", "creatinine")
LABELS = {"potassium": "K", "magnesium": "Mg", "creatinine": "Cr", "bmi": "BMI"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(value):
    return f"{value}*" if value < 5 else str(value)


def predict_frozen(data, spec):
    columns = {}
    for name in spec["features"]:
        original = data[name].to_numpy(dtype=float)
        missing = ~np.isfinite(original)
        imputed = np.where(missing, spec["medians"][name], original)
        if name == "phosphate":
            a, b, c = spec["knots"]
            h1 = np.maximum(imputed - a, 0) ** 3
            h2 = np.maximum(imputed - b, 0) ** 3
            h3 = np.maximum(imputed - c, 0) ** 3
            columns[name] = imputed
            columns["phosphate_rcs"] = (h1 - h2 * (c - a) / (c - b) + h3 * (b - a) / (c - b)) / (c - a) ** 2
        elif name == "creatinine":
            columns["log_creatinine"] = np.log(imputed)
        else:
            columns[name] = imputed
        if name in spec["missing_flags"]:
            columns[name + "_missing"] = missing
    matrix = np.column_stack([columns[column] for column in spec["columns"]])
    standardized = (matrix - spec["mean"]) / spec["scale"]
    return expit(standardized @ spec["coef"] + spec["intercept"])


def load_analysis_cohort(source, spec, baseline, thresholds):
    private = ROOT / "private"
    data = pd.read_parquet(private / f"{source}_features.parquet")
    data = data.loc[data["followup_observed"]].reset_index(drop=True)
    predictions = pd.read_parquet(private / f"{source}_predictions.parquet")
    event = data["event"].to_numpy(dtype=int)
    score = predictions[MODEL].to_numpy(dtype=float)

    np.testing.assert_array_equal(predictions["event"], event)
    if not np.isfinite(score).all() or not np.isin(event, [0, 1]).all():
        raise ValueError(f"Invalid saved outcomes or predictions for {source}")
    if not np.array_equal(data[spec["features"]].isna(), ~np.isfinite(data[spec["features"]])):
        raise ValueError(f"Nonmissing infinite predictor in {source}")
    if source == "mimic":
        oof = pd.read_parquet(private / "mimic_oof_predictions.parquet")
        np.testing.assert_array_equal(oof["event"], event)
        np.testing.assert_array_equal(oof["hospital"], data["hospital"])
        np.testing.assert_array_equal(oof[MODEL], score)
    else:
        np.testing.assert_allclose(predict_frozen(data, spec), score, rtol=1e-13, atol=1e-14)

    reference = thresholds.loc[(source, 0.05)]
    counted = [len(data), sum(event), np.count_nonzero((event == 1) & (score < 0.05))]
    np.testing.assert_array_equal(counted, reference[["n", "events", "missed_events"]].astype(int))
    for variable in spec["features"]:
        frozen_missing = baseline.loc[(source, variable), "missing"]
        if data[variable].isna().sum() != frozen_missing:
            raise ValueError(f"Frozen missing count disagrees for {source}, {variable}")
    return data, event, score


def main():
    model_file = ROOT / "results/locked_models.json"
    freeze_file = ROOT / "audit/model_freeze.json"
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    spec = json.loads(model_file.read_text(encoding="utf-8"))[MODEL]
    raw_file = ROOT / "results/raw_equations.json"
    baseline_file = ROOT / "results/baseline.csv"
    threshold_file = ROOT / "results/thresholds.csv"
    implementation = ROOT / "scripts/analyze.py"
    protected_inputs = [model_file, freeze_file, raw_file, baseline_file, threshold_file, implementation]
    protected_inputs += [
        ROOT / "private" / f"{source}_{kind}.parquet"
        for source in SOURCES
        for kind in ("features", "predictions")
    ]
    protected_inputs.append(ROOT / "private/mimic_oof_predictions.parquet")
    before = {path: digest(path) for path in protected_inputs}
    if before[model_file] != freeze["model_sha256"]:
        raise ValueError("Locked model SHA-256 mismatch")
    if before[ROOT / "private/mimic_features.parquet"] != freeze["development_features_sha256"]:
        raise ValueError("Locked development features SHA-256 mismatch")
    if sklearn_version != freeze["sklearn"]:
        raise ValueError(f"Expected frozen scikit-learn version {freeze['sklearn']}")

    flags = (*LABS, "bmi")
    if set(flags) != set(spec["missing_flags"]):
        raise ValueError("Frozen model missing flags differ from the audit specification")
    baseline = pd.read_csv(baseline_file).set_index(["source", "variable"])
    thresholds = pd.read_csv(threshold_file).set_index(["source", "threshold"])
    k1 = []
    k2 = []
    for source, label in SOURCES.items():
        data, event, score = load_analysis_cohort(source, spec, baseline, thresholds)
        for variable in LABS:
            mask = data[variable].isna().to_numpy()
            subgroup = score[mask]
            interval = f"{subgroup.min():.6f}–{subgroup.max():.6f}" if len(subgroup) else "—"
            if 0 < len(subgroup) < 5:
                interval += "*"
            k1.append(f"| {label} | {LABELS[variable]} | {count(len(subgroup))} | {count(sum(event[mask]))} | {interval} |")

        missed = (event == 1) & (score < 0.05)
        missing = data.loc[missed, list(flags)].isna()
        missing_counts = [count(value) for value in missing.sum()]
        k2.append(
            f"| {label} | {count(sum(missed))} | {count(missing.any(axis=1).sum())} | "
            + " | ".join(missing_counts) + " |"
        )
        if source == "mimic":
            development = data
            development_event = event

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    k3 = []
    for fold, (train, _) in enumerate(splitter.split(development, development_event), 1):
        subset = development.iloc[train][spec["features"]]
        missing_counts = subset.isna().sum()
        kept = [name for name, n in missing_counts.items() if 0 < n < len(train)]
        absent = set(flags) - set(kept)
        added = set(kept) - set(flags)
        difference = ", ".join(sorted(absent | added)) or "无"
        items = " | ".join(count(missing_counts[name]) for name in flags)
        k3.append(f"| {fold} | {len(train)} | {items} | {difference} |")

    original_coef = np.asarray(spec["coef"]) / np.asarray(spec["scale"])
    original_intercept = spec["intercept"] - float(np.dot(spec["mean"], original_coef))
    frozen_equations = json.loads(raw_file.read_text(encoding="utf-8"))
    frozen_raw = next(item for item in frozen_equations if item["model"] == MODEL)
    np.testing.assert_allclose(original_intercept, frozen_raw["intercept"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        original_coef, [frozen_raw["coefficients"][name] for name in spec["columns"]], atol=1e-12, rtol=0
    )
    k4 = [f"| 截距 | {spec['intercept']:.9f} | {original_intercept:.9f} |"]
    k4.extend(
        f"| {name} | {standard:.9f} | {raw:.9f} |"
        for name, standard, raw in zip(spec["columns"], spec["coef"], original_coef, strict=True)
    )

    if any(digest(path) != before[path] for path in protected_inputs):
        raise ValueError("A protected input changed during the audit")
    lines = [
        "# 检查点 1 · 阶段 1 缺失指示变量核查",
        "",
        "输入为原工程已有的特征与预测文件。分析限于主分析人群（具有随访磷检测），MIMIC-IV 使用已保存的 OOF 预测，eICU-CRD 与 SICdb 使用已保存的外部预测。外部预测逐项与冻结模型的线性预测计算核对。未拟合模型或查库。",
        "",
        "K1 各变量分别统计，缺失子集可重叠。K2 的任一缺失指标为冻结六变量模型的 Cr、Mg、K 或 BMI 指示变量，标记规则为预测概率 ≥0.05。* 表示计数低于 5 或范围基于少于 5 人。空子集无概率范围。",
        "",
        "## K1 基线实验室变量缺失",
        "",
        "| 队列 | 缺失变量 | 人数 | 其中事件数 | 六变量模型预测概率范围 |",
        "|---|---|---:|---:|---:|",
        *k1,
        "",
        "缺失人数逐项与冻结 results/baseline.csv 核对一致。事件数和概率范围为本轮额外聚合核查。",
        "",
        "## K2 六变量模型 5% 阈值未标记事件",
        "",
        "| 队列 | 未标记事件 | 任一缺失指示为 1 | K 缺失 | Mg 缺失 | Cr 缺失 | BMI 缺失 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *k2,
        "",
        "三库未标记事件数与冻结 results/thresholds.csv 的 5% 行一致。",
        "",
        "## K3 MIMIC-IV 五个训练折中的缺失人数",
        "",
        "| 训练折 | 人数 | K | Mg | Cr | BMI | 与最终模型指示变量集合的差异 |",
        "|---|---:|---:|---:|---:|---:|---|",
        *k3,
        "",
        "按原文件顺序及原代码 StratifiedKFold(n_splits=5, shuffle=True, random_state=20260928) 重建训练折。四项指示在五折内均有变异。原 fit() 仅在训练折内同时存在缺失和非缺失值时生成对应指示列，因此常数缺失指示列会被省略。对于实际进入设计矩阵的列，代码把为 0 的标准差替换为 1。磷和年龄在整个 MIMIC-IV 主分析人群中无缺失，均未生成指示列。",
        "",
        "## K4 冻结六变量模型系数",
        "",
        "| 项 | 标准化尺度 | 原始变换后尺度 |",
        "|---|---:|---:|",
        *k4,
        "",
        "原始尺度保留磷样条项和 ln(Cr)，仅移除中心化和标准化。换算系数为标准化系数除以训练集标准差。换算截距为标准化截距减去各换算系数与相应训练均值的乘积之和。所有项与冻结 results/raw_equations.json 的差绝对值不超过 1e-12。",
        "",
    ]
    (REVISION / "checkpoint1_stage1_k1_k4.md").write_text("\n".join(lines), encoding="utf-8")
    print("Checkpoint 1 K1–K4 aggregate audit written; frozen inputs unchanged.")


if __name__ == "__main__":
    main()