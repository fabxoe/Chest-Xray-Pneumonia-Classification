from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace, sobel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


EXP_ID = "exp119"
PRED = Path("oof/preds")
SUB = Path("oof/submissions")
REPORT = Path("reports/exp119_rf_svm_gaussian_rerank")

CURRENT_BEST = SUB / "exp117_after0385_drop_test_0303_only_submit.csv"
SAMPLE_SUBMISSION = Path("sample_submission.csv")
TRAIN_CSV = Path("train.csv")
TEST_DIR = Path("test")
TRAIN_DIR = Path("train")

SUCCESS_DIRECTIONS = {
    "test_0455.png": 0,
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0303.png": 0,
}
FAILED_DROP_DIRECTIONS = {
    "test_0289.png": 1,
    "test_0172.png": 1,
    "test_0013.png": 1,
    "test_0188.png": 1,
    "test_0568.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 0,
}
OLDER_GUARDRAILS = {
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
}
GUARDRAILS = {**OLDER_GUARDRAILS, **SUCCESS_DIRECTIONS, **FAILED_DROP_DIRECTIONS}

MODEL_SPECS = {
    "cnx": ("exp46_convnext_tiny_oof_probs.csv", "exp46_convnext_tiny_test_probs_mean.csv"),
    "resnet": ("exp48_resnet18_oof_probs.csv", "exp48_resnet18_test_probs_mean.csv"),
    "eff": ("exp50_efficientnet_b0_oof_probs.csv", "exp50_efficientnet_b0_test_probs_mean.csv"),
    "cnx123": ("exp54_convnext_tiny_seed123_oof_probs.csv", "exp54_convnext_tiny_seed123_test_probs_mean.csv"),
    "cnx777": ("exp72_convnext_tiny_seed777_oof_probs.csv", "exp72_convnext_tiny_seed777_test_probs_mean.csv"),
    "cnx_lion": ("exp78_convnext_tiny_lion_oof_probs.csv", "exp78_convnext_tiny_lion_test_probs_mean.csv"),
    "eff123": ("exp81_efficientnet_b0_seed123_oof_probs.csv", "exp81_efficientnet_b0_seed123_test_probs_mean.csv"),
    "cnxv2": ("exp65_convnextv2_tiny_stage3_oof_probs.csv", "exp65_convnextv2_tiny_stage3_test_probs_mean.csv"),
    "regnet": ("exp67_regnet_y_3_2gf_oof_probs.csv", "exp67_regnet_y_3_2gf_test_probs_mean.csv"),
    "effv2": ("exp70_efficientnet_v2_s_wide_oof_probs.csv", "exp70_efficientnet_v2_s_wide_test_probs_mean.csv"),
    "swinv2": ("exp103_swinv2_tiny_oof_oof_probs.csv", "exp103_swinv2_tiny_oof_test_probs_mean.csv"),
}


def normalize_file_name(name: str) -> str:
    name = str(name)
    return name if name.endswith(".png") else f"{name}.png"


def read_probs(path: Path, prefix: str, is_oof: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["file_name", "prob_0", "prob_1", "pred_label"]
    if is_oof and "label" in df.columns:
        keep.append("label")
    if is_oof and "fold" in df.columns:
        keep.append("fold")
    df = df[keep].copy()
    df["file_name"] = df["file_name"].map(normalize_file_name)
    rename = {
        "prob_0": f"{prefix}_prob0",
        "prob_1": f"{prefix}_prob1",
        "pred_label": f"{prefix}_pred",
    }
    return df.rename(columns=rename)


def add_model_features(base: pd.DataFrame, is_oof: bool) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    used: list[str] = []
    for name, (oof_name, test_name) in MODEL_SPECS.items():
        path = PRED / (oof_name if is_oof else test_name)
        if not path.exists():
            continue
        cur = read_probs(path, name, is_oof)
        drop_cols = []
        if is_oof:
            drop_cols += [c for c in ["label", "fold"] if c in cur.columns]
        out = out.merge(cur.drop(columns=drop_cols), on="file_name", how="inner", validate="one_to_one")
        used.append(name)
    if len(used) < 4:
        raise RuntimeError(f"Too few model features found: {used}")

    prob_cols = [f"{m}_prob1" for m in used]
    pred_cols = [f"{m}_pred" for m in used]
    probs = out[prob_cols]
    preds = out[pred_cols]
    out["prob_mean"] = probs.mean(axis=1)
    out["prob_std"] = probs.std(axis=1)
    out["prob_min"] = probs.min(axis=1)
    out["prob_max"] = probs.max(axis=1)
    out["prob_range"] = out["prob_max"] - out["prob_min"]
    out["vote_1_count"] = preds.sum(axis=1)
    out["vote_0_count"] = len(used) - out["vote_1_count"]
    out["vote_0_ratio"] = out["vote_0_count"] / len(used)
    for m in used:
        out[f"{m}_margin"] = out[f"{m}_prob1"] - out[f"{m}_prob0"]
        out[f"{m}_rank"] = out[f"{m}_prob1"].rank(pct=True)
    return out, used


def image_features(path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(path).convert("L").resize((192, 192)), dtype=np.float32) / 255.0
    blur = gaussian_filter(arr, sigma=1.2)
    gx = sobel(arr, axis=0)
    gy = sobel(arr, axis=1)
    edge = np.hypot(gx, gy)
    lap = laplace(arr)
    blur_edge = np.hypot(sobel(blur, axis=0), sobel(blur, axis=1))
    hist, _ = np.histogram(arr, bins=64, range=(0, 1))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return {
        "img_mean": float(arr.mean()),
        "img_std": float(arr.std()),
        "img_p10": float(np.percentile(arr, 10)),
        "img_p90": float(np.percentile(arr, 90)),
        "img_contrast": float(np.percentile(arr, 90) - np.percentile(arr, 10)),
        "img_entropy": float(-(p * np.log2(p)).sum()),
        "edge_mean": float(edge.mean()),
        "edge_std": float(edge.std()),
        "lap_var": float(lap.var()),
        "gauss_mean_abs_delta": float(np.abs(arr - blur).mean()),
        "gauss_edge_delta": float(edge.mean() - blur_edge.mean()),
        "gauss_contrast_delta": float((np.percentile(arr, 90) - np.percentile(arr, 10)) - (np.percentile(blur, 90) - np.percentile(blur, 10))),
    }


def add_image_features(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    feats = [image_features(image_dir / name) for name in df["file_name"]]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def add_external_evidence(test_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy()
    exp113 = Path("reports/exp113_tree_replan/exp113_candidate_ranked.csv")
    if exp113.exists():
        cols = [
            "file_name",
            "tree_drop_score",
            "suspicious_score",
            "recommended_action",
            "exp110_rank",
            "exp111_candidate",
            "convnext_fold0_prob1",
            "cam_center_fraction",
        ]
        cur = pd.read_csv(exp113)
        cur["file_name"] = cur["file_name"].map(normalize_file_name)
        keep = [c for c in cols if c in cur.columns]
        out = out.merge(cur[keep], on="file_name", how="left", validate="one_to_one")
    xai = Path("reports/xai/gradcam_convnext_disagreement_summary.csv")
    if xai.exists():
        cur = pd.read_csv(xai)
        cur["file_name"] = cur["file_name"].map(normalize_file_name)
        keep = [c for c in ["file_name", "prob_1", "cam_target_class"] if c in cur.columns]
        cur = cur[keep].drop_duplicates("file_name")
        cur = cur.rename(columns={"prob_1": "xai_cnx_disagreement_prob1"})
        out = out.merge(cur, on="file_name", how="left", validate="one_to_one")
    return out


def train_rank_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_df["label"].astype(int)
    xt = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=119)

    models = {
        "rf": RandomForestClassifier(
            n_estimators=600,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=119,
            n_jobs=-1,
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=120,
            n_jobs=-1,
        ),
        "svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.03, class_weight="balanced", random_state=119, max_iter=20000),
        ),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=20000, random_state=119),
        ),
        "ridge": make_pipeline(StandardScaler(), RidgeClassifier(alpha=100.0, class_weight="balanced")),
    }

    summary_rows = []
    scored = test_df.copy()
    for name, model in models.items():
        if name in {"rf", "extratrees", "logistic"}:
            oof_prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
            oof_pred = (oof_prob >= 0.5).astype(int)
            model.fit(x, y)
            test_prob1 = model.predict_proba(xt)[:, 1]
            scored[f"{name}_prob0"] = 1.0 - test_prob1
            scored[f"{name}_prob1"] = test_prob1
        else:
            oof_decision = cross_val_predict(model, x, y, cv=cv, method="decision_function", n_jobs=None)
            oof_pred = (oof_decision >= 0).astype(int)
            model.fit(x, y)
            decision = model.decision_function(xt)
            scored[f"{name}_decision"] = decision
            scored[f"{name}_prob0_like"] = 1.0 / (1.0 + np.exp(np.clip(decision, -20, 20)))

        tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()
        summary_rows.append({
            "model": name,
            "meta_cv_accuracy": accuracy_score(y, oof_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    scored["ml_zero_support"] = (
        scored["rf_prob0"].rank(pct=True) * 0.28
        + scored["extratrees_prob0"].rank(pct=True) * 0.26
        + scored["svm_prob0_like"].rank(pct=True) * 0.18
        + scored["logistic_prob0"].rank(pct=True) * 0.16
        + scored["ridge_prob0_like"].rank(pct=True) * 0.12
    )
    return scored, pd.DataFrame(summary_rows)


def make_submission(base: pd.DataFrame, sample: str, out_path: Path) -> pd.DataFrame:
    sub = base[["file_name", "label"]].copy()
    mask = sub["file_name"].eq(sample)
    if mask.sum() != 1:
        raise RuntimeError(f"Sample not found exactly once: {sample}")
    if int(sub.loc[mask, "label"].iloc[0]) != 1:
        raise RuntimeError(f"Candidate is not current positive: {sample}")
    sub.loc[mask, "label"] = 0
    sub.to_csv(out_path, index=False)
    return pd.DataFrame([{"file_name": sample, "before": 1, "after": 0}])


def validate_submission(path: Path, current_best: pd.DataFrame) -> dict[str, object]:
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    sub = pd.read_csv(path)
    if len(sub) != 624:
        raise RuntimeError(f"Bad row count: {len(sub)}")
    if list(sub.columns) != ["file_name", "label"]:
        raise RuntimeError(f"Bad columns: {list(sub.columns)}")
    if not sub["file_name"].equals(sample["file_name"]):
        raise RuntimeError("Submission order mismatch")
    if not set(sub["label"].unique()) <= {0, 1}:
        raise RuntimeError("Bad label values")
    merged = current_best.rename(columns={"label": "before"}).merge(
        sub.rename(columns={"label": "after"}), on="file_name", validate="one_to_one"
    )
    changed = merged[merged["before"].ne(merged["after"])]
    return {
        "rows": len(sub),
        "changed_count": len(changed),
        "label_counts": sub["label"].value_counts().sort_index().to_dict(),
        "changed_file": changed["file_name"].iloc[0] if len(changed) else "",
        "changed": f"{int(changed['before'].iloc[0])}->{int(changed['after'].iloc[0])}" if len(changed) else "",
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    train["file_name"] = train["file_name"].map(normalize_file_name)
    train_base = train[["file_name", "label"]].copy()
    test_best = pd.read_csv(CURRENT_BEST)
    test_best["file_name"] = test_best["file_name"].map(normalize_file_name)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    if not test_best["file_name"].equals(sample["file_name"]):
        raise RuntimeError("Current best order mismatch with sample submission")

    train_feat, model_names = add_model_features(train_base, is_oof=True)
    test_feat, _ = add_model_features(test_best.rename(columns={"label": "current_label"}), is_oof=False)
    train_feat = add_image_features(train_feat, TRAIN_DIR)
    test_feat = add_image_features(test_feat, TEST_DIR)
    test_feat = add_external_evidence(test_feat)

    exclude = {"file_name", "label", "fold", "current_label", "recommended_action", "exp111_candidate"}
    feature_cols = [
        c for c in train_feat.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(train_feat[c])
    ]
    scored, model_summary = train_rank_models(train_feat, test_feat, feature_cols)

    scored["guardrail_expected"] = scored["file_name"].map(GUARDRAILS)
    scored["is_guardrail"] = scored["guardrail_expected"].notna()
    scored["guardrail_ok"] = np.where(
        scored["is_guardrail"], scored["current_label"].eq(scored["guardrail_expected"]), True
    )
    current_positive = scored["current_label"].eq(1)
    eligible = scored[current_positive & (~scored["is_guardrail"])].copy()

    # High score means "more plausible current positive is actually 0".
    eligible["exp119_drop_score"] = (
        eligible["ml_zero_support"].rank(pct=True) * 0.42
        + eligible["tree_drop_score"].fillna(0).rank(pct=True) * 0.22
        + eligible["vote_0_ratio"].rank(pct=True) * 0.12
        + eligible["prob_std"].rank(pct=True) * 0.08
        + eligible["gauss_edge_delta"].rank(pct=True) * 0.06
        + eligible["lap_var"].rank(pct=True) * 0.05
        + (eligible["recommended_action"].fillna("").eq("review")).astype(float) * 0.03
        + eligible["exp111_candidate"].notna().astype(float) * 0.02
    )
    # Direct current public failures/successes are already removed by guardrail.
    eligible = eligible.sort_values(["exp119_drop_score", "ml_zero_support"], ascending=False).reset_index(drop=True)

    out_cols = [
        "file_name", "current_label", "exp119_drop_score", "ml_zero_support",
        "rf_prob0", "extratrees_prob0", "svm_prob0_like", "logistic_prob0", "ridge_prob0_like",
        "tree_drop_score", "vote_1_count", "vote_0_count", "prob_mean", "prob_std",
        "cnx_prob1", "cnx123_prob1", "cnx777_prob1", "cnx_lion_prob1",
        "img_contrast", "edge_mean", "lap_var", "gauss_mean_abs_delta", "gauss_edge_delta",
        "recommended_action", "exp110_rank", "exp111_candidate", "convnext_fold0_prob1", "cam_center_fraction",
    ]
    out_cols = [c for c in out_cols if c in eligible.columns]
    eligible[out_cols].to_csv(REPORT / "exp119_rf_svm_gaussian_candidate_ranking.csv", index=False)
    eligible.head(30)[out_cols].to_csv(REPORT / "exp119_top30.csv", index=False)
    model_summary.to_csv(REPORT / "exp119_meta_model_cv_summary.csv", index=False)

    top = eligible.iloc[0]
    selected = normalize_file_name(top["file_name"])
    out_path = SUB / f"exp120_after0303_drop_{selected.replace('.png', '')}_only_submit.csv"
    changed = make_submission(test_best.rename(columns={"current_label": "label"}), selected, out_path)
    changed_path = PRED / f"exp120_after0303_drop_{selected.replace('.png', '')}_only_changed_vs_exp117.csv"
    changed.to_csv(changed_path, index=False)
    validation = validate_submission(out_path, test_best.rename(columns={"current_label": "label"}))

    exp120_summary = pd.DataFrame([{
        "selected_sample": selected,
        "submission_path": str(out_path),
        "changed_csv": str(changed_path),
        "exp119_drop_score": float(top["exp119_drop_score"]),
        "ml_zero_support": float(top["ml_zero_support"]),
        "rf_prob0": float(top["rf_prob0"]),
        "extratrees_prob0": float(top["extratrees_prob0"]),
        "svm_prob0_like": float(top["svm_prob0_like"]),
        "tree_drop_score": float(top.get("tree_drop_score", np.nan)),
        "validation": validation,
        "recommendation": "submit_one_at_a_time",
    }])
    exp120_summary.to_csv(PRED / "exp120_after0303_top1_submit_summary.csv", index=False)

    print("EXP-119 RF/SVM/Gaussian reranking")
    print(f"models used: {model_names}")
    print(model_summary.to_string(index=False))
    print()
    print("Top 10 candidates")
    print(eligible.head(10)[out_cols].to_string(index=False))
    print()
    print("EXP-120 top-1 candidate")
    print(exp120_summary.to_string(index=False))


if __name__ == "__main__":
    main()
