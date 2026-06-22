from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


EXP_ID = "exp58"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_BEST_PATH = PRED_DIR / "exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv"
CURRENT_BEST_META_CV_ACCURACY = 0.9865804597701149
META_CV_TOLERANCE = 0.0015

MODELS = {
    "cnx": {
        "oof": PRED_DIR / "exp46_convnext_tiny_oof_probs.csv",
        "test": PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv",
    },
    "cnx123": {
        "oof": PRED_DIR / "exp54_convnext_tiny_seed123_oof_probs.csv",
        "test": PRED_DIR / "exp54_convnext_tiny_seed123_test_probs_mean.csv",
    },
    "eff": {
        "oof": PRED_DIR / "exp50_efficientnet_b0_oof_probs.csv",
        "test": PRED_DIR / "exp50_efficientnet_b0_test_probs_mean.csv",
    },
}

PROTECTED = {
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
}

THRESHOLDS = [0.40, 0.425, 0.45, 0.475, 0.49, 0.50, 0.51, 0.525, 0.55, 0.575, 0.60]


def add_model_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_rank"] = out[f"{prefix}_prob1"].rank(pct=True)
    out[f"{prefix}_margin"] = out[f"{prefix}_prob1"] - out[f"{prefix}_prob0"]
    out[f"{prefix}_uncertainty"] = 1.0 - np.abs(out[f"{prefix}_margin"])
    return out


def load_oof_and_test():
    oof_base = None
    test_base = None
    for name, paths in MODELS.items():
        oof = pd.read_csv(paths["oof"])
        test = pd.read_csv(paths["test"])
        oof_cur = oof[["file_name", "label", "fold", "prob_0", "prob_1", "pred_label"]].rename(
            columns={"prob_0": f"{name}_prob0", "prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"}
        )
        test_cur = test[["file_name", "prob_0", "prob_1", "pred_label"]].rename(
            columns={"prob_0": f"{name}_prob0", "prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"}
        )
        if oof_base is None:
            oof_base = oof_cur
        else:
            oof_base = oof_base.merge(
                oof_cur.drop(columns=["label", "fold"]),
                on="file_name",
                how="inner",
                validate="one_to_one",
            )
        test_base = test_cur if test_base is None else test_base.merge(test_cur, on="file_name", how="inner", validate="one_to_one")

    for name in MODELS:
        oof_base = add_model_features(oof_base, name)
        test_base = add_model_features(test_base, name)

    for df in [oof_base, test_base]:
        probs = df[[f"{name}_prob1" for name in MODELS]]
        margins = df[[f"{name}_margin" for name in MODELS]]
        df["prob_mean_all"] = probs.mean(axis=1)
        df["prob_std_all"] = probs.std(axis=1)
        df["prob_min_all"] = probs.min(axis=1)
        df["prob_max_all"] = probs.max(axis=1)
        df["margin_mean_all"] = margins.mean(axis=1)
        df["margin_std_all"] = margins.std(axis=1)
        df["vote_sum_all"] = df[[f"{name}_pred" for name in MODELS]].sum(axis=1)
    return oof_base.sort_values("file_name").reset_index(drop=True), test_base.sort_values("file_name").reset_index(drop=True)


def feature_sets():
    specs = {}
    model_names = list(MODELS)
    for size in [2, 3]:
        for subset in combinations(model_names, size):
            tag = "_".join(subset)
            probs = [f"{m}_prob1" for m in subset]
            ranks = [f"{m}_rank" for m in subset]
            margins = [f"{m}_margin" for m in subset]
            uncertainties = [f"{m}_uncertainty" for m in subset]
            preds = [f"{m}_pred" for m in subset]
            specs[f"{tag}_prob_only"] = probs
            specs[f"{tag}_prob_rank"] = probs + ranks
            specs[f"{tag}_prob_margin"] = probs + margins
            specs[f"{tag}_compact"] = probs + ranks + margins + uncertainties + preds
    specs["all_prob_stats"] = [
        "cnx_prob1",
        "cnx123_prob1",
        "eff_prob1",
        "prob_mean_all",
        "prob_std_all",
        "prob_min_all",
        "prob_max_all",
    ]
    specs["all_margin_stats"] = [
        "cnx_margin",
        "cnx123_margin",
        "eff_margin",
        "margin_mean_all",
        "margin_std_all",
        "vote_sum_all",
    ]
    return specs


def make_base(kind: str, value: float):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=value))
    if kind == "svc":
        return make_pipeline(StandardScaler(), LinearSVC(C=value, dual="auto", max_iter=5000))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=value, solver="liblinear", max_iter=1000))
    raise ValueError(kind)


def make_calibrated(kind: str, value: float, method: str):
    return CalibratedClassifierCV(estimator=make_base(kind, value), method=method, cv=3, ensemble=True)


def meta_cv_predict(oof: pd.DataFrame, cols: list[str], kind: str, value: float, method: str):
    y = oof["label"].to_numpy()
    folds = oof["fold"].to_numpy()
    prob = np.zeros(len(oof), dtype=float)
    for fold in sorted(oof["fold"].unique()):
        train_idx = folds != fold
        valid_idx = folds == fold
        clf = make_calibrated(kind, value, method)
        clf.fit(oof.loc[train_idx, cols], y[train_idx])
        prob[valid_idx] = clf.predict_proba(oof.loc[valid_idx, cols])[:, 1]
    return prob


def final_test_predict(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], kind: str, value: float, method: str):
    clf = make_calibrated(kind, value, method)
    clf.fit(oof[cols], oof["label"].to_numpy())
    return clf.predict_proba(test[cols])[:, 1]


def metric_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "meta_cv_accuracy": accuracy_score(y_true, y_pred),
        "meta_cv_precision": precision_score(y_true, y_pred, zero_division=0),
        "meta_cv_recall": recall_score(y_true, y_pred, zero_division=0),
        "meta_cv_f1": f1_score(y_true, y_pred, zero_division=0),
        "meta_cv_tn": int(tn),
        "meta_cv_fp": int(fp),
        "meta_cv_fn": int(fn),
        "meta_cv_tp": int(tp),
    }


def add_common(row, test, test_pred, test_prob, exp26_pred, current_best_pred):
    row["test_label0"] = int(np.sum(test_pred == 0))
    row["test_label1"] = int(np.sum(test_pred == 1))
    row["changed_vs_exp26"] = int(np.sum(test_pred != exp26_pred))
    row["changed_vs_current_best"] = int(np.sum(test_pred != current_best_pred))
    for file_name in PROTECTED:
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        tag = file_name.replace(".png", "")
        row[f"{tag}_pred"] = int(test_pred[idx])
        row[f"{tag}_prob"] = float(test_prob[idx])
    return row


def protected_condition(summary):
    checks = []
    for file_name, expected in PROTECTED.items():
        checks.append(summary[f"{file_name.replace('.png', '')}_pred"].eq(expected))
    out = checks[0]
    for check in checks[1:]:
        out = out & check
    return out


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p")


def main():
    oof, test = load_oof_and_test()
    y = oof["label"].to_numpy()
    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    current_best = pd.read_csv(CURRENT_BEST_PATH).sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test["file_name"].to_numpy(), exp26["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-26")
    if not np.array_equal(test["file_name"].to_numpy(), current_best["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against current best")
    exp26_pred = exp26["pred"].to_numpy()
    current_best_pred = current_best["pred"].to_numpy()

    search_space = [
        ("ridge", [30, 50, 75, 100, 150, 300, 500, 1000, 3000]),
        ("logistic", [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0]),
        ("svc", [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]),
    ]

    rows = []
    artifacts = []
    for spec, cols in feature_sets().items():
        for kind, values in search_space:
            for value in values:
                for method in ["sigmoid", "isotonic"]:
                    cv_prob = meta_cv_predict(oof, cols, kind, value, method)
                    test_prob = final_test_predict(oof, test, cols, kind, value, method)
                    default_cv_pred = (cv_prob >= 0.5).astype(int)
                    default_test_pred = (test_prob >= 0.5).astype(int)
                    for threshold in THRESHOLDS:
                        cv_pred = (cv_prob >= threshold).astype(int)
                        test_pred = (test_prob >= threshold).astype(int)
                        row = {
                            "spec": spec,
                            "model": kind,
                            "param": value,
                            "calibration": method,
                            "threshold": threshold,
                            "features": "+".join(cols),
                            "n_features": len(cols),
                            "default_meta_cv_changed": int(np.sum(cv_pred != default_cv_pred)),
                            "default_test_changed": int(np.sum(test_pred != default_test_pred)),
                        }
                        row.update(metric_row(y, cv_pred))
                        row = add_common(row, test, test_pred, test_prob, exp26_pred, current_best_pred)
                        rows.append(row)
                        artifacts.append((spec, kind, value, method, threshold, cols, row, test_pred, test_prob))

    summary = pd.DataFrame(rows)
    summary["passes_protected"] = protected_condition(summary)
    summary["passes_label_band"] = summary["test_label1"].between(395, 399)
    summary["passes_low_change_current"] = summary["changed_vs_current_best"].between(1, 4)
    summary["passes_preferred_low_change"] = summary["changed_vs_current_best"].between(1, 2)
    summary["passes_meta_cv_floor"] = summary["meta_cv_accuracy"].ge(CURRENT_BEST_META_CV_ACCURACY - META_CV_TOLERANCE)
    summary["not_duplicate_current"] = summary["changed_vs_current_best"].gt(0)
    summary["candidate"] = (
        summary["passes_protected"]
        & summary["passes_label_band"]
        & summary["passes_low_change_current"]
        & summary["passes_meta_cv_floor"]
        & summary["not_duplicate_current"]
    )
    summary = summary.sort_values(
        [
            "candidate",
            "passes_preferred_low_change",
            "passes_protected",
            "changed_vs_current_best",
            "meta_cv_accuracy",
            "meta_cv_recall",
            "changed_vs_exp26",
        ],
        ascending=[False, False, False, True, False, False, True],
    )
    summary_path = PRED_DIR / f"{EXP_ID}_calibrated_stacking_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(10)
    saved = 0
    for _, sel in selected.iterrows():
        for spec, kind, value, method, threshold, cols, row, test_pred, test_prob in artifacts:
            if (
                spec == sel["spec"]
                and kind == sel["model"]
                and value == sel["param"]
                and method == sel["calibration"]
                and threshold == sel["threshold"]
            ):
                name = f"{EXP_ID}_{kind}_{spec}_{method}_p{safe_name(value)}_t{safe_name(threshold)}"
                sub = pd.DataFrame({"file_name": test["file_name"], "label": test_pred.astype(int)})
                sub_path = SUB_DIR / f"{name}_submit.csv"
                if sub_path.exists():
                    continue
                sub.to_csv(sub_path, index=False)
                score_df = pd.DataFrame(
                    {
                        "file_name": test["file_name"],
                        "prob1": test_prob,
                        "pred": test_pred.astype(int),
                    }
                )
                score_df.to_csv(PRED_DIR / f"{name}_test_scores.csv", index=False)
                changed = score_df[test_pred != current_best_pred].copy()
                changed["current_best_pred"] = current_best_pred[test_pred != current_best_pred]
                changed.to_csv(PRED_DIR / f"{name}_changed_vs_current_best.csv", index=False)
                saved += 1
                break

    print(f"summary: {summary_path}")
    print(f"candidate count: {int(summary['candidate'].sum())}")
    print(f"saved submissions: {saved}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
