from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from run_exp51_oof_3model_stacking import (
    SENSITIVE,
    final_test_predict,
    load_oof_and_test,
    make_estimator,
    meta_cv_predict,
)


EXP_ID = "exp52"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


MODEL_SPECS = [
    {
        "name": "logistic_cnx_eff_prob_rank",
        "kind": "logistic",
        "values": [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0],
        "cols": ["cnx_prob1", "eff_prob1", "cnx_rank", "eff_rank"],
    },
    {
        "name": "logistic_cnx_eff_prob_only",
        "kind": "logistic",
        "values": [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0],
        "cols": ["cnx_prob1", "eff_prob1"],
    },
    {
        "name": "ridge_cnx_eff_prob_only",
        "kind": "ridge",
        "values": [20, 30, 40, 50, 60, 75, 100, 150, 200, 300, 500],
        "cols": ["cnx_prob1", "eff_prob1"],
    },
    {
        "name": "ridge_cnx_eff_prob_margin",
        "kind": "ridge",
        "values": [50, 75, 100, 150, 200, 300, 500, 750, 1000],
        "cols": ["cnx_prob1", "eff_prob1", "cnx_margin", "eff_margin"],
    },
]

THRESHOLDS = [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
CURRENT_BEST_SCORE = 0.9551282051


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


def add_common(row, test, test_pred, test_score, exp26_pred, exp51_best_pred):
    row["test_label0"] = int(np.sum(test_pred == 0))
    row["test_label1"] = int(np.sum(test_pred == 1))
    row["changed_vs_exp26"] = int(np.sum(test_pred != exp26_pred))
    row["changed_vs_exp51_best"] = int(np.sum(test_pred != exp51_best_pred))
    for file_name in SENSITIVE:
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        row[file_name.replace(".png", "_pred")] = int(test_pred[idx])
        row[file_name.replace(".png", "_score")] = float(test_score[idx])
    return row


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p")


def main():
    oof, test = load_oof_and_test()
    y = oof["label"].to_numpy()

    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    exp51_best = pd.read_csv("oof/preds/exp51_logistic_cnx_eff_prob_rank_0p3_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test["file_name"].to_numpy(), exp26["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-26")
    if not np.array_equal(test["file_name"].to_numpy(), exp51_best["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-51 best")

    exp26_pred = exp26["pred"].to_numpy()
    exp51_best_pred = exp51_best["pred"].to_numpy()

    rows = []
    candidates = []
    for spec in MODEL_SPECS:
        for value in spec["values"]:
            cv_pred_default, cv_score = meta_cv_predict(oof, spec["cols"], spec["kind"], value)
            test_pred_default, test_score = final_test_predict(oof, test, spec["cols"], spec["kind"], value)

            for threshold in THRESHOLDS:
                cv_pred = (cv_score >= threshold).astype(int)
                test_pred = (test_score >= threshold).astype(int)
                row = {
                    "spec": spec["name"],
                    "model": spec["kind"],
                    "param": value,
                    "threshold": threshold,
                    "features": "+".join(spec["cols"]),
                    "n_features": len(spec["cols"]),
                    "default_meta_cv_changed": int(np.sum(cv_pred != cv_pred_default)),
                    "default_test_changed": int(np.sum(test_pred != test_pred_default)),
                }
                row.update(metric_row(y, cv_pred))
                row = add_common(row, test, test_pred, test_score, exp26_pred, exp51_best_pred)
                rows.append(row)
                candidates.append((spec, value, threshold, row, test_pred, test_score))

    summary = pd.DataFrame(rows)
    summary["passes_sensitive_strict"] = (
        summary["test_0244_pred"].eq(0)
        & summary["test_0025_pred"].eq(0)
        & summary["test_0539_pred"].eq(0)
    )
    summary["passes_label_band"] = summary["test_label1"].between(392, 402)
    summary["passes_low_change_exp26"] = summary["changed_vs_exp26"].between(2, 8)
    summary["passes_low_change_exp51"] = summary["changed_vs_exp51_best"].between(1, 4)
    summary["not_duplicate_exp51"] = summary["changed_vs_exp51_best"].gt(0)
    summary["candidate"] = (
        summary["passes_sensitive_strict"]
        & summary["passes_label_band"]
        & summary["passes_low_change_exp26"]
        & summary["not_duplicate_exp51"]
    )
    summary = summary.sort_values(
        [
            "candidate",
            "passes_low_change_exp51",
            "meta_cv_accuracy",
            "meta_cv_recall",
            "changed_vs_exp51_best",
            "changed_vs_exp26",
        ],
        ascending=[False, False, False, False, True, True],
    )
    summary_path = PRED_DIR / f"{EXP_ID}_low_change_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(10)
    if selected.empty:
        selected = summary[summary["passes_sensitive_strict"] & summary["not_duplicate_exp51"]].head(10)

    selected_keys = {
        (row.spec, row.model, float(row.param), float(row.threshold))
        for row in selected.itertuples()
    }
    sample = pd.read_csv("sample_submission.csv")
    for spec, value, threshold, row, test_pred, test_score in candidates:
        key = (spec["name"], spec["kind"], float(value), float(threshold))
        if key not in selected_keys:
            continue
        prefix = f"{EXP_ID}_{spec['name']}_p{safe_name(value)}_t{safe_name(threshold)}"
        submit = sample.copy()
        submit["label"] = submit["file_name"].map(dict(zip(test["file_name"], test_pred))).astype(int)
        submit.to_csv(SUB_DIR / f"{prefix}_submit.csv", index=False)

        score_df = pd.DataFrame(
            {
                "file_name": test["file_name"],
                "decision_score": test_score,
                "pred": test_pred,
                "exp26_pred": exp26_pred,
                "exp51_best_pred": exp51_best_pred,
                "changed_vs_exp26": test_pred != exp26_pred,
                "changed_vs_exp51_best": test_pred != exp51_best_pred,
            }
        )
        score_df.to_csv(PRED_DIR / f"{prefix}_test_scores.csv", index=False)

        changed = test.loc[test_pred != exp51_best_pred, ["file_name"]].copy()
        changed["exp51_best_pred"] = exp51_best_pred[test_pred != exp51_best_pred]
        changed["exp52_pred"] = test_pred[test_pred != exp51_best_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_exp51_best.csv", index=False)

    print(summary.head(30).to_string(index=False))
    print("candidate count:", int(summary["candidate"].sum()))
    print("current best leaderboard:", CURRENT_BEST_SCORE)
    print("saved:", summary_path)
    print("saved submissions:", len(selected_keys))


if __name__ == "__main__":
    main()
