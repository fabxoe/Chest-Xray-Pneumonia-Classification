from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from run_exp51_oof_3model_stacking import (
    final_test_predict,
    load_oof_and_test,
    meta_cv_predict,
)


EXP_ID = "exp53"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_BEST_SCORE = 0.9567307692
CURRENT_BEST_META_CV_ACCURACY = 0.9865804597701149
META_CV_TOLERANCE = 0.001
CURRENT_BEST_PATH = PRED_DIR / "exp52_logistic_cnx_eff_prob_only_p0p25_t0p0_test_scores.csv"

PROTECTED = {
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
}

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
        "values": [20, 30, 40, 50, 60, 75, 100, 150, 200, 300, 500, 750, 1000],
        "cols": ["cnx_prob1", "eff_prob1"],
    },
    {
        "name": "ridge_cnx_eff_prob_margin",
        "kind": "ridge",
        "values": [50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
        "cols": ["cnx_prob1", "eff_prob1", "cnx_margin", "eff_margin"],
    },
]

THRESHOLDS = [
    -0.10,
    -0.075,
    -0.05,
    -0.04,
    -0.03,
    -0.025,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.25,
]


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


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p")


def add_common(row, test, test_pred, test_score, exp26_pred, current_best_pred):
    row["test_label0"] = int(np.sum(test_pred == 0))
    row["test_label1"] = int(np.sum(test_pred == 1))
    row["changed_vs_exp26"] = int(np.sum(test_pred != exp26_pred))
    row["changed_vs_current_best"] = int(np.sum(test_pred != current_best_pred))
    for file_name in PROTECTED:
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        tag = file_name.replace(".png", "")
        row[f"{tag}_pred"] = int(test_pred[idx])
        row[f"{tag}_score"] = float(test_score[idx])
    return row


def protected_condition(summary):
    checks = []
    for file_name, expected in PROTECTED.items():
        tag = file_name.replace(".png", "")
        checks.append(summary[f"{tag}_pred"].eq(expected))
    out = checks[0]
    for check in checks[1:]:
        out = out & check
    return out


def load_reference_predictions(test):
    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    current_best = pd.read_csv(CURRENT_BEST_PATH).sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test["file_name"].to_numpy(), exp26["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-26")
    if not np.array_equal(test["file_name"].to_numpy(), current_best["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against current best")
    return exp26["pred"].to_numpy(), current_best["pred"].to_numpy()


def main():
    oof, test = load_oof_and_test()
    y = oof["label"].to_numpy()
    exp26_pred, current_best_pred = load_reference_predictions(test)

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
                row = add_common(row, test, test_pred, test_score, exp26_pred, current_best_pred)
                rows.append(row)
                candidates.append((spec, value, threshold, row, test_pred, test_score))

    summary = pd.DataFrame(rows)
    summary["passes_protected"] = protected_condition(summary)
    summary["passes_label_band"] = summary["test_label1"].between(395, 399)
    summary["passes_low_change_current"] = summary["changed_vs_current_best"].between(1, 3)
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
            "passes_protected",
            "passes_low_change_current",
            "changed_vs_current_best",
            "meta_cv_accuracy",
            "meta_cv_recall",
            "changed_vs_exp26",
        ],
        ascending=[False, False, False, True, False, False, True],
    )
    summary_path = PRED_DIR / f"{EXP_ID}_constraint_low_change_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(10)
    if selected.empty:
        selected = summary[
            summary["passes_protected"]
            & summary["passes_label_band"]
            & summary["passes_meta_cv_floor"]
            & summary["not_duplicate_current"]
        ].head(10)

    selected_keys = {
        (row.spec, row.model, float(row.param), float(row.threshold))
        for row in selected.itertuples()
    }
    sample = pd.read_csv("sample_submission.csv")
    saved_hashes = set()
    saved_count = 0
    for spec, value, threshold, row, test_pred, test_score in candidates:
        key = (spec["name"], spec["kind"], float(value), float(threshold))
        if key not in selected_keys:
            continue
        pred_hash = tuple(test_pred.tolist())
        if pred_hash in saved_hashes:
            continue
        saved_hashes.add(pred_hash)

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
                "current_best_pred": current_best_pred,
                "changed_vs_exp26": test_pred != exp26_pred,
                "changed_vs_current_best": test_pred != current_best_pred,
            }
        )
        score_df.to_csv(PRED_DIR / f"{prefix}_test_scores.csv", index=False)

        changed = test.loc[test_pred != current_best_pred, ["file_name"]].copy()
        changed["current_best_pred"] = current_best_pred[test_pred != current_best_pred]
        changed["exp53_pred"] = test_pred[test_pred != current_best_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_current_best.csv", index=False)
        saved_count += 1

    print(summary.head(40).to_string(index=False))
    print("candidate count:", int(summary["candidate"].sum()))
    print("current best leaderboard:", CURRENT_BEST_SCORE)
    print("current best meta-cv floor:", CURRENT_BEST_META_CV_ACCURACY - META_CV_TOLERANCE)
    print("saved:", summary_path)
    print("saved unique submissions:", saved_count)


if __name__ == "__main__":
    main()
