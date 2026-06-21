from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


EXP_ID = "exp47"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.3, 1, 3, 10, 30, 50, 75, 100, 150, 300, 500, 1000, 3000]


def metric_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "oof_accuracy": accuracy_score(y_true, y_pred),
        "oof_precision": precision_score(y_true, y_pred, zero_division=0),
        "oof_recall": recall_score(y_true, y_pred, zero_division=0),
        "oof_f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def add_features(df: pd.DataFrame, rank_source: pd.Series | None = None):
    out = df.copy()
    if "prob_0" not in out.columns:
        out["prob_0"] = 1.0 - out["prob_1"]
    source = rank_source if rank_source is not None else out["prob_1"]
    out["rank"] = source.rank(pct=True).to_numpy()
    out["margin"] = out["prob_1"] - out["prob_0"]
    out["uncertainty"] = 1.0 - np.abs(out["margin"])
    return out


def main():
    oof = pd.read_csv(PRED_DIR / "exp46_convnext_tiny_oof_probs.csv")
    test = pd.read_csv(PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv")
    sample = pd.read_csv("sample_submission.csv")

    oof = add_features(oof)
    test = add_features(test)

    y = oof["label"].to_numpy()
    direct_oof_pred = oof["pred_label"].to_numpy()
    direct_test_pred = test["pred_label"].to_numpy()

    feature_sets = {
        "prob1_only": ["prob_1"],
        "prob1_rank": ["prob_1", "rank"],
        "prob01_rank": ["prob_0", "prob_1", "rank"],
        "prob1_margin": ["prob_1", "margin"],
        "prob1_rank_margin_unc": ["prob_1", "rank", "margin", "uncertainty"],
    }

    rows = []
    candidates = []
    for spec, cols in feature_sets.items():
        for alpha in ALPHAS:
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(oof[cols], y)
            oof_pred = clf.predict(oof[cols])
            test_pred = clf.predict(test[cols])
            score = clf.decision_function(test[cols])

            row = {
                "spec": spec,
                "alpha": alpha,
                "features": "+".join(cols),
                "n_features": len(cols),
                "oof_changed_vs_direct": int(np.sum(oof_pred != direct_oof_pred)),
                "test_changed_vs_direct": int(np.sum(test_pred != direct_test_pred)),
                "test_label0": int(np.sum(test_pred == 0)),
                "test_label1": int(np.sum(test_pred == 1)),
            }
            row.update(metric_row(y, oof_pred))
            for file_name in ["test_0244.png", "test_0025.png", "test_0539.png"]:
                idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
                row[file_name.replace(".png", "_pred")] = int(test_pred[idx])
                row[file_name.replace(".png", "_score")] = float(score[idx])
            rows.append(row)
            candidates.append((spec, alpha, cols, row, test_pred, score))

    direct_row = {
        "spec": "direct_convnext_oof_mean",
        "alpha": np.nan,
        "features": "prob_1>=0.5",
        "n_features": 1,
        "oof_changed_vs_direct": 0,
        "test_changed_vs_direct": 0,
        "test_label0": int(np.sum(direct_test_pred == 0)),
        "test_label1": int(np.sum(direct_test_pred == 1)),
    }
    direct_row.update(metric_row(y, direct_oof_pred))
    for file_name in ["test_0244.png", "test_0025.png", "test_0539.png"]:
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        direct_row[file_name.replace(".png", "_pred")] = int(direct_test_pred[idx])
        direct_row[file_name.replace(".png", "_score")] = float(test["prob_1"].iloc[idx])
    rows.append(direct_row)

    summary = pd.DataFrame(rows).sort_values(
        ["oof_accuracy", "oof_recall", "test_changed_vs_direct"], ascending=[False, False, True]
    )
    summary_path = PRED_DIR / f"{EXP_ID}_oof_ridge_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["spec"] != "direct_convnext_oof_mean"].head(5)
    selected_keys = {(row.spec, float(row.alpha)) for row in selected.itertuples()}

    for spec, alpha, cols, row, test_pred, score in candidates:
        if (spec, float(alpha)) not in selected_keys:
            continue
        alpha_name = str(alpha).replace(".", "p")
        prefix = f"{EXP_ID}_{spec}_alpha{alpha_name}"
        submit = sample.copy()
        pred_map = dict(zip(test["file_name"], test_pred))
        submit["label"] = submit["file_name"].map(pred_map).astype(int)
        submit_path = SUB_DIR / f"{prefix}_submit.csv"
        submit.to_csv(submit_path, index=False)
        score_path = PRED_DIR / f"{prefix}_test_scores.csv"
        pd.DataFrame({"file_name": test["file_name"], "decision_score": score, "pred": test_pred}).to_csv(
            score_path, index=False
        )

    print(summary.head(20).to_string(index=False))
    print("saved:", summary_path)
    print("saved submissions:", len(selected_keys))


if __name__ == "__main__":
    main()
