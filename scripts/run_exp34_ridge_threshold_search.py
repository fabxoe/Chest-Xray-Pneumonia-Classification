from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


THRESHOLDS = [
    -0.20,
    -0.15,
    -0.12,
    -0.10,
    -0.08,
    -0.06,
    -0.04,
    -0.02,
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
]


def metric_row(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "valid_accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_name(th):
    sign = "m" if th < 0 else "p"
    return f"{sign}{abs(th):.2f}".replace(".", "p")


def main():
    preds_dir = Path("preds")
    sub_dir = Path("submissions")
    preds_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)

    valid = pd.read_csv(preds_dir / "exp26_ridge_without_densenet_alpha30_valid_scores.csv")
    test = pd.read_csv(preds_dir / "exp26_ridge_without_densenet_alpha30_test_scores.csv")
    base_test_pred = test["pred"].to_numpy()

    rows = []
    for th in THRESHOLDS:
        valid_pred = (valid["decision_score"].to_numpy() >= th).astype(int)
        test_pred = (test["decision_score"].to_numpy() >= th).astype(int)
        row = {
            "threshold": th,
            "changed_test_vs_exp26": int(np.sum(test_pred != base_test_pred)),
            "test_label_0_count": int(np.sum(test_pred == 0)),
            "test_label_1_count": int(np.sum(test_pred == 1)),
        }
        row.update(metric_row(valid["label"].to_numpy(), valid_pred))
        rows.append(row)

        if th != 0.0 and row["changed_test_vs_exp26"] > 0:
            out_path = sub_dir / f"exp34_ridge_threshold_{threshold_name(th)}_submit.csv"
            pd.DataFrame({"file_name": test["file_name"], "label": test_pred}).to_csv(out_path, index=False)

            changed = test.loc[test_pred != base_test_pred, ["file_name", "decision_score"]].copy()
            changed["exp26_pred"] = base_test_pred[test_pred != base_test_pred]
            changed["exp34_pred"] = test_pred[test_pred != base_test_pred]
            changed.to_csv(preds_dir / f"exp34_ridge_threshold_{threshold_name(th)}_changed_vs_exp26.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(preds_dir / "exp34_ridge_threshold_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("saved:", preds_dir / "exp34_ridge_threshold_summary.csv")


if __name__ == "__main__":
    main()
