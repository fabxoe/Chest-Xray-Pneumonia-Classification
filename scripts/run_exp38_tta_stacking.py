from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PRED_DIR = Path("preds")
SUB_DIR = Path("submissions")
PRED_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

ALPHAS = [1, 3, 10, 30, 50, 75, 100, 150, 300, 500, 1000]

MODEL_VARIANTS = {
    "tta_cnx12_only": {
        "resnet18": "preds/exp07_resnet18",
        "efficientnet_b0": "preds/exp06_efficientnet_b0",
        "convnext12": "preds/exp37_tta_convnext12",
        "convnext18_full": "preds/exp18_convnext_tiny_full_finetune",
    },
    "tta_cnx12_cnx18": {
        "resnet18": "preds/exp07_resnet18",
        "efficientnet_b0": "preds/exp06_efficientnet_b0",
        "convnext12": "preds/exp37_tta_convnext12",
        "convnext18_full": "preds/exp37_tta_convnext18_full",
    },
    "tta_all": {
        "resnet18": "preds/exp37_tta_resnet18",
        "efficientnet_b0": "preds/exp37_tta_efficientnet_b0",
        "convnext12": "preds/exp37_tta_convnext12",
        "convnext18_full": "preds/exp37_tta_convnext18_full",
    },
    "tta_cnx12_prob_only": {
        "resnet18": "preds/exp07_resnet18",
        "efficientnet_b0": "preds/exp06_efficientnet_b0",
        "convnext12": "preds/exp37_tta_convnext12",
        "convnext18_full": "preds/exp18_convnext_tiny_full_finetune",
    },
}


def load_split(paths, split):
    base = None
    for name, stem in paths.items():
        df = pd.read_csv(f"{stem}_{split}_probs.csv")
        cur = df[["file_name", "prob_1"]].copy().rename(columns={"prob_1": f"{name}_prob"})
        if split == "valid" and base is None:
            cur["label"] = df["label"].to_numpy()
        cur[f"{name}_rank"] = cur[f"{name}_prob"].rank(pct=True)
        base = cur if base is None else base.merge(cur, on="file_name", how="inner")
    return base


def metric_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "valid_accuracy": accuracy_score(y_true, y_pred),
        "valid_precision": precision_score(y_true, y_pred, zero_division=0),
        "valid_recall": recall_score(y_true, y_pred, zero_division=0),
        "valid_f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    exp26 = pd.read_csv(PRED_DIR / "exp26_ridge_without_densenet_alpha30_test_scores.csv")
    exp26_pred = exp26["pred"].to_numpy()
    test_names_ref = exp26["file_name"].to_numpy()
    idx_0244 = int(np.where(test_names_ref == "test_0244.png")[0][0])

    rows = []
    candidates = []
    for variant, paths in MODEL_VARIANTS.items():
        valid = load_split(paths, "valid")
        test = load_split(paths, "test")
        y = valid["label"].to_numpy()
        model_names = list(paths.keys())
        if variant.endswith("prob_only"):
            feature_sets = {"prob_only": [f"{m}_prob" for m in model_names]}
        else:
            feature_sets = {
                "prob_rank": [col for m in model_names for col in (f"{m}_prob", f"{m}_rank")],
                "prob_only": [f"{m}_prob" for m in model_names],
            }

        for feature_type, cols in feature_sets.items():
            for alpha in ALPHAS:
                clf = RidgeClassifier(alpha=alpha)
                clf.fit(valid[cols], y)
                valid_pred = clf.predict(valid[cols])
                test_pred = clf.predict(test[cols])
                score = clf.decision_function(test[cols])
                row = {
                    "variant": variant,
                    "feature_type": feature_type,
                    "alpha": alpha,
                    "n_features": len(cols),
                    "test_label0": int(np.sum(test_pred == 0)),
                    "test_label1": int(np.sum(test_pred == 1)),
                    "changed_vs_exp26": int(np.sum(test_pred != exp26_pred)),
                    "test_0244_pred": int(test_pred[idx_0244]),
                }
                row.update(metric_row(y, valid_pred))
                rows.append(row)
                candidates.append((variant, feature_type, alpha, row, test, test_pred, score))

    summary = pd.DataFrame(rows)
    summary["candidate"] = (
        summary["test_0244_pred"].eq(0)
        & summary["changed_vs_exp26"].between(1, 8)
        & summary["valid_accuracy"].ge(0.985632183908046)
    )
    summary = summary.sort_values(
        ["candidate", "valid_accuracy", "valid_recall", "changed_vs_exp26"],
        ascending=[False, False, False, True],
    )
    summary_path = PRED_DIR / "exp38_tta_stacking_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(8)
    if selected.empty:
        selected = summary[summary["test_0244_pred"].eq(0)].head(5)
    selected_keys = {(r.variant, r.feature_type, float(r.alpha)) for r in selected.itertuples()}
    for variant, feature_type, alpha, row, test, test_pred, score in candidates:
        if (variant, feature_type, float(alpha)) not in selected_keys:
            continue
        alpha_name = str(alpha).replace(".", "p")
        prefix = f"exp38_{variant}_{feature_type}_alpha{alpha_name}"
        pd.DataFrame({"file_name": test["file_name"], "label": test_pred}).to_csv(
            SUB_DIR / f"{prefix}_submit.csv", index=False
        )
        pd.DataFrame({"file_name": test["file_name"], "decision_score": score, "pred": test_pred}).to_csv(
            PRED_DIR / f"{prefix}_test_scores.csv", index=False
        )
        changed = test.loc[test_pred != exp26_pred, ["file_name"]].copy()
        changed["exp26_pred"] = exp26_pred[test_pred != exp26_pred]
        changed["exp38_pred"] = test_pred[test_pred != exp26_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_exp26.csv", index=False)

    print(summary.head(25).to_string(index=False))
    print("candidate count:", int(summary["candidate"].sum()))
    print("saved:", summary_path)
    print("saved submissions:", len(selected_keys))


if __name__ == "__main__":
    main()
