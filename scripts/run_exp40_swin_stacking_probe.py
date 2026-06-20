from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PRED_DIR = Path("preds")
SUB_DIR = Path("submissions")

MODEL_PATHS = {
    "resnet18": "preds/exp07_resnet18",
    "efficientnet_b0": "preds/exp06_efficientnet_b0",
    "convnext12": "preds/exp12_convnext_tiny",
    "convnext18_full": "preds/exp18_convnext_tiny_full_finetune",
    "swin_tiny": "preds/exp40_swin_tiny",
}

VARIANTS = {
    "no_den_plus_swin": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full", "swin_tiny"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full", "swin_tiny"],
    },
    "no_den_swin_rank_only": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full", "swin_tiny"],
    },
    "res_eff_cnx12_swin": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12", "swin_tiny"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "swin_tiny"],
    },
    "res_eff_swin": {
        "prob_models": ["resnet18", "efficientnet_b0", "swin_tiny"],
        "rank_models": ["resnet18", "efficientnet_b0", "swin_tiny"],
    },
}

ALPHAS = [1, 10, 30, 50, 100, 300, 1000]


def load_split(split):
    base = None
    for name, stem in MODEL_PATHS.items():
        df = pd.read_csv(f"{stem}_{split}_probs.csv")
        cur = df[["file_name", "prob_1"]].copy().rename(columns={"prob_1": f"{name}_prob"})
        if split == "valid" and base is None:
            cur["label"] = df["label"].to_numpy()
        cur[f"{name}_rank"] = cur[f"{name}_prob"].rank(pct=True)
        base = cur if base is None else base.merge(cur, on="file_name", how="inner")
    return base


def features(spec):
    out = []
    out.extend([f"{m}_prob" for m in spec["prob_models"]])
    out.extend([f"{m}_rank" for m in spec["rank_models"]])
    return out


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
    valid = load_split("valid")
    test = load_split("test")
    y = valid["label"].to_numpy()
    exp26 = pd.read_csv(PRED_DIR / "exp26_ridge_without_densenet_alpha30_test_scores.csv")
    exp26_pred = exp26["pred"].to_numpy()

    rows = []
    candidates = []
    for variant, spec in VARIANTS.items():
        cols = features(spec)
        for alpha in ALPHAS:
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(valid[cols], y)
            valid_pred = clf.predict(valid[cols])
            test_pred = clf.predict(test[cols])
            score = clf.decision_function(test[cols])
            row = {
                "variant": variant,
                "alpha": alpha,
                "n_features": len(cols),
                "features": "+".join(cols),
                "test_label0": int(np.sum(test_pred == 0)),
                "test_label1": int(np.sum(test_pred == 1)),
                "changed_vs_exp26": int(np.sum(test_pred != exp26_pred)),
                "test_0244_pred": int(test_pred[test["file_name"].to_numpy() == "test_0244.png"][0]),
            }
            row.update(metric_row(y, valid_pred))
            rows.append(row)
            candidates.append((variant, alpha, row, test_pred, score))

    summary = pd.DataFrame(rows).sort_values(
        ["test_0244_pred", "valid_accuracy", "valid_recall", "changed_vs_exp26"],
        ascending=[True, False, False, True],
    )
    summary_path = PRED_DIR / "exp40_swin_stacking_probe_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[(summary["test_0244_pred"] == 0) & (summary["changed_vs_exp26"] <= 5)].head(6)
    selected_keys = {(row.variant, int(row.alpha)) for row in selected.itertuples()}
    for variant, alpha, row, test_pred, score in candidates:
        if (variant, int(alpha)) not in selected_keys:
            continue
        prefix = f"exp40_stack_{variant}_alpha{int(alpha)}"
        pd.DataFrame({"file_name": test["file_name"], "label": test_pred}).to_csv(
            SUB_DIR / f"{prefix}_submit.csv", index=False
        )
        pd.DataFrame({"file_name": test["file_name"], "decision_score": score, "pred": test_pred}).to_csv(
            PRED_DIR / f"{prefix}_test_scores.csv", index=False
        )
        changed = test.loc[test_pred != exp26_pred, ["file_name"]].copy()
        changed["exp26_pred"] = exp26_pred[test_pred != exp26_pred]
        changed["exp40_pred"] = test_pred[test_pred != exp26_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_exp26.csv", index=False)

    print(summary.head(20).to_string(index=False))
    print("saved:", summary_path)
    print("saved candidates:", len(selected_keys))


if __name__ == "__main__":
    main()
