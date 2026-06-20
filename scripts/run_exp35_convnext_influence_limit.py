from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PRED_DIR = Path("preds")
SUB_DIR = Path("submissions")
PRED_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

MODEL_PATHS = {
    "resnet18": "preds/exp07_resnet18",
    "efficientnet_b0": "preds/exp06_efficientnet_b0",
    "convnext12": "preds/exp12_convnext_tiny",
    "convnext18_full": "preds/exp18_convnext_tiny_full_finetune",
}

ALPHAS = [1, 10, 30, 50, 100, 300, 1000]


VARIANTS = {
    "base_rebuilt": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
    },
    "conv_rank_only": {
        "prob_models": ["resnet18", "efficientnet_b0"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
    },
    "conv12_rank_only": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext18_full"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
    },
    "conv18_rank_only": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12", "convnext18_full"],
    },
    "without_conv12": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext18_full"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext18_full"],
    },
    "without_conv18": {
        "prob_models": ["resnet18", "efficientnet_b0", "convnext12"],
        "rank_models": ["resnet18", "efficientnet_b0", "convnext12"],
    },
    "res_eff_only": {
        "prob_models": ["resnet18", "efficientnet_b0"],
        "rank_models": ["resnet18", "efficientnet_b0"],
    },
}


def load_split(split):
    base = None
    for name, stem in MODEL_PATHS.items():
        df = pd.read_csv(f"{stem}_{split}_probs.csv")
        current = df[["file_name", "prob_1"]].copy().rename(columns={"prob_1": f"{name}_prob"})
        if split == "valid" and base is None:
            current["label"] = df["label"].to_numpy()
        current[f"{name}_rank"] = current[f"{name}_prob"].rank(pct=True)
        base = current if base is None else base.merge(current, on="file_name", how="inner")
    return base


def get_features(spec):
    features = []
    for model_name in spec["prob_models"]:
        features.append(f"{model_name}_prob")
    for model_name in spec["rank_models"]:
        features.append(f"{model_name}_rank")
    return features


def metrics(y_true, y_pred):
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
    for variant_name, spec in VARIANTS.items():
        features = get_features(spec)
        for alpha in ALPHAS:
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(valid[features], y)
            valid_pred = clf.predict(valid[features])
            test_pred = clf.predict(test[features])
            score = clf.decision_function(test[features])

            row = {
                "variant": variant_name,
                "alpha": alpha,
                "n_features": len(features),
                "features": "+".join(features),
                "test_label0": int(np.sum(test_pred == 0)),
                "test_label1": int(np.sum(test_pred == 1)),
                "changed_vs_exp26": int(np.sum(test_pred != exp26_pred)),
            }
            row.update(metrics(y, valid_pred))
            rows.append(row)

            candidates.append((variant_name, alpha, row, test_pred, score))

    summary = pd.DataFrame(rows).sort_values(
        ["valid_accuracy", "valid_recall", "changed_vs_exp26"], ascending=[False, False, True]
    )
    summary_path = PRED_DIR / "exp35_convnext_influence_limit_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Save a compact set of candidates: top validation rows and low-change rows.
    selected_keys = set()
    for _, row in summary.head(8).iterrows():
        selected_keys.add((row["variant"], int(row["alpha"])))
    low_change = summary[(summary["changed_vs_exp26"] > 0) & (summary["changed_vs_exp26"] <= 3)]
    for _, row in low_change.head(8).iterrows():
        selected_keys.add((row["variant"], int(row["alpha"])))

    for variant_name, alpha, row, test_pred, score in candidates:
        if (variant_name, int(alpha)) not in selected_keys:
            continue
        prefix = f"exp35_{variant_name}_alpha{int(alpha)}"
        pd.DataFrame({"file_name": test["file_name"], "label": test_pred}).to_csv(
            SUB_DIR / f"{prefix}_submit.csv", index=False
        )
        score_df = pd.DataFrame({"file_name": test["file_name"], "decision_score": score, "pred": test_pred})
        score_df.to_csv(PRED_DIR / f"{prefix}_test_scores.csv", index=False)
        changed = score_df.loc[test_pred != exp26_pred, ["file_name", "decision_score", "pred"]].copy()
        changed = changed.rename(columns={"pred": "exp35_pred"})
        changed["exp26_pred"] = exp26_pred[test_pred != exp26_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_exp26.csv", index=False)

    print(summary.head(20).to_string(index=False))
    print("saved:", summary_path)
    print("saved candidates:", len(selected_keys))


if __name__ == "__main__":
    main()
