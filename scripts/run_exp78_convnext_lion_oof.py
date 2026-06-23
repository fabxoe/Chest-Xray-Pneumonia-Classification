import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from timm.optim.lion import Lion
import torch

import run_exp46_convnext_oof_remaining_folds as convnext_oof


EXP_ID = "exp78"
MODEL_NAME = "convnext_tiny_lion"
# Resume after the IDE/Docker interruption. Fold 0 and 1 outputs already exist.
FOLDS_TO_RUN = [2, 3, 4]
COMBINE_FOLDS = [0, 1, 2, 3, 4]
SEED = 42
LR = 2e-5
WEIGHT_DECAY = 1e-4


def configure_base_module():
    convnext_oof.EXP_ID = EXP_ID
    convnext_oof.MODEL_NAME = MODEL_NAME
    convnext_oof.FOLDS_TO_RUN = FOLDS_TO_RUN
    convnext_oof.ALL_FOLDS = COMBINE_FOLDS
    convnext_oof.SEED = SEED
    convnext_oof.LR = LR

    def lion_factory(params, lr, weight_decay):
        return Lion(params, lr=lr, weight_decay=weight_decay)

    convnext_oof.optim.AdamW = lion_factory


def load_fold_summary(fold):
    result_path = Path(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["final_metrics"]
    return {
        "fold": fold,
        "best_epoch": result.get("best_epoch"),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "loss": metrics["loss"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }


def combine_outputs():
    oof_parts = []
    test_parts = []
    for fold in COMBINE_FOLDS:
        oof_parts.append(pd.read_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_oof_probs.csv"))
        test_parts.append(pd.read_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_test_probs.csv"))

    oof_all = pd.concat(oof_parts, ignore_index=True).sort_values("file_name").reset_index(drop=True)
    test_by_fold = pd.concat(test_parts, ignore_index=True)
    test_mean = (
        test_by_fold.groupby("file_name", as_index=False)[["prob_0", "prob_1"]]
        .mean()
        .assign(pred_label=lambda df: (df["prob_1"] >= df["prob_0"]).astype(int))
    )

    oof_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_oof_probs.csv"
    test_by_fold_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_by_fold.csv"
    test_mean_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_mean.csv"
    oof_all.to_csv(oof_path, index=False)
    test_by_fold.to_csv(test_by_fold_path, index=False)
    test_mean.to_csv(test_mean_path, index=False)

    submission = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(test_mean["file_name"], test_mean["pred_label"]))
    submission["label"] = submission["file_name"].map(pred_map).astype(int)
    submission_path = f"oof/submissions/{EXP_ID}_{MODEL_NAME}_mean_submit.csv"
    submission.to_csv(submission_path, index=False)

    y = oof_all["label"].to_numpy()
    pred = oof_all["pred_label"].to_numpy()
    oof_metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }

    summary = pd.DataFrame([load_fold_summary(fold) for fold in COMBINE_FOLDS])
    summary_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold_summary.csv"
    summary.to_csv(summary_path, index=False)

    watched = [
        "test_0025.png",
        "test_0539.png",
        "test_0244.png",
        "test_0490.png",
        "test_0201.png",
        "test_0222.png",
        "test_0504.png",
        "test_0141.png",
        "test_0188.png",
        "test_0568.png",
    ]
    watched_samples = {}
    for file_name in watched:
        row = test_mean.loc[test_mean["file_name"] == file_name].iloc[0]
        watched_samples[file_name] = {"pred": int(row["pred_label"]), "prob_1": float(row["prob_1"])}

    combined = {
        "exp_id": EXP_ID,
        "model": MODEL_NAME,
        "seed": SEED,
        "optimizer": "Lion",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "run_folds": FOLDS_TO_RUN,
        "folds": COMBINE_FOLDS,
        "oof_rows": int(len(oof_all)),
        "oof_duplicate_file_names": int(oof_all["file_name"].duplicated().sum()),
        "test_by_fold_rows": int(len(test_by_fold)),
        "test_mean_rows": int(len(test_mean)),
        "oof_metrics": oof_metrics,
        "test_mean_pred_counts": submission["label"].value_counts().sort_index().astype(int).to_dict(),
        "watched_samples": watched_samples,
        "paths": {
            "oof": oof_path,
            "test_by_fold": test_by_fold_path,
            "test_mean": test_mean_path,
            "fold_summary": summary_path,
            "submission": submission_path,
        },
    }
    combined_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_combined_result.json"
    Path(combined_path).write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("combined:", json.dumps(combined, indent=2), flush=True)


def main():
    configure_base_module()
    Path("oof/models").mkdir(parents=True, exist_ok=True)
    Path("oof/preds").mkdir(parents=True, exist_ok=True)
    Path("oof/submissions").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    train_all = pd.read_csv("train.csv")
    folds = pd.read_csv("oof/folds_seed42.csv")
    train_all = train_all.merge(folds[["file_name", "fold"]], on="file_name", how="inner")
    if len(train_all) != len(folds):
        raise ValueError(f"fold merge mismatch: train={len(train_all)} folds={len(folds)}")
    test_df = pd.read_csv("test.csv")

    summary_rows = []
    for fold in FOLDS_TO_RUN:
        summary_rows.append(convnext_oof.train_one_fold(fold, train_all, test_df, device))
    combine_outputs()


if __name__ == "__main__":
    main()
