import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import RegNet_Y_3_2GF_Weights


EXP_ID = "exp67"
MODEL_NAME = "regnet_y_3_2gf"
FOLDS_TO_RUN = [0, 1, 2, 3, 4]
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 6
LAST_FEATURE_LR = 1e-5
HEAD_LR = 5e-5
WEIGHT_DECAY = 1e-4

WATCH_SAMPLES = [
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class XrayCSV(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None, has_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["file_name"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.has_label:
            return image, torch.tensor(row["label"], dtype=torch.long), row["file_name"]
        return image, row["file_name"]


def build_model(device: torch.device):
    model = models.regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.trunk_output.block4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, probs, names = [], [], [], []

    for images, labels, file_names in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
        names.extend(file_names)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics, names, y_true, probs, y_pred


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    names, probs, preds = [], [], []

    for images, file_names in loader:
        images = images.to(device)
        logits = model(images)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        names.extend(file_names)
        probs.extend(prob.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())

    return names, probs, preds


def train_one_fold(fold: int, train_all: pd.DataFrame, test_df: pd.DataFrame, device: torch.device):
    seed_everything(SEED + fold)
    train_df = train_all[train_all["fold"] != fold].reset_index(drop=True)
    valid_df = train_all[train_all["fold"] == fold].reset_index(drop=True)
    print(f"fold {fold}: train={len(train_df)} valid={len(valid_df)}", flush=True)

    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class weights:", class_weights.detach().cpu().numpy(), flush=True)

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(XrayCSV(train_df, "train", train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(XrayCSV(valid_df, "train", eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(XrayCSV(test_df, "test", eval_tf, has_label=False), batch_size=48, shuffle=False, num_workers=0)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": model.trunk_output.block4.parameters(), "lr": LAST_FEATURE_LR},
            {"params": model.fc.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("trainable params:", trainable_params, "total params:", total_params, flush=True)

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            if batch_idx % 60 == 0:
                print(f"  fold {fold} epoch {epoch:02d} batch {batch_idx}/{len(train_loader)}", flush=True)

        metrics, *_ = evaluate(model, valid_loader, criterion, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / len(train_loader.dataset)
        history.append(metrics)
        print(
            f"Fold {fold} Epoch {epoch:02d} | train_loss {metrics['train_loss']:.4f} | "
            f"valid_loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
            f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | f1 {metrics['f1']:.4f}",
            flush=True,
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model_path = Path(f"oof/models/{EXP_ID}_{MODEL_NAME}_fold{fold}.pt")
    torch.save(best_state, model_path)

    final_metrics, valid_names, valid_y, valid_probs, valid_preds = evaluate(model, valid_loader, criterion, device)
    oof_df = pd.DataFrame(
        {
            "file_name": valid_names,
            "label": valid_y,
            "fold": fold,
            "prob_0": [p[0] for p in valid_probs],
            "prob_1": [p[1] for p in valid_probs],
            "pred_label": valid_preds,
        }
    )
    oof_path = Path(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_oof_probs.csv")
    oof_df.to_csv(oof_path, index=False)

    test_names, test_probs, test_preds = predict_test(model, test_loader, device)
    test_df_out = pd.DataFrame(
        {
            "file_name": test_names,
            "fold": fold,
            "prob_0": [p[0] for p in test_probs],
            "prob_1": [p[1] for p in test_probs],
            "pred_label": test_preds,
        }
    )
    test_path = Path(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_test_probs.csv")
    test_df_out.to_csv(test_path, index=False)

    result = {
        "exp_id": EXP_ID,
        "model": MODEL_NAME,
        "fold": fold,
        "seed": SEED + fold,
        "best_epoch": best_epoch,
        "history": history,
        "final_metrics": final_metrics,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_pred_counts": test_df_out["pred_label"].value_counts().sort_index().astype(int).to_dict(),
        "paths": {"model": str(model_path), "oof_probs": str(oof_path), "test_probs": str(test_path)},
        "elapsed_sec": round(time.time() - start, 1),
    }
    result_path = Path(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_result.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return {
        "exp": EXP_ID,
        "model": MODEL_NAME,
        "fold": fold,
        "best_epoch": best_epoch,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "valid_accuracy": final_metrics["accuracy"],
        "valid_precision": final_metrics["precision"],
        "valid_recall": final_metrics["recall"],
        "valid_f1": final_metrics["f1"],
        "valid_cm": str(final_metrics["confusion_matrix"]),
        "elapsed_sec": result["elapsed_sec"],
        "oof_rows": len(oof_df),
        "test_rows": len(test_df_out),
        "test_label0": result["test_pred_counts"].get(0, result["test_pred_counts"].get("0", 0)),
        "test_label1": result["test_pred_counts"].get(1, result["test_pred_counts"].get("1", 0)),
        "model_path": str(model_path),
        "oof_probs": str(oof_path),
        "test_probs": str(test_path),
    }


def combine_outputs(summary_rows):
    oof_parts = []
    test_parts = []
    for fold in FOLDS_TO_RUN:
        oof_parts.append(pd.read_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_oof_probs.csv"))
        test_parts.append(pd.read_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold{fold}_test_probs.csv"))

    oof_all = pd.concat(oof_parts, ignore_index=True).sort_values("file_name").reset_index(drop=True)
    test_by_fold = pd.concat(test_parts, ignore_index=True)
    test_mean = (
        test_by_fold.groupby("file_name", as_index=False)[["prob_0", "prob_1"]]
        .mean()
        .assign(pred_label=lambda df: (df["prob_1"] >= df["prob_0"]).astype(int))
    )

    oof_all.to_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_oof_probs.csv", index=False)
    test_by_fold.to_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_by_fold.csv", index=False)
    test_mean.to_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_mean.csv", index=False)

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

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold_summary.csv", index=False)

    watched = {}
    for name in WATCH_SAMPLES:
        row = test_mean.loc[test_mean["file_name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        watched[name] = {"pred": int(row["pred_label"]), "prob_1": float(row["prob_1"])}

    current_best = pd.read_csv("oof/preds/exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv")
    current_best = current_best.sort_values("file_name").reset_index(drop=True)
    test_mean_sorted = test_mean.sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test_mean_sorted["file_name"].to_numpy(), current_best["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-56")
    changed = test_mean_sorted.loc[test_mean_sorted["pred_label"].to_numpy() != current_best["pred"].to_numpy(), ["file_name", "prob_1", "pred_label"]].copy()
    changed["exp56_pred"] = current_best.loc[test_mean_sorted["pred_label"].to_numpy() != current_best["pred"].to_numpy(), "pred"].to_numpy()
    changed.to_csv(f"oof/preds/{EXP_ID}_{MODEL_NAME}_changed_vs_exp56.csv", index=False)

    combined = {
        "exp_id": EXP_ID,
        "model": MODEL_NAME,
        "oof_rows": int(len(oof_all)),
        "oof_duplicate_file_names": int(oof_all["file_name"].duplicated().sum()),
        "test_by_fold_rows": int(len(test_by_fold)),
        "test_mean_rows": int(len(test_mean)),
        "oof_metrics": oof_metrics,
        "test_mean_pred_counts": submission["label"].value_counts().sort_index().astype(int).to_dict(),
        "watched_samples": watched,
        "changed_vs_exp56_count": int(len(changed)),
        "submission": submission_path,
    }
    Path(f"oof/preds/{EXP_ID}_{MODEL_NAME}_combined_result.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print("combined:", json.dumps(combined, indent=2), flush=True)


def main():
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
        summary_rows.append(train_one_fold(fold, train_all, test_df, device))
    combine_outputs(summary_rows)


if __name__ == "__main__":
    main()
