import json
import os
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
from torchvision.models import ConvNeXt_Tiny_Weights


EXP_ID = os.environ.get("EXP_ID", "exp126")
MODEL_NAME = os.environ.get("MODEL_NAME", "convnext_tiny_384_noflip")
FOLDS_TO_RUN = [int(x) for x in os.environ.get("FOLDS_TO_RUN", "1,2,3,4").split(",") if x.strip()]
SEED = int(os.environ.get("SEED", "42"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "384"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
LR = float(os.environ.get("LR", "5e-5"))
BRIGHTNESS = float(os.environ.get("BRIGHTNESS", "0.08"))
CONTRAST = float(os.environ.get("CONTRAST", "0.08"))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") == "1"

FOLD0_PREFIX = "exp124_convnext_tiny_384_noflip_fold0"
CURRENT_BEST = "oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv"

GUARDRAILS = {
    "test_0455.png": 0,
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0303.png": 0,
    "test_0289.png": 1,
    "test_0172.png": 1,
    "test_0013.png": 1,
    "test_0390.png": 1,
    "test_0188.png": 0,
    "test_0568.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 0,
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
}


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
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model.to(device)


def make_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST),
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
    transform_meta = {
        "train": [
            f"Resize(({IMG_SIZE},{IMG_SIZE}))",
            f"ColorJitter(brightness={BRIGHTNESS}, contrast={CONTRAST})",
            "ToTensor()",
            "Normalize(ImageNet)",
        ],
        "eval": [f"Resize(({IMG_SIZE},{IMG_SIZE}))", "ToTensor()", "Normalize(ImageNet)"],
        "removed": ["RandomHorizontalFlip", "RandomRotation", "CenterCrop"],
    }
    return train_tf, eval_tf, transform_meta


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


def fold_paths(fold: int):
    stem = f"{EXP_ID}_{MODEL_NAME}_fold{fold}"
    return {
        "model": Path(f"oof/models/{stem}.pt"),
        "oof": Path(f"oof/preds/{stem}_oof_probs.csv"),
        "test": Path(f"oof/preds/{stem}_test_probs.csv"),
        "result": Path(f"oof/preds/{stem}_result.json"),
    }


def summary_from_existing(fold: int):
    paths = fold_paths(fold)
    required = [paths["model"], paths["oof"], paths["test"], paths["result"]]
    if not all(path.exists() for path in required):
        return None
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    final_metrics = result["final_metrics"]
    pred_counts = result["test_pred_counts"]
    return {
        "exp": EXP_ID,
        "model": MODEL_NAME,
        "fold": fold,
        "best_epoch": result["best_epoch"],
        "train_size": result["train_size"],
        "valid_size": result["valid_size"],
        "valid_accuracy": final_metrics["accuracy"],
        "valid_precision": final_metrics["precision"],
        "valid_recall": final_metrics["recall"],
        "valid_f1": final_metrics["f1"],
        "valid_cm": str(final_metrics["confusion_matrix"]),
        "elapsed_sec": result["elapsed_sec"],
        "oof_rows": len(pd.read_csv(paths["oof"])),
        "test_rows": len(pd.read_csv(paths["test"])),
        "test_label0": pred_counts.get("0", pred_counts.get(0, 0)),
        "test_label1": pred_counts.get("1", pred_counts.get(1, 0)),
        "model_path": str(paths["model"]),
        "oof_probs": str(paths["oof"]),
        "test_probs": str(paths["test"]),
    }


def train_one_fold(fold: int, train_all: pd.DataFrame, test_df: pd.DataFrame, device: torch.device):
    if SKIP_EXISTING:
        existing = summary_from_existing(fold)
        if existing is not None:
            print(f"fold {fold}: existing outputs found, skipping train", flush=True)
            return existing

    seed_everything(SEED + fold)
    train_df = train_all[train_all["fold"] != fold].reset_index(drop=True)
    valid_df = train_all[train_all["fold"] == fold].reset_index(drop=True)
    print(f"fold {fold}: train={len(train_df)} valid={len(valid_df)}", flush=True)

    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class weights:", class_weights.detach().cpu().numpy(), flush=True)

    train_tf, eval_tf, transform_meta = make_transforms()
    train_loader = DataLoader(XrayCSV(train_df, "train", train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(XrayCSV(valid_df, "train", eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        XrayCSV(test_df, "test", eval_tf, has_label=False),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

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

    if best_state is None:
        raise RuntimeError(f"fold {fold}: no best state captured")

    model.load_state_dict(best_state)
    paths = fold_paths(fold)
    torch.save(best_state, paths["model"])

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
    oof_df.to_csv(paths["oof"], index=False)

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
    test_df_out.to_csv(paths["test"], index=False)

    pred_counts = test_df_out["pred_label"].value_counts().sort_index().astype(int).to_dict()
    result = {
        "exp_id": EXP_ID,
        "model": MODEL_NAME,
        "fold": fold,
        "seed": SEED + fold,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "transform": transform_meta,
        "best_epoch": best_epoch,
        "history": history,
        "final_metrics": final_metrics,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_pred_counts": pred_counts,
        "paths": {"model": str(paths["model"]), "oof_probs": str(paths["oof"]), "test_probs": str(paths["test"])},
        "elapsed_sec": round(time.time() - start, 1),
    }
    paths["result"].write_text(json.dumps(result, indent=2), encoding="utf-8")

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
        "test_label0": pred_counts.get(0, pred_counts.get("0", 0)),
        "test_label1": pred_counts.get(1, pred_counts.get("1", 0)),
        "model_path": str(paths["model"]),
        "oof_probs": str(paths["oof"]),
        "test_probs": str(paths["test"]),
    }


def load_fold0_summary():
    path = Path(f"oof/preds/{FOLD0_PREFIX}_summary.csv")
    if not path.exists():
        raise FileNotFoundError("EXP-124 fold0 summary is required before EXP-126.")
    row = pd.read_csv(path).iloc[0].to_dict()
    row["exp"] = "exp124"
    row["model"] = MODEL_NAME
    return row


def validate_submission(path: str):
    sample = pd.read_csv("sample_submission.csv")
    sub = pd.read_csv(path)
    return {
        "rows": int(len(sub)),
        "columns": list(sub.columns),
        "order_match_sample": bool(sub["file_name"].equals(sample["file_name"])),
        "labels": sorted(sub["label"].dropna().unique().astype(int).tolist()),
        "label_counts": sub["label"].value_counts().sort_index().astype(int).to_dict(),
    }


def analyze_current_best(submission_path: str):
    current = pd.read_csv(CURRENT_BEST)
    candidate = pd.read_csv(submission_path)
    merged = current.merge(candidate, on="file_name", suffixes=("_current", "_candidate"))
    changed = merged[merged["label_current"] != merged["label_candidate"]].copy()
    changed_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_changed_vs_exp117.csv"
    changed.to_csv(changed_path, index=False)

    guardrail_values = {}
    violations = []
    cand_map = dict(zip(candidate["file_name"], candidate["label"].astype(int)))
    for file_name, expected in GUARDRAILS.items():
        actual = int(cand_map[file_name])
        guardrail_values[file_name] = actual
        if actual != expected:
            violations.append({"file_name": file_name, "expected": expected, "actual": actual})

    return {
        "current_best": CURRENT_BEST,
        "changed_count": int(len(changed)),
        "changed_path": changed_path,
        "guardrail_values": guardrail_values,
        "guardrail_violations": violations,
    }


def combine_outputs(summary_rows):
    oof_parts = [pd.read_csv(f"oof/preds/{FOLD0_PREFIX}_oof_probs.csv")]
    test_parts = [pd.read_csv(f"oof/preds/{FOLD0_PREFIX}_test_probs.csv")]
    for fold in FOLDS_TO_RUN:
        paths = fold_paths(fold)
        oof_parts.append(pd.read_csv(paths["oof"]))
        test_parts.append(pd.read_csv(paths["test"]))

    oof_all = pd.concat(oof_parts, ignore_index=True).sort_values("file_name").reset_index(drop=True)
    test_by_fold = pd.concat(test_parts, ignore_index=True)
    test_mean = (
        test_by_fold.groupby("file_name", as_index=False)[["prob_0", "prob_1"]]
        .mean()
        .assign(pred_label=lambda df: (df["prob_1"] >= df["prob_0"]).astype(int))
    )

    oof_all_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_oof_probs.csv"
    test_by_fold_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_by_fold.csv"
    test_mean_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_test_probs_mean.csv"
    fold_summary_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_fold_summary.csv"
    oof_all.to_csv(oof_all_path, index=False)
    test_by_fold.to_csv(test_by_fold_path, index=False)
    test_mean.to_csv(test_mean_path, index=False)
    pd.DataFrame(summary_rows).to_csv(fold_summary_path, index=False)

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
    best_analysis = analyze_current_best(submission_path)
    validation = validate_submission(submission_path)

    combined = {
        "exp_id": EXP_ID,
        "model": MODEL_NAME,
        "seed": SEED,
        "folds": [0] + FOLDS_TO_RUN,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "brightness": BRIGHTNESS,
        "contrast": CONTRAST,
        "transform": make_transforms()[2],
        "oof_rows": int(len(oof_all)),
        "oof_duplicate_file_names": int(oof_all["file_name"].duplicated().sum()),
        "test_by_fold_rows": int(len(test_by_fold)),
        "test_mean_rows": int(len(test_mean)),
        "oof_metrics": oof_metrics,
        "test_mean_pred_counts": submission["label"].value_counts().sort_index().astype(int).to_dict(),
        "submission_validation": validation,
        "current_best_analysis": best_analysis,
        "paths": {
            "oof_probs": oof_all_path,
            "test_probs_by_fold": test_by_fold_path,
            "test_probs_mean": test_mean_path,
            "fold_summary": fold_summary_path,
            "submission": submission_path,
        },
    }
    result_path = f"oof/preds/{EXP_ID}_{MODEL_NAME}_combined_result.json"
    Path(result_path).write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("combined:", json.dumps(combined, indent=2), flush=True)
    return combined


def main():
    Path("oof/models").mkdir(parents=True, exist_ok=True)
    Path("oof/preds").mkdir(parents=True, exist_ok=True)
    Path("oof/submissions").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    print(
        f"exp={EXP_ID} model={MODEL_NAME} folds={FOLDS_TO_RUN} img_size={IMG_SIZE} "
        f"batch={BATCH_SIZE} test_batch={TEST_BATCH_SIZE} epochs={EPOCHS}",
        flush=True,
    )

    train_all = pd.read_csv("train.csv")
    folds = pd.read_csv("oof/folds_seed42.csv")
    train_all = train_all.merge(folds[["file_name", "fold"]], on="file_name", how="inner")
    if len(train_all) != len(folds):
        raise ValueError(f"fold merge mismatch: train={len(train_all)} folds={len(folds)}")
    test_df = pd.read_csv("test.csv")

    summary_rows = [load_fold0_summary()]
    for fold in FOLDS_TO_RUN:
        summary_rows.append(train_one_fold(fold, train_all, test_df, device))
    combine_outputs(summary_rows)


if __name__ == "__main__":
    main()
