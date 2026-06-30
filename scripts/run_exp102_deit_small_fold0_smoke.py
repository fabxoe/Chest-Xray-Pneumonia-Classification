import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EXP_ID = "exp102"
MODEL_NAME = "deit_small_fold0_smoke"
TIMM_MODEL = "deit3_small_patch16_224.fb_in1k"
FOLD = 0
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 2
STAGE_LR = 2e-5
HEAD_LR = 8e-5
WEIGHT_DECAY = 1e-4

PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
MODEL_DIR = Path("oof/models")

GUARDRAILS = {
    "test_0025.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 1,
    "test_0188.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0504.png": 0,
    "test_0539.png": 0,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class XrayCSV(Dataset):
    def __init__(self, df, image_dir, transform=None, has_label=True):
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


def build_model(device):
    model = timm.create_model(TIMM_MODEL, pretrained=True, num_classes=2)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.blocks[-1].parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={TIMM_MODEL} total_params={total:,} trainable_params={trainable:,}", flush=True)
    return model.to(device), total, trainable


def make_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
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
    return train_tf, eval_tf


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


def make_guardrail_report(test_prob_df):
    best = pd.read_csv("oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv")
    sample = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(test_prob_df["file_name"], test_prob_df["pred_label"]))
    sub = sample[["file_name"]].copy()
    sub["label"] = sub["file_name"].map(pred_map).astype(int)
    changed = sub.merge(best.rename(columns={"label": "exp56_label"}), on="file_name", validate="one_to_one")
    changed = changed[changed["label"].ne(changed["exp56_label"])].copy()
    changed_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_changed_vs_exp56.csv"
    changed.to_csv(changed_path, index=False)

    guardrail_rows = []
    for file_name, expected in GUARDRAILS.items():
        actual = int(sub.loc[sub["file_name"].eq(file_name), "label"].iloc[0])
        prob_1 = float(test_prob_df.loc[test_prob_df["file_name"].eq(file_name), "prob_1"].iloc[0])
        guardrail_rows.append({"file_name": file_name, "expected": expected, "actual": actual, "prob_1": prob_1, "pass": actual == expected})
    guardrail = pd.DataFrame(guardrail_rows)
    guardrail_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_guardrail.csv"
    guardrail.to_csv(guardrail_path, index=False)

    sub_path = SUB_DIR / f"{EXP_ID}_{MODEL_NAME}_submit.csv"
    sub.to_csv(sub_path, index=False)
    return sub_path, changed_path, guardrail_path, changed, guardrail


def main():
    seed_everything(SEED)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    train_all = pd.read_csv("train.csv")
    folds = pd.read_csv("oof/folds_seed42.csv")
    train_all = train_all.merge(folds[["file_name", "fold"]], on="file_name", how="inner", validate="one_to_one")
    test_df = pd.read_csv("test.csv")
    train_df = train_all[train_all["fold"].ne(FOLD)].reset_index(drop=True)
    valid_df = train_all[train_all["fold"].eq(FOLD)].reset_index(drop=True)
    print(f"fold={FOLD} train={len(train_df)} valid={len(valid_df)} test={len(test_df)}", flush=True)

    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class_weights:", class_weights.detach().cpu().numpy(), flush=True)

    train_tf, eval_tf = make_transforms()
    train_loader = DataLoader(XrayCSV(train_df, "train", train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(XrayCSV(valid_df, "train", eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(XrayCSV(test_df, "test", eval_tf, has_label=False), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    model, total_params, trainable_params = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": [p for p in model.blocks[-1].parameters() if p.requires_grad], "lr": STAGE_LR},
            {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = None
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
            f"Epoch {epoch:02d} | train_loss {metrics['train_loss']:.4f} | "
            f"valid_loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
            f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | f1 {metrics['f1']:.4f}",
            flush=True,
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model_path = MODEL_DIR / f"{EXP_ID}_{MODEL_NAME}_fold{FOLD}.pt"
    torch.save(best_state, model_path)

    final_metrics, valid_names, valid_y, valid_probs, valid_preds = evaluate(model, valid_loader, criterion, device)
    oof_df = pd.DataFrame(
        {
            "file_name": valid_names,
            "label": valid_y,
            "fold": FOLD,
            "prob_0": [p[0] for p in valid_probs],
            "prob_1": [p[1] for p in valid_probs],
            "pred_label": valid_preds,
        }
    )
    oof_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_fold{FOLD}_oof_probs.csv"
    oof_df.to_csv(oof_path, index=False)

    test_names, test_probs, test_preds = predict_test(model, test_loader, device)
    test_prob_df = pd.DataFrame(
        {
            "file_name": test_names,
            "fold": FOLD,
            "prob_0": [p[0] for p in test_probs],
            "prob_1": [p[1] for p in test_probs],
            "pred_label": test_preds,
        }
    )
    test_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_fold{FOLD}_test_probs.csv"
    test_prob_df.to_csv(test_path, index=False)

    sub_path, changed_path, guardrail_path, changed, guardrail = make_guardrail_report(test_prob_df)

    result = {
        "exp_id": EXP_ID,
        "model_name": MODEL_NAME,
        "timm_model": TIMM_MODEL,
        "fold": FOLD,
        "seed": SEED,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "stage_lr": STAGE_LR,
        "head_lr": HEAD_LR,
        "best_epoch": best_epoch,
        "history": history,
        "final_metrics": final_metrics,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_pred_counts": test_prob_df["pred_label"].value_counts().sort_index().astype(int).to_dict(),
        "changed_vs_exp56": int(len(changed)),
        "guardrail_pass_count": int(guardrail["pass"].sum()),
        "guardrail_fail_count": int((~guardrail["pass"]).sum()),
        "paths": {
            "model": str(model_path),
            "oof_probs": str(oof_path),
            "test_probs": str(test_path),
            "submission": str(sub_path),
            "changed_vs_exp56": str(changed_path),
            "guardrail": str(guardrail_path),
        },
        "elapsed_sec": round(time.time() - start, 1),
    }
    result_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_fold{FOLD}_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "exp": EXP_ID,
                "model": MODEL_NAME,
                "timm_model": TIMM_MODEL,
                "fold": FOLD,
                "best_epoch": best_epoch,
                "valid_accuracy": final_metrics["accuracy"],
                "valid_precision": final_metrics["precision"],
                "valid_recall": final_metrics["recall"],
                "valid_f1": final_metrics["f1"],
                "valid_cm": str(final_metrics["confusion_matrix"]),
                "test_label0": result["test_pred_counts"].get(0, result["test_pred_counts"].get("0", 0)),
                "test_label1": result["test_pred_counts"].get(1, result["test_pred_counts"].get("1", 0)),
                "changed_vs_exp56": len(changed),
                "guardrail_fail_count": int((~guardrail["pass"]).sum()),
                "elapsed_sec": result["elapsed_sec"],
                "submission": str(sub_path),
            }
        ]
    )
    summary_path = PRED_DIR / f"{EXP_ID}_{MODEL_NAME}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(json.dumps(result, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    if len(changed):
        print("changed_vs_exp56 head:", flush=True)
        print(changed.head(20).to_string(index=False), flush=True)
    if (~guardrail["pass"]).any():
        print("guardrail failures:", flush=True)
        print(guardrail[~guardrail["pass"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
