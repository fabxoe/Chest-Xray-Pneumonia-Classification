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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import RegNet_Y_3_2GF_Weights


EXP_ID = "exp43"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 6
LAST_FEATURE_LR = 1e-5
HEAD_LR = 5e-5
WEIGHT_DECAY = 1e-4


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
    weights = RegNet_Y_3_2GF_Weights.IMAGENET1K_V1
    model = models.regnet_y_3_2gf(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    for param in model.parameters():
        param.requires_grad = False

    # First probe: train the final RegNet stage and classifier.
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


def main():
    seed_everything(SEED)
    Path("models").mkdir(exist_ok=True)
    Path("preds").mkdir(exist_ok=True)
    Path("submissions").mkdir(exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device, flush=True)

    train_all = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    train_df, valid_df = train_test_split(
        train_all,
        test_size=0.2,
        stratify=train_all["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

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

    train_loader = DataLoader(
        XrayCSV(train_df, "train", train_tf, has_label=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        XrayCSV(valid_df, "train", eval_tf, has_label=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        XrayCSV(test_df, "test", eval_tf, has_label=False),
        batch_size=48,
        shuffle=False,
        num_workers=0,
    )

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
            if batch_idx % 30 == 0:
                print(f"  epoch {epoch:02d} batch {batch_idx}/{len(train_loader)}", flush=True)

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
    model_path = Path("models/exp43_regnet_y_3_2gf_best.pt")
    torch.save(best_state, model_path)

    final_metrics, valid_names, valid_y, valid_probs, valid_preds = evaluate(model, valid_loader, criterion, device)
    valid_prob_df = pd.DataFrame(
        {
            "file_name": valid_names,
            "label": valid_y,
            "prob_1": [p[1] for p in valid_probs],
            "pred": valid_preds,
        }
    )
    valid_prob_path = Path("preds/exp43_regnet_y_3_2gf_valid_probs.csv")
    valid_prob_df.to_csv(valid_prob_path, index=False)

    test_names, test_probs, test_preds = predict_test(model, test_loader, device)
    test_prob_df = pd.DataFrame(
        {
            "file_name": test_names,
            "prob_1": [p[1] for p in test_probs],
            "pred": test_preds,
        }
    )
    test_prob_path = Path("preds/exp43_regnet_y_3_2gf_test_probs.csv")
    test_prob_df.to_csv(test_prob_path, index=False)

    submission = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(test_names, test_preds))
    submission["label"] = submission["file_name"].map(pred_map).astype(int)
    submission_path = Path("submissions/exp43_regnet_y_3_2gf_submit.csv")
    submission.to_csv(submission_path, index=False)

    exp26 = pd.read_csv("submissions/exp26_ridge_without_densenet_alpha30_submit.csv")
    changed = submission.merge(exp26, on="file_name", suffixes=("_exp43", "_exp26"))
    changed = changed[changed["label_exp43"] != changed["label_exp26"]].copy()
    changed.to_csv("preds/exp43_regnet_y_3_2gf_changed_vs_exp26.csv", index=False)

    result = {
        "exp_id": EXP_ID,
        "model": "regnet_y_3_2gf",
        "best_epoch": best_epoch,
        "history": history,
        "final_metrics": final_metrics,
        "test_pred_counts": submission["label"].value_counts().sort_index().astype(int).to_dict(),
        "changed_vs_exp26": int(len(changed)),
        "test_0244_pred": int(submission.loc[submission.file_name == "test_0244.png", "label"].iloc[0]),
        "test_0025_pred": int(submission.loc[submission.file_name == "test_0025.png", "label"].iloc[0]),
        "test_0539_pred": int(submission.loc[submission.file_name == "test_0539.png", "label"].iloc[0]),
        "paths": {
            "model": str(model_path),
            "valid_probs": str(valid_prob_path),
            "test_probs": str(test_prob_path),
            "submission": str(submission_path),
        },
        "elapsed_sec": round(time.time() - start, 1),
    }
    result_path = Path("preds/exp43_regnet_y_3_2gf_result.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "exp": EXP_ID,
                "model": "RegNetY-3.2GF",
                "best_epoch": best_epoch,
                "valid_accuracy": final_metrics["accuracy"],
                "valid_precision": final_metrics["precision"],
                "valid_recall": final_metrics["recall"],
                "valid_f1": final_metrics["f1"],
                "valid_cm": str(final_metrics["confusion_matrix"]),
                "test_label0": result["test_pred_counts"].get(0, result["test_pred_counts"].get("0", 0)),
                "test_label1": result["test_pred_counts"].get(1, result["test_pred_counts"].get("1", 0)),
                "changed_vs_exp26": result["changed_vs_exp26"],
                "test_0244_pred": result["test_0244_pred"],
                "test_0025_pred": result["test_0025_pred"],
                "test_0539_pred": result["test_0539_pred"],
                "submission": str(submission_path),
            }
        ]
    )
    summary_path = Path("preds/exp43_regnet_y_3_2gf_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("best_epoch:", best_epoch, flush=True)
    print("final_metrics:", final_metrics, flush=True)
    print("test_pred_counts:", result["test_pred_counts"], flush=True)
    print("changed_vs_exp26:", result["changed_vs_exp26"], flush=True)
    print("sensitive:", {k: result[k] for k in ["test_0244_pred", "test_0025_pred", "test_0539_pred"]}, flush=True)
    print("submission:", submission_path, flush=True)
    print("elapsed_sec:", result["elapsed_sec"], flush=True)


if __name__ == "__main__":
    main()
