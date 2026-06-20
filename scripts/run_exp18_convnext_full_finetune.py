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
from torchvision.models import ConvNeXt_Tiny_Weights


EXP_ID = "exp18"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-5
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
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)

    for param in model.parameters():
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
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
    print("device:", device)

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
    print("class weights:", class_weights.detach().cpu().numpy())

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
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("trainable params:", trainable_params, "total params:", total_params)

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
            f"Epoch {epoch:02d} | train_loss {metrics['train_loss']:.4f} | "
            f"valid_loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
            f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | f1 {metrics['f1']:.4f}"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model_path = Path("models/exp18_convnext_tiny_full_finetune_best.pt")
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
    valid_prob_path = Path("preds/exp18_convnext_tiny_full_finetune_valid_probs.csv")
    valid_prob_df.to_csv(valid_prob_path, index=False)

    test_names, test_probs, test_preds = predict_test(model, test_loader, device)
    test_prob_df = pd.DataFrame(
        {
            "file_name": test_names,
            "prob_1": [p[1] for p in test_probs],
            "pred": test_preds,
        }
    )
    test_prob_path = Path("preds/exp18_convnext_tiny_full_finetune_test_probs.csv")
    test_prob_df.to_csv(test_prob_path, index=False)

    submission = pd.read_csv("sample_submission.csv")
    pred_map = dict(zip(test_names, test_preds))
    submission["label"] = submission["file_name"].map(pred_map).astype(int)
    submission_path = Path("submissions/exp18_convnext_tiny_full_finetune_submit.csv")
    submission.to_csv(submission_path, index=False)

    result = {
        "exp_id": EXP_ID,
        "model": "convnext_tiny_full_finetune",
        "best_epoch": best_epoch,
        "history": history,
        "final_metrics": final_metrics,
        "test_pred_counts": submission["label"].value_counts().sort_index().astype(int).to_dict(),
        "paths": {
            "model": str(model_path),
            "valid_probs": str(valid_prob_path),
            "test_probs": str(test_prob_path),
            "submission": str(submission_path),
        },
        "settings": {
            "backbone_lr": BACKBONE_LR,
            "classifier_lr": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        "elapsed_sec": round(time.time() - start, 1),
    }
    result_path = Path("preds/exp18_convnext_tiny_full_finetune_result.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("best_epoch:", best_epoch)
    print("final_metrics:", final_metrics)
    print("test_pred_counts:", result["test_pred_counts"])
    print("submission:", submission_path)
    print("elapsed_sec:", result["elapsed_sec"])


if __name__ == "__main__":
    main()
