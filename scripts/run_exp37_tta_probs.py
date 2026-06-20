import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMG_SIZE = 224
BATCH_SIZE = 48
PRED_DIR = Path("preds")
PRED_DIR.mkdir(exist_ok=True)


MODEL_SPECS = {
    "resnet18": {
        "path": "models/exp07_resnet18_layer4_fc_best.pt",
        "valid_ref": "preds/exp07_resnet18_valid_probs.csv",
        "builder": "resnet18",
    },
    "efficientnet_b0": {
        "path": "models/exp06_efficientnet_b0_best.pt",
        "valid_ref": "preds/exp06_efficientnet_b0_valid_probs.csv",
        "builder": "efficientnet_b0",
    },
    "convnext12": {
        "path": "models/exp12_convnext_tiny_best.pt",
        "valid_ref": "preds/exp12_convnext_tiny_valid_probs.csv",
        "builder": "convnext_tiny",
    },
    "convnext18_full": {
        "path": "models/exp18_convnext_tiny_full_finetune_best.pt",
        "valid_ref": "preds/exp18_convnext_tiny_full_finetune_valid_probs.csv",
        "builder": "convnext_tiny",
    },
}


class XrayDataset(Dataset):
    def __init__(self, df, image_dir, transform, has_label):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["file_name"]).convert("RGB")
        image = self.transform(image)
        if self.has_label:
            return image, int(row["label"]), row["file_name"]
        return image, row["file_name"]


def load_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def build_model(kind, path, device):
    if kind == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif kind == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif kind == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.load_state_dict(load_state(path))
    model.to(device)
    model.eval()
    return model


def make_transform(hflip=False):
    steps = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if hflip:
        steps.append(transforms.Lambda(lambda image: image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


@torch.no_grad()
def predict_one_transform(model, df, image_dir, transform, has_label, device):
    loader = DataLoader(
        XrayDataset(df, image_dir, transform, has_label),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    names, labels, probs = [], [], []
    for batch in loader:
        if has_label:
            images, y, file_names = batch
            labels.extend([int(v) for v in y])
        else:
            images, file_names = batch
        images = images.to(device)
        prob = torch.softmax(model(images), dim=1).detach().cpu().numpy()
        probs.append(prob)
        names.extend(file_names)
    probs = np.concatenate(probs, axis=0)
    return names, labels if has_label else None, probs


def metrics(y_true, prob_1):
    pred = (prob_1 >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_model(name, spec, valid_df, test_df, device):
    start = time.time()
    model = build_model(spec["builder"], spec["path"], device)
    tfs = {
        "orig": make_transform(hflip=False),
        "hflip": make_transform(hflip=True),
    }

    valid_probs = []
    valid_names = None
    valid_y = None
    for tf_name, tf in tfs.items():
        names, labels, probs = predict_one_transform(model, valid_df, "train", tf, True, device)
        if valid_names is None:
            valid_names = names
            valid_y = labels
        valid_probs.append(probs)
        print(f"{name} valid {tf_name} done")
    valid_mean = np.mean(valid_probs, axis=0)
    valid_prob_1 = valid_mean[:, 1]
    valid_pred = (valid_prob_1 >= 0.5).astype(int)
    valid_out = pd.DataFrame(
        {
            "file_name": valid_names,
            "label": valid_y,
            "prob_1": valid_prob_1,
            "pred": valid_pred,
        }
    )
    valid_path = PRED_DIR / f"exp37_tta_{name}_valid_probs.csv"
    valid_out.to_csv(valid_path, index=False)

    test_probs = []
    test_names = None
    for tf_name, tf in tfs.items():
        names, _, probs = predict_one_transform(model, test_df, "test", tf, False, device)
        if test_names is None:
            test_names = names
        test_probs.append(probs)
        print(f"{name} test {tf_name} done")
    test_mean = np.mean(test_probs, axis=0)
    test_prob_1 = test_mean[:, 1]
    test_pred = (test_prob_1 >= 0.5).astype(int)
    test_out = pd.DataFrame(
        {
            "file_name": test_names,
            "prob_1": test_prob_1,
            "pred": test_pred,
        }
    )
    test_path = PRED_DIR / f"exp37_tta_{name}_test_probs.csv"
    test_out.to_csv(test_path, index=False)

    ref = pd.read_csv(spec["valid_ref"])
    ref_metric = metrics(ref["label"].to_numpy(), ref["prob_1"].to_numpy())
    tta_metric = metrics(valid_out["label"].to_numpy(), valid_out["prob_1"].to_numpy())
    test_ref = pd.read_csv(spec["valid_ref"].replace("_valid_", "_test_"))
    changed_test = int((test_out["pred"].to_numpy() != test_ref["pred"].to_numpy()).sum())
    row = {
        "model": name,
        "tta": "orig+hflip",
        "valid_accuracy_before": ref_metric["accuracy"],
        "valid_accuracy_tta": tta_metric["accuracy"],
        "valid_recall_before": ref_metric["recall"],
        "valid_recall_tta": tta_metric["recall"],
        "valid_fp_tta": tta_metric["fp"],
        "valid_fn_tta": tta_metric["fn"],
        "test_label0_tta": int((test_pred == 0).sum()),
        "test_label1_tta": int((test_pred == 1).sum()),
        "changed_test_vs_original": changed_test,
        "valid_probs": str(valid_path),
        "test_probs": str(test_path),
        "elapsed_sec": round(time.time() - start, 1),
    }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return row


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)
    valid_ref = pd.read_csv("preds/exp12_convnext_tiny_valid_probs.csv")
    valid_df = valid_ref[["file_name", "label"]].copy()
    test_df = pd.read_csv("test.csv")

    rows = []
    for name, spec in MODEL_SPECS.items():
        print(f"running TTA: {name}")
        rows.append(run_model(name, spec, valid_df, test_df, device))
        print(rows[-1])

    summary = pd.DataFrame(rows)
    summary_path = PRED_DIR / "exp37_tta_summary.csv"
    summary.to_csv(summary_path, index=False)
    result_path = PRED_DIR / "exp37_tta_result.json"
    result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("saved:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
