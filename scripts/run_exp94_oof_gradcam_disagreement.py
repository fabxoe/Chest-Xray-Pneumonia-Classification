import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
OUT = Path("reports/exp94_gradcam")
OUT.mkdir(parents=True, exist_ok=True)


def convnext_path(fold):
    if fold == 0:
        return Path("oof/models/exp45_convnext_tiny_fold0.pt")
    return Path(f"oof/models/exp46_convnext_tiny_fold{fold}.pt")


def build_convnext(fold):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(torch.load(convnext_path(fold), map_location="cpu"))
    return model.to(DEVICE).eval(), model.features[-1]


def build_resnet(fold):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(f"oof/models/exp48_resnet18_fold{fold}.pt", map_location="cpu"))
    return model.to(DEVICE).eval(), model.layer4[-1]


TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = layer.register_forward_hook(self._activation)
        self.h2 = layer.register_full_backward_hook(self._gradient)

    def _activation(self, module, inputs, output):
        self.activations = output.detach()

    def _gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def run(self, x, target=1):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        logits[:, target].sum().backward()
        weights = self.gradients.mean((2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, probs.detach().cpu().numpy()

    def close(self):
        self.h1.remove(); self.h2.remove()


def cam_stats(cam):
    h, w = cam.shape
    total = cam.sum() + 1e-8
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = float((yy * cam).sum() / total / h), float((xx * cam).sum() / total / w)
    center = cam[h // 4:3 * h // 4, w // 4:3 * w // 4].sum() / total
    top = cam[:h // 2].sum() / total
    left = cam[:, :w // 2].sum() / total
    flat = cam.ravel() / total
    entropy = float(-(flat[flat > 0] * np.log(flat[flat > 0] + 1e-12)).sum())
    return {"cam_center_fraction": float(center), "cam_top_fraction": float(top),
            "cam_left_fraction": float(left), "cam_centroid_x": cx, "cam_centroid_y": cy,
            "cam_entropy": entropy}


def overlay(gray, cam):
    base = np.asarray(ImageOps.autocontrast(gray.resize((IMG_SIZE, IMG_SIZE))), dtype=np.float32) / 255
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * np.stack([base] * 3, -1) + 0.45 * heat, 0, 1)


def main():
    features = pd.read_csv("oof/preds/exp93_disagreement_image_features.csv")
    selected = []
    for group in ["cnx0_res1", "cnx1_res0"]:
        g = features[features.group.eq(group)].copy()
        # Include both labels when available, then fill by largest confidence gap.
        for label in [0, 1]:
            selected.extend(g[g.label.eq(label)].nlargest(2, "prob_gap").to_dict("records"))
    selected = pd.DataFrame(selected).drop_duplicates("file_name").head(8)

    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 3.4 * len(selected)))
    rows = []
    for i, row in enumerate(selected.itertuples()):
        image = Image.open(Path("train") / row.file_name).convert("RGB")
        gray = image.convert("L")
        x = TF(image).unsqueeze(0).to(DEVICE)
        cams = {}
        probs = {}
        for name, builder in [("convnext", build_convnext), ("resnet", build_resnet)]:
            model, layer = builder(int(row.fold))
            runner = GradCAM(model, layer)
            cam, prob = runner.run(x, target=1)
            runner.close()
            cams[name], probs[name] = cam, prob
            del model, runner
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        threshold_c = cams["convnext"] >= np.quantile(cams["convnext"], 0.8)
        threshold_r = cams["resnet"] >= np.quantile(cams["resnet"], 0.8)
        union = np.logical_or(threshold_c, threshold_r).sum()
        iou = float(np.logical_and(threshold_c, threshold_r).sum() / max(1, union))
        corr = float(np.corrcoef(cams["convnext"].ravel(), cams["resnet"].ravel())[0, 1])

        axes[i, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[i, 0].set_title(f"{row.file_name} y={row.label}\n{row.group}")
        for j, name in enumerate(["convnext", "resnet"], start=1):
            axes[i, j].imshow(overlay(gray, cams[name]))
            axes[i, j].set_title(f"{name} p1={probs[name][1]:.3f}\nCAM target=1")
        for ax in axes[i]: ax.axis("off")

        for name in ["convnext", "resnet"]:
            item = {"file_name": row.file_name, "label": row.label, "fold": row.fold,
                    "group": row.group, "model": name, "prob_1": float(probs[name][1]),
                    "cam_pair_corr": corr, "cam_top20_iou": iou}
            item.update(cam_stats(cams[name]))
            rows.append(item)

    plt.tight_layout()
    panel = OUT / "oof_gradcam_disagreement_panel.png"
    fig.savefig(panel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "oof_gradcam_disagreement_summary.csv", index=False)
    group_summary = detail.groupby(["group", "model"]).agg(
        n=("file_name", "size"), prob1_mean=("prob_1", "mean"),
        center_fraction=("cam_center_fraction", "mean"), top_fraction=("cam_top_fraction", "mean"),
        left_fraction=("cam_left_fraction", "mean"), centroid_x=("cam_centroid_x", "mean"),
        centroid_y=("cam_centroid_y", "mean"), entropy=("cam_entropy", "mean"),
        pair_corr=("cam_pair_corr", "mean"), top20_iou=("cam_top20_iou", "mean"),
    ).reset_index()
    group_summary.to_csv(OUT / "oof_gradcam_group_summary.csv", index=False)
    result = {"samples": len(selected), "panel": str(panel),
              "mean_pair_corr": float(detail.drop_duplicates("file_name").cam_pair_corr.mean()),
              "mean_top20_iou": float(detail.drop_duplicates("file_name").cam_top20_iou.mean()),
              "group_summary": group_summary.to_dict("records")}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
