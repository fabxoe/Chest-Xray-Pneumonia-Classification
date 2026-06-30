from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/exp112_drop_candidate_gradcam"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = ROOT / "oof/models/exp45_convnext_tiny_fold0.pt"

TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = layer.register_forward_hook(self._activation)
        self.h2 = layer.register_full_backward_hook(self._gradient)

    def _activation(self, module, inputs, output):
        self.activations = output.detach()

    def _gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def run(self, x: torch.Tensor, target: int = 1) -> tuple[np.ndarray, np.ndarray]:
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

    def close(self) -> None:
        self.h1.remove()
        self.h2.remove()


def build_model() -> tuple[nn.Module, nn.Module]:
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = model.to(DEVICE).eval()
    return model, model.features[-1]


def overlay(gray: Image.Image, cam: np.ndarray) -> np.ndarray:
    base = np.asarray(ImageOps.autocontrast(gray.resize((IMG_SIZE, IMG_SIZE))), dtype=np.float32) / 255
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * np.stack([base] * 3, -1) + 0.45 * heat, 0, 1)


def cam_stats(cam: np.ndarray) -> dict[str, float]:
    h, w = cam.shape
    total = float(cam.sum() + 1e-8)
    yy, xx = np.mgrid[0:h, 0:w]
    center = cam[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].sum() / total
    top = cam[: h // 2].sum() / total
    left = cam[:, : w // 2].sum() / total
    cx = float((xx * cam).sum() / total / w)
    cy = float((yy * cam).sum() / total / h)
    return {
        "cam_center_fraction": float(center),
        "cam_top_fraction": float(top),
        "cam_left_fraction": float(left),
        "cam_centroid_x": cx,
        "cam_centroid_y": cy,
    }


def load_selected() -> pd.DataFrame:
    rows = []
    exp110 = pd.read_csv(ROOT / "oof/preds/exp110_convnext_fp_drop_candidates_summary.csv")
    for row in exp110.itertuples():
        rows.append({"file_name": row.changed_sample, "source": f"EXP-110 rank{row.rank}"})
    exp111 = pd.read_csv(ROOT / "oof/preds/exp111_convnext_threshold_replay_summary.csv")
    for row in exp111.itertuples():
        for file_name in str(row.changed_samples).split(";"):
            rows.append({"file_name": file_name, "source": f"EXP-111 {row.score_spec} drop{row.drop_count}"})
    selected = pd.DataFrame(rows).drop_duplicates("file_name").head(10)

    context = pd.read_csv(ROOT / "reports/exp95_suspicious_cases/exp95_suspicious_cases.csv")
    keep = [
        "file_name",
        "exp56_label",
        "exp56_score_margin",
        "prob_mean",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "suspicious_score",
        "recommended_action",
    ]
    return selected.merge(context[[c for c in keep if c in context.columns]], on="file_name", how="left")


def main() -> None:
    selected = load_selected()
    if selected.empty:
        raise ValueError("No EXP-110/111 candidates found")

    model, layer = build_model()
    runner = GradCAM(model, layer)

    fig, axes = plt.subplots(len(selected), 2, figsize=(8.5, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])
    rows = []
    for i, row in enumerate(selected.itertuples(index=False)):
        image = Image.open(ROOT / "test" / row.file_name).convert("RGB")
        gray = image.convert("L")
        x = TF(image).unsqueeze(0).to(DEVICE)
        cam, probs = runner.run(x, target=1)
        axes[i, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[i, 0].set_title(f"{row.file_name}\n{row.source}")
        axes[i, 1].imshow(overlay(gray, cam))
        axes[i, 1].set_title(
            f"ConvNeXt fold0 CAM target=1\np1={probs[1]:.3f}, exp56_margin={row.exp56_score_margin:.3f}"
        )
        for ax in axes[i]:
            ax.axis("off")
        item = row._asdict()
        item["convnext_fold0_prob1"] = float(probs[1])
        item.update(cam_stats(cam))
        rows.append(item)

    runner.close()
    plt.tight_layout()
    panel = OUT / "exp112_drop_candidate_gradcam_panel.png"
    fig.savefig(panel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "exp112_drop_candidate_gradcam_summary.csv", index=False)
    result = {
        "samples": len(detail),
        "panel": str(panel),
        "summary": str(OUT / "exp112_drop_candidate_gradcam_summary.csv"),
        "device": str(DEVICE),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
