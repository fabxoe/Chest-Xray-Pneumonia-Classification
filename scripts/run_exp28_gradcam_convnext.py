from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import models, transforms


IMG_SIZE = 224
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SAMPLES = [
    "test_0244.png",
    "test_0267.png",
    "test_0371.png",
    "test_0535.png",
    "test_0049.png",
    "test_0342.png",
]


def build_convnext(model_path: str):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(image: Image.Image):
    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(image.convert("RGB")).unsqueeze(0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        if target_class is None:
            target_class = pred

        score = logits[:, target_class].sum()
        score.backward()

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, probs.detach().cpu().numpy()[0], pred, int(target_class)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def overlay_cam(gray_image, cam):
    gray = np.array(ImageOps.autocontrast(gray_image.resize((IMG_SIZE, IMG_SIZE)).convert("L"))) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    overlay = 0.55 * np.stack([gray, gray, gray], axis=-1) + 0.45 * heat
    return np.clip(overlay, 0, 1)


def read_submission_label(path, file_name):
    df = pd.read_csv(path)
    return int(df.loc[df["file_name"] == file_name, "label"].iloc[0])


def main():
    out_dir = Path("reports/xai")
    out_dir.mkdir(parents=True, exist_ok=True)

    models_info = {
        "EXP-12 stage4": {
            "model": build_convnext("models/exp12_convnext_tiny_best.pt"),
            "submission": "submissions/exp12_convnext_tiny_submit.csv",
        },
        "EXP-18 full": {
            "model": build_convnext("models/exp18_convnext_tiny_full_finetune_best.pt"),
            "submission": "submissions/exp18_convnext_tiny_full_finetune_submit.csv",
        },
    }

    cams = {
        name: GradCAM(info["model"], info["model"].features[-1])
        for name, info in models_info.items()
    }

    summary_rows = []
    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(12, 4 * len(SAMPLES)))
    if len(SAMPLES) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, file_name in enumerate(SAMPLES):
        image = Image.open(Path("test") / file_name).convert("RGB")
        gray = image.convert("L")

        axes[row_idx, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[row_idx, 0].axis("off")

        exp20_label = read_submission_label("submissions/exp20_ridge_stacking_alpha100_submit.csv", file_name)
        exp26_label = read_submission_label("submissions/exp26_ridge_without_densenet_alpha30_submit.csv", file_name)
        axes[row_idx, 0].set_title(f"{file_name}\nEXP20={exp20_label}, EXP26={exp26_label}", fontsize=10)

        x = preprocess(image).to(DEVICE)
        for col_idx, (name, info) in enumerate(models_info.items(), start=1):
            label = read_submission_label(info["submission"], file_name)
            cam, probs, pred, target_class = cams[name](x, target_class=label)
            axes[row_idx, col_idx].imshow(overlay_cam(gray, cam))
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(
                f"{name}\nsub={label}, pred={pred}, p1={probs[1]:.3f}, CAM class={target_class}",
                fontsize=10,
            )
            summary_rows.append(
                {
                    "file_name": file_name,
                    "model": name,
                    "submission_label": label,
                    "pred_label": pred,
                    "prob_0": float(probs[0]),
                    "prob_1": float(probs[1]),
                    "cam_target_class": target_class,
                }
            )

    plt.tight_layout()
    out_path = out_dir / "gradcam_convnext_disagreement_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_dir / "gradcam_convnext_disagreement_summary.csv", index=False)

    for cam in cams.values():
        cam.close()

    print("saved:", out_path)
    print("saved:", out_dir / "gradcam_convnext_disagreement_summary.csv")


if __name__ == "__main__":
    main()
