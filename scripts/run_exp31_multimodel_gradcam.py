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
]


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def build_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(load_state("models/exp07_resnet18_layer4_fc_best.pt"))
    return model, model.layer4[-1]


def build_densenet():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(load_state("models/exp09_densenet121_best.pt"))
    return model, model.features.denseblock4


def build_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(load_state("models/exp06_efficientnet_b0_best.pt"))
    return model, model.features[-1]


def build_convnext():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(load_state("models/exp12_convnext_tiny_best.pt"))
    return model, model.features[-1]


def preprocess(image):
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
        self.model = model.to(DEVICE).eval()
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

        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
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


def read_label(path, file_name):
    df = pd.read_csv(path)
    return int(df.loc[df["file_name"] == file_name, "label"].iloc[0])


def main():
    out_dir = Path("reports/xai")
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "ResNet18": build_resnet,
        "DenseNet121": build_densenet,
        "EffNetB0": build_efficientnet,
        "ConvNeXt12": build_convnext,
    }
    cams = {}
    for name, builder in builders.items():
        model, layer = builder()
        cams[name] = GradCAM(model, layer)

    fig, axes = plt.subplots(len(SAMPLES), len(cams) + 1, figsize=(4 * (len(cams) + 1), 4 * len(SAMPLES)))
    if len(SAMPLES) == 1:
        axes = axes.reshape(1, -1)

    rows = []
    for row_idx, file_name in enumerate(SAMPLES):
        image = Image.open(Path("test") / file_name).convert("RGB")
        gray = image.convert("L")
        x = preprocess(image).to(DEVICE)

        exp20 = read_label("submissions/exp20_ridge_stacking_alpha100_submit.csv", file_name)
        exp26 = read_label("submissions/exp26_ridge_without_densenet_alpha30_submit.csv", file_name)

        axes[row_idx, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(f"{file_name}\nEXP20={exp20}, EXP26={exp26}", fontsize=10)

        for col_idx, (name, cam_runner) in enumerate(cams.items(), start=1):
            cam, probs, pred, target = cam_runner(x, target_class=None)
            axes[row_idx, col_idx].imshow(overlay_cam(gray, cam))
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(f"{name}\npred={pred}, p1={probs[1]:.3f}", fontsize=10)
            rows.append(
                {
                    "file_name": file_name,
                    "model": name,
                    "pred_label": pred,
                    "prob_0": float(probs[0]),
                    "prob_1": float(probs[1]),
                    "cam_target_class": target,
                    "exp20_label": exp20,
                    "exp26_label": exp26,
                }
            )

    plt.tight_layout()
    out_path = out_dir / "gradcam_multimodel_disagreement_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "gradcam_multimodel_disagreement_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    for cam_runner in cams.values():
        cam_runner.close()

    print("saved:", out_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
