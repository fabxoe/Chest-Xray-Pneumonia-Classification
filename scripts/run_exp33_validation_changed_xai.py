from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lime import lime_image
from PIL import Image, ImageOps
from skimage.segmentation import mark_boundaries
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
    "train_1229.png",  # EXP20 FP -> EXP26 OK
    "train_2325.png",  # EXP20 OK -> EXP26 FN
    "train_4170.png",  # EXP20 OK -> EXP26 FN
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


def build_convnext(model_path="models/exp12_convnext_tiny_best.pt"):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(load_state(model_path))
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


def normalize_batch(images):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensors = []
    for image in images:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        tensors.append(tf(pil))
    return torch.stack(tensors)


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
        logits[:, target_class].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
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


def make_predict_fn(model):
    model = model.to(DEVICE).eval()

    def predict_fn(images):
        probs = []
        with torch.no_grad():
            for start in range(0, len(images), 32):
                batch = normalize_batch(images[start : start + 32]).to(DEVICE)
                prob = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0)

    return predict_fn


def main():
    out_dir = Path("reports/xai")
    out_dir.mkdir(parents=True, exist_ok=True)
    error_df = pd.read_csv(out_dir / "validation_error_summary.csv")

    builders = {
        "ResNet18": build_resnet,
        "DenseNet121": build_densenet,
        "EffNetB0": build_efficientnet,
        "ConvNeXt12": build_convnext,
    }
    cams = {name: GradCAM(*builder()) for name, builder in builders.items()}
    convnext_model, _ = build_convnext()
    lime_predict_fn = make_predict_fn(convnext_model)
    explainer = lime_image.LimeImageExplainer(random_state=42)

    fig, axes = plt.subplots(len(SAMPLES), 7, figsize=(24, 4 * len(SAMPLES)))
    rows = []

    for row_idx, file_name in enumerate(SAMPLES):
        meta = error_df.loc[error_df.file_name == file_name].iloc[0].to_dict()
        image = Image.open(Path("train") / file_name).convert("RGB")
        gray = image.convert("L")
        x = preprocess(image).to(DEVICE)
        arr = np.array(ImageOps.autocontrast(image.resize((IMG_SIZE, IMG_SIZE))))

        axes[row_idx, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"{file_name}\nlabel={int(meta['label'])}\n"
            f"EXP20={int(meta['exp20_pred'])}({meta['exp20_error_type']}) "
            f"EXP26={int(meta['exp26_pred'])}({meta['exp26_error_type']})",
            fontsize=10,
        )

        for col_idx, (name, cam_runner) in enumerate(cams.items(), start=1):
            cam, probs, pred, target = cam_runner(x)
            axes[row_idx, col_idx].imshow(overlay_cam(gray, cam))
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(f"{name}\npred={pred}, p1={probs[1]:.3f}", fontsize=10)
            rows.append(
                {
                    "file_name": file_name,
                    "view": "gradcam",
                    "model": name,
                    "label": int(meta["label"]),
                    "exp20_pred": int(meta["exp20_pred"]),
                    "exp26_pred": int(meta["exp26_pred"]),
                    "pred_label": pred,
                    "prob_1": float(probs[1]),
                    "target_class": target,
                }
            )

        explanation = explainer.explain_instance(
            arr,
            lime_predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=500,
            batch_size=32,
        )
        for offset, target_label in enumerate([0, 1], start=5):
            temp, mask = explanation.get_image_and_mask(
                target_label,
                positive_only=False,
                num_features=8,
                hide_rest=False,
            )
            axes[row_idx, offset].imshow(mark_boundaries(temp / 255.0, mask))
            axes[row_idx, offset].axis("off")
            axes[row_idx, offset].set_title(f"ConvNeXt LIME\nclass {target_label}", fontsize=10)
            top_weights = sorted(explanation.local_exp.get(target_label, []), key=lambda x: abs(x[1]), reverse=True)[:8]
            rows.append(
                {
                    "file_name": file_name,
                    "view": "lime",
                    "model": "ConvNeXt12",
                    "label": int(meta["label"]),
                    "exp20_pred": int(meta["exp20_pred"]),
                    "exp26_pred": int(meta["exp26_pred"]),
                    "pred_label": int(meta["convnext12_pred"]),
                    "prob_1": float(meta["convnext12_prob1"]),
                    "target_class": target_label,
                    "top_superpixel_weights": str(top_weights),
                }
            )

    plt.tight_layout()
    panel_path = out_dir / "validation_changed_xai_panel.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "validation_changed_xai_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    for cam_runner in cams.values():
        cam_runner.close()

    print("saved:", panel_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
