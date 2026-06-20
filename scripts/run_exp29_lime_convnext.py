from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from lime import lime_image
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
    "test_0244.png",
    "test_0267.png",
    "test_0371.png",
    "test_0535.png",
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


def normalize_batch(images: np.ndarray):
    tensors = []
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for image in images:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        tensors.append(tf(pil))
    return torch.stack(tensors)


def make_predict_fn(model):
    def predict_fn(images):
        probs = []
        batch_size = 32
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = normalize_batch(images[start : start + batch_size]).to(DEVICE)
                logits = model(batch)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0)

    return predict_fn


def read_submission_label(path, file_name):
    df = pd.read_csv(path)
    return int(df.loc[df["file_name"] == file_name, "label"].iloc[0])


def image_array(file_name):
    img = Image.open(Path("test") / file_name).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(ImageOps.autocontrast(img))


def main():
    out_dir = Path("reports/xai")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_convnext("models/exp12_convnext_tiny_best.pt")
    predict_fn = make_predict_fn(model)
    explainer = lime_image.LimeImageExplainer(random_state=42)

    rows = []
    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(12, 4 * len(SAMPLES)))
    if len(SAMPLES) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, file_name in enumerate(SAMPLES):
        arr = image_array(file_name)
        base_prob = predict_fn(np.array([arr]))[0]
        exp12_label = read_submission_label("submissions/exp12_convnext_tiny_submit.csv", file_name)
        exp20_label = read_submission_label("submissions/exp20_ridge_stacking_alpha100_submit.csv", file_name)
        exp26_label = read_submission_label("submissions/exp26_ridge_without_densenet_alpha30_submit.csv", file_name)

        explanation = explainer.explain_instance(
            arr,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=600,
            batch_size=32,
        )

        axes[row_idx, 0].imshow(arr)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"{file_name}\nEXP12={exp12_label}, EXP20={exp20_label}, EXP26={exp26_label}\n"
            f"ConvNeXt p1={base_prob[1]:.3f}",
            fontsize=10,
        )

        for col_idx, target_label in enumerate([0, 1], start=1):
            temp, mask = explanation.get_image_and_mask(
                target_label,
                positive_only=False,
                num_features=8,
                hide_rest=False,
            )
            axes[row_idx, col_idx].imshow(mark_boundaries(temp / 255.0, mask))
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(f"LIME target class {target_label}", fontsize=10)

            weights = explanation.local_exp.get(target_label, [])
            top_weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:8]
            rows.append(
                {
                    "file_name": file_name,
                    "target_class": target_label,
                    "convnext_prob_0": float(base_prob[0]),
                    "convnext_prob_1": float(base_prob[1]),
                    "exp12_label": exp12_label,
                    "exp20_label": exp20_label,
                    "exp26_label": exp26_label,
                    "top_superpixel_weights": str(top_weights),
                }
            )

    plt.tight_layout()
    out_path = out_dir / "lime_convnext_disagreement_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "lime_convnext_disagreement_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("saved:", out_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
