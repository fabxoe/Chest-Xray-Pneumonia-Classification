from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/exp125_aug_xai_compare"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CURRENT_BEST = ROOT / "oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv"
EXP45_MODEL = ROOT / "oof/models/exp45_convnext_tiny_fold0.pt"
EXP124_MODEL = ROOT / "oof/models/exp124_convnext_tiny_384_noflip_fold0.pt"
EXP45_SUMMARY = ROOT / "oof/preds/exp45_convnext_tiny_fold0_summary.csv"
EXP124_SUMMARY = ROOT / "oof/preds/exp124_convnext_tiny_384_noflip_fold0_summary.csv"
EXP45_OOF = ROOT / "oof/preds/exp45_convnext_tiny_fold0_oof_probs.csv"
EXP124_OOF = ROOT / "oof/preds/exp124_convnext_tiny_384_noflip_fold0_oof_probs.csv"
EXP45_TEST = ROOT / "oof/preds/exp45_convnext_tiny_fold0_test_probs.csv"
EXP124_TEST = ROOT / "oof/preds/exp124_convnext_tiny_384_noflip_fold0_test_probs.csv"
EXP46_TEST_MEAN = ROOT / "oof/preds/exp46_convnext_tiny_test_probs_mean.csv"
EXP124_SUBMIT = ROOT / "oof/submissions/exp124_convnext_tiny_384_noflip_fold0_submit.csv"

SUCCESS_DROP = ["test_0455.png", "test_0057.png", "test_0385.png", "test_0303.png"]
FAILED_KEEP = ["test_0289.png", "test_0172.png", "test_0013.png", "test_0390.png"]
OLD_GUARDRAILS = [
    "test_0188.png",
    "test_0568.png",
    "test_0049.png",
    "test_0141.png",
    "test_0025.png",
    "test_0539.png",
    "test_0244.png",
    "test_0490.png",
    "test_0201.png",
    "test_0222.png",
    "test_0504.png",
]
TARGET_LABELS = {
    "test_0455.png": 0,
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0303.png": 0,
    "test_0289.png": 1,
    "test_0172.png": 1,
    "test_0013.png": 1,
    "test_0390.png": 1,
    "test_0188.png": 0,
    "test_0568.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 0,
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
}


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, img_size: int):
        self.model = model
        self.img_size = img_size
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
        cam = torch.nn.functional.interpolate(cam, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, probs.detach().cpu().numpy()

    def close(self) -> None:
        self.h1.remove()
        self.h2.remove()


def check_required(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required EXP-124/EXP-45 artifacts are missing:\n" + "\n".join(missing))


def build_model(model_path: Path) -> tuple[nn.Module, nn.Module]:
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(DEVICE).eval()
    return model, model.features[-1]


def tf(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def metric_row(name: str, oof_path: Path) -> dict:
    df = pd.read_csv(oof_path)
    y = df["label"].to_numpy()
    pred = df["pred_label"].to_numpy()
    return {
        "source": name,
        "rows": int(len(df)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def compare_submissions() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST)
    exp124 = pd.read_csv(EXP124_SUBMIT)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if current["file_name"].tolist() != sample["file_name"].tolist():
        raise ValueError("Current-best submission order does not match sample_submission.csv")
    if exp124["file_name"].tolist() != sample["file_name"].tolist():
        raise ValueError("EXP-124 submission order does not match sample_submission.csv")

    changed = current.rename(columns={"label": "current_best_label"}).merge(
        exp124.rename(columns={"label": "exp124_label"}), on="file_name", how="inner"
    )
    changed = changed[changed["current_best_label"] != changed["exp124_label"]].copy()

    exp124_test = pd.read_csv(EXP124_TEST)[["file_name", "prob_0", "prob_1", "pred_label"]].rename(
        columns={"prob_0": "exp124_prob0", "prob_1": "exp124_prob1", "pred_label": "exp124_pred"}
    )
    exp45_test = pd.read_csv(EXP45_TEST)[["file_name", "prob_1", "pred_label"]].rename(
        columns={"prob_1": "exp45_prob1", "pred_label": "exp45_pred"}
    )
    exp46_test = pd.read_csv(EXP46_TEST_MEAN)[["file_name", "prob_1", "pred_label"]].rename(
        columns={"prob_1": "exp46_mean_prob1", "pred_label": "exp46_mean_pred"}
    )
    changed = changed.merge(exp124_test, on="file_name", how="left")
    changed = changed.merge(exp45_test, on="file_name", how="left")
    changed = changed.merge(exp46_test, on="file_name", how="left")
    return changed.sort_values(["current_best_label", "exp124_prob1", "file_name"]).reset_index(drop=True)


def guardrail_table() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST).set_index("file_name")["label"]
    exp124 = pd.read_csv(EXP124_SUBMIT).set_index("file_name")["label"]
    exp45 = pd.read_csv(EXP45_TEST).set_index("file_name")
    exp46 = pd.read_csv(EXP46_TEST_MEAN).set_index("file_name")
    exp124_prob = pd.read_csv(EXP124_TEST).set_index("file_name")
    rows = []
    for file_name, target in TARGET_LABELS.items():
        rows.append(
            {
                "file_name": file_name,
                "target_guardrail_label": target,
                "current_best_label": int(current.loc[file_name]),
                "exp124_label": int(exp124.loc[file_name]),
                "exp124_violates_guardrail": int(exp124.loc[file_name]) != target,
                "exp124_prob1": float(exp124_prob.loc[file_name, "prob_1"]),
                "exp45_prob1": float(exp45.loc[file_name, "prob_1"]),
                "exp46_mean_prob1": float(exp46.loc[file_name, "prob_1"]),
                "group": "success_drop" if file_name in SUCCESS_DROP else "failed_keep" if file_name in FAILED_KEEP else "old_guardrail",
            }
        )
    return pd.DataFrame(rows)


def cam_stats(cam: np.ndarray) -> dict[str, float]:
    h, w = cam.shape
    total = float(cam.sum() + 1e-8)
    yy, xx = np.mgrid[0:h, 0:w]
    return {
        "cam_center_fraction": float(cam[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].sum() / total),
        "cam_top_fraction": float(cam[: h // 2].sum() / total),
        "cam_left_fraction": float(cam[:, : w // 2].sum() / total),
        "cam_centroid_x": float((xx * cam).sum() / total / w),
        "cam_centroid_y": float((yy * cam).sum() / total / h),
    }


def overlay(gray: Image.Image, cam: np.ndarray, img_size: int) -> np.ndarray:
    base = np.asarray(ImageOps.autocontrast(gray.resize((img_size, img_size))), dtype=np.float32) / 255
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * np.stack([base] * 3, -1) + 0.45 * heat, 0, 1)


def make_gradcam_panel() -> pd.DataFrame:
    selected = SUCCESS_DROP + FAILED_KEEP
    exp45_model, exp45_layer = build_model(EXP45_MODEL)
    exp124_model, exp124_layer = build_model(EXP124_MODEL)
    exp45_cam = GradCAM(exp45_model, exp45_layer, 224)
    exp124_cam = GradCAM(exp124_model, exp124_layer, 384)
    tf45 = tf(224)
    tf124 = tf(384)

    exp45_probs = pd.read_csv(EXP45_TEST).set_index("file_name")
    exp124_probs = pd.read_csv(EXP124_TEST).set_index("file_name")
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.2 * len(selected)))
    rows = []
    for i, file_name in enumerate(selected):
        image = Image.open(ROOT / "test" / file_name).convert("RGB")
        gray = image.convert("L")

        cam45, probs45 = exp45_cam.run(tf45(image).unsqueeze(0).to(DEVICE), target=1)
        cam124, probs124 = exp124_cam.run(tf124(image).unsqueeze(0).to(DEVICE), target=1)

        axes[i, 0].imshow(ImageOps.autocontrast(gray), cmap="gray")
        axes[i, 0].set_title(f"{file_name}\n{('success_drop' if file_name in SUCCESS_DROP else 'failed_keep')}")
        axes[i, 1].imshow(overlay(gray, cam45, 224))
        axes[i, 1].set_title(f"EXP-45 224 flip/rot\np1={probs45[1]:.3f}")
        axes[i, 2].imshow(overlay(gray, cam124, 384))
        axes[i, 2].set_title(f"EXP-124 384 no-flip\np1={probs124[1]:.3f}")
        for ax in axes[i]:
            ax.axis("off")

        item = {
            "file_name": file_name,
            "group": "success_drop" if file_name in SUCCESS_DROP else "failed_keep",
            "exp45_prob1_file": float(exp45_probs.loc[file_name, "prob_1"]),
            "exp124_prob1_file": float(exp124_probs.loc[file_name, "prob_1"]),
            "exp45_gradcam_prob1": float(probs45[1]),
            "exp124_gradcam_prob1": float(probs124[1]),
        }
        item.update({f"exp45_{k}": v for k, v in cam_stats(cam45).items()})
        item.update({f"exp124_{k}": v for k, v in cam_stats(cam124).items()})
        rows.append(item)

    exp45_cam.close()
    exp124_cam.close()
    plt.tight_layout()
    panel_path = OUT / "exp125_gradcam_exp45_vs_exp124_panel.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def write_report(metric_df: pd.DataFrame, guard_df: pd.DataFrame, changed_df: pd.DataFrame, cam_df: pd.DataFrame) -> None:
    exp124_metric = metric_df.loc[metric_df["source"] == "exp124_convnext_tiny_384_noflip_fold0"].iloc[0]
    exp45_metric = metric_df.loc[metric_df["source"] == "exp45_convnext_tiny_fold0"].iloc[0]
    guard_violations = int(guard_df["exp124_violates_guardrail"].sum())
    current_changes = int(len(changed_df))
    success_cam = cam_df[cam_df["group"] == "success_drop"]
    failed_cam = cam_df[cam_df["group"] == "failed_keep"]
    gate_decision = (
        "EXP-126 5-fold 확장은 보류한다. fold0 accuracy는 개선됐지만 current-best 대비 변경 수가 많고, "
        "이미 public으로 방향이 확인된 guardrail 위반이 8개 발생했으며, Grad-CAM 분리도도 충분히 명확하지 않다."
    )
    report = f"""# EXP-125 Augmentation/XAI 비교

## 요약

- EXP-124는 `384 resize + no flip/no rotation + weak brightness/contrast` fold0 smoke 실험이다.
- EXP-45 fold0 accuracy: `{exp45_metric['accuracy']:.5f}`
- EXP-124 fold0 accuracy: `{exp124_metric['accuracy']:.5f}`
- 현재 최고 EXP-117 대비 EXP-124 단독 제출 변경 수: `{current_changes}`
- guardrail 위반 수: `{guard_violations}`

## Grad-CAM 요약

- success drop 샘플 EXP-45 평균 center fraction: `{success_cam['exp45_cam_center_fraction'].mean():.4f}`
- success drop 샘플 EXP-124 평균 center fraction: `{success_cam['exp124_cam_center_fraction'].mean():.4f}`
- failed keep 샘플 EXP-45 평균 center fraction: `{failed_cam['exp45_cam_center_fraction'].mean():.4f}`
- failed keep 샘플 EXP-124 평균 center fraction: `{failed_cam['exp124_cam_center_fraction'].mean():.4f}`

## 판단

- 이 리포트는 EXP-124를 바로 제출하기 위한 것이 아니라, 384/no-flip 증강이 기존 ConvNeXt와 다른 근거를 제공하는지 확인하기 위한 자료다.
- direct submission은 변경 수와 guardrail 위반을 기준으로 보류/제외한다.
- {gate_decision}
"""
    (OUT / "exp125_aug_xai_compare.md").write_text(report, encoding="utf-8")


def main() -> None:
    check_required(
        [
            CURRENT_BEST,
            EXP45_MODEL,
            EXP124_MODEL,
            EXP45_SUMMARY,
            EXP124_SUMMARY,
            EXP45_OOF,
            EXP124_OOF,
            EXP45_TEST,
            EXP124_TEST,
            EXP46_TEST_MEAN,
            EXP124_SUBMIT,
        ]
    )

    metric_df = pd.DataFrame(
        [
            metric_row("exp45_convnext_tiny_fold0", EXP45_OOF),
            metric_row("exp124_convnext_tiny_384_noflip_fold0", EXP124_OOF),
        ]
    )
    metric_df.to_csv(OUT / "exp125_fold0_metric_compare.csv", index=False)

    changed_df = compare_submissions()
    changed_df.to_csv(OUT / "exp125_changed_vs_exp117_current_best.csv", index=False)

    guard_df = guardrail_table()
    guard_df.to_csv(OUT / "exp125_guardrail_compare.csv", index=False)

    cam_df = make_gradcam_panel()
    cam_df.to_csv(OUT / "exp125_gradcam_summary.csv", index=False)

    group_summary = (
        cam_df.groupby("group")
        .agg(
            samples=("file_name", "count"),
            exp45_prob1=("exp45_gradcam_prob1", "mean"),
            exp124_prob1=("exp124_gradcam_prob1", "mean"),
            exp45_center=("exp45_cam_center_fraction", "mean"),
            exp124_center=("exp124_cam_center_fraction", "mean"),
            exp45_top=("exp45_cam_top_fraction", "mean"),
            exp124_top=("exp124_cam_top_fraction", "mean"),
            exp45_left=("exp45_cam_left_fraction", "mean"),
            exp124_left=("exp124_cam_left_fraction", "mean"),
        )
        .reset_index()
    )
    group_summary.to_csv(OUT / "exp125_gradcam_group_summary.csv", index=False)

    write_report(metric_df, guard_df, changed_df, cam_df)

    result = {
        "device": str(DEVICE),
        "metric_compare": str(OUT / "exp125_fold0_metric_compare.csv"),
        "changed_vs_current_best": str(OUT / "exp125_changed_vs_exp117_current_best.csv"),
        "guardrail_compare": str(OUT / "exp125_guardrail_compare.csv"),
        "gradcam_summary": str(OUT / "exp125_gradcam_summary.csv"),
        "gradcam_group_summary": str(OUT / "exp125_gradcam_group_summary.csv"),
        "gradcam_panel": str(OUT / "exp125_gradcam_exp45_vs_exp124_panel.png"),
        "report": str(OUT / "exp125_aug_xai_compare.md"),
        "exp124_guardrail_violations": int(guard_df["exp124_violates_guardrail"].sum()),
        "exp124_changed_vs_current_best": int(len(changed_df)),
        "exp126_gate_decision": "hold",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
