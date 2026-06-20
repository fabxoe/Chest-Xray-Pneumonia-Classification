from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


OUT_DIR = Path("reports/xai")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_probs(path, prefix):
    df = pd.read_csv(path).copy()
    label_col = "label" if "label" in df.columns else "y_true"
    pred_col = "pred" if "pred" in df.columns else "pred_label"
    out = df[["file_name", label_col, pred_col, "prob_1"]].rename(
        columns={label_col: "label", pred_col: f"{prefix}_pred", "prob_1": f"{prefix}_prob1"}
    )
    return out


def main():
    exp20 = pd.read_csv("preds/exp20_ridge_stacking_alpha100_valid_scores.csv")
    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_valid_scores.csv")
    exp20 = exp20.rename(columns={"pred": "exp20_pred", "decision_score": "exp20_score"})
    exp26 = exp26.rename(columns={"pred": "exp26_pred", "decision_score": "exp26_score"})

    base = exp20[["file_name", "label", "exp20_pred", "exp20_score"]].merge(
        exp26[["file_name", "exp26_pred", "exp26_score"]], on="file_name", how="left"
    )

    model_paths = {
        "resnet18": "preds/exp07_resnet18_valid_probs.csv",
        "densenet121": "preds/exp09_densenet121_valid_probs.csv",
        "efficientnet_b0": "preds/exp06_efficientnet_b0_valid_probs.csv",
        "convnext12": "preds/exp12_convnext_tiny_valid_probs.csv",
        "convnext18_full": "preds/exp18_convnext_tiny_full_finetune_valid_probs.csv",
    }
    for name, path in model_paths.items():
        base = base.merge(load_probs(path, name).drop(columns=["label"]), on="file_name", how="left")

    base["exp20_error_type"] = np.where(
        (base.label == 0) & (base.exp20_pred == 1),
        "FP",
        np.where((base.label == 1) & (base.exp20_pred == 0), "FN", "OK"),
    )
    base["exp26_error_type"] = np.where(
        (base.label == 0) & (base.exp26_pred == 1),
        "FP",
        np.where((base.label == 1) & (base.exp26_pred == 0), "FN", "OK"),
    )

    errors = base[(base.exp20_error_type != "OK") | (base.exp26_error_type != "OK")].copy()
    errors["abs_exp20_score"] = errors.exp20_score.abs()
    errors["abs_exp26_score"] = errors.exp26_score.abs()
    errors = errors.sort_values(["exp20_error_type", "abs_exp20_score"], ascending=[True, False])
    errors.to_csv(OUT_DIR / "validation_error_summary.csv", index=False)

    # Select a compact panel: EXP20 FP/FN and EXP26-only changed errors if available.
    selected = []
    for err_type in ["FP", "FN"]:
        subset = errors[errors.exp20_error_type == err_type].sort_values("abs_exp20_score", ascending=False)
        selected.extend(subset.head(4).file_name.tolist())
    for fn in errors[errors.exp20_pred != errors.exp26_pred].file_name.tolist():
        if fn not in selected:
            selected.append(fn)
    selected = selected[:10]

    thumb_w, thumb_h = 240, 240
    row_h = 330
    width = 1500
    panel = Image.new("RGB", (width, row_h * len(selected)), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:
        font_title = font = font_small = None

    for idx, file_name in enumerate(selected):
        row = base[base.file_name == file_name].iloc[0]
        y0 = idx * row_h
        image = Image.open(Path("train") / file_name).convert("L").resize((thumb_w, thumb_h))
        image = ImageOps.autocontrast(image).convert("RGB")
        panel.paste(image, (25, y0 + 55))

        draw.text((25, y0 + 20), file_name, fill="black", font=font_title)
        header = (
            f"label={int(row.label)} | EXP20={int(row.exp20_pred)}({row.exp20_error_type}) "
            f"score={row.exp20_score:.3f} | EXP26={int(row.exp26_pred)}({row.exp26_error_type}) "
            f"score={row.exp26_score:.3f}"
        )
        draw.text((290, y0 + 55), header, fill="black", font=font)

        prob_lines = [
            f"ConvNeXt12 p1={row.convnext12_prob1:.3f}, pred={int(row.convnext12_pred)}",
            f"ConvNeXt18 full p1={row.convnext18_full_prob1:.3f}, pred={int(row.convnext18_full_pred)}",
            f"ResNet18 p1={row.resnet18_prob1:.3f}, pred={int(row.resnet18_pred)}",
            f"DenseNet121 p1={row.densenet121_prob1:.3f}, pred={int(row.densenet121_pred)}",
            f"EfficientNet-B0 p1={row.efficientnet_b0_prob1:.3f}, pred={int(row.efficientnet_b0_pred)}",
        ]
        for j, line in enumerate(prob_lines):
            draw.text((290, y0 + 95 + j * 26), line, fill="black", font=font_small)

        note = (
            "FN: true positive case predicted as negative. Review carefully in medical classification."
            if row.label == 1
            else "FP: true negative case predicted as positive. Candidate for over-response review."
        )
        draw.text((290, y0 + 245), note, fill=(70, 70, 70), font=font_small)

    panel_path = OUT_DIR / "validation_error_grid.png"
    panel.save(panel_path)

    counts = {
        "exp20_fp": int(((base.label == 0) & (base.exp20_pred == 1)).sum()),
        "exp20_fn": int(((base.label == 1) & (base.exp20_pred == 0)).sum()),
        "exp26_fp": int(((base.label == 0) & (base.exp26_pred == 1)).sum()),
        "exp26_fn": int(((base.label == 1) & (base.exp26_pred == 0)).sum()),
        "selected_samples": selected,
    }
    pd.Series(counts, dtype="object").to_json(OUT_DIR / "validation_error_counts.json", force_ascii=False, indent=2)

    print("saved:", OUT_DIR / "validation_error_summary.csv")
    print("saved:", panel_path)
    print(counts)


if __name__ == "__main__":
    main()
