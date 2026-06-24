import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PRED = Path("oof/preds")
REPORT = Path("reports/exp93_disagreement")
REPORT.mkdir(parents=True, exist_ok=True)


def image_features(file_name):
    image = Image.open(Path("train") / file_name).convert("L")
    original_w, original_h = image.size
    arr = np.asarray(image.resize((256, 256)), dtype=np.float32) / 255.0
    h, w = arr.shape
    cy0, cy1, cx0, cx1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    center = arr[cy0:cy1, cx0:cx1]
    border_mask = np.ones_like(arr, dtype=bool)
    border_mask[cy0:cy1, cx0:cx1] = False
    border = arr[border_mask]
    hist, _ = np.histogram(arr, bins=64, range=(0, 1), density=False)
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    left, right = arr[:, : w // 2], arr[:, w // 2 :]
    top, bottom = arr[: h // 2], arr[h // 2 :]
    p10, p90 = np.percentile(arr, [10, 90])
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p10": float(p10), "p90": float(p90), "dynamic_range": float(p90 - p10),
        "entropy": entropy, "gradient_x": float(gx), "gradient_y": float(gy),
        "center_mean": float(center.mean()), "border_mean": float(border.mean()),
        "center_border_delta": float(center.mean() - border.mean()),
        "left_right_diff": float(abs(left.mean() - right.mean())),
        "top_bottom_diff": float(abs(top.mean() - bottom.mean())),
        "aspect_ratio": float(original_w / original_h),
    }


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(1, len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def make_montage(df, path, title, n=12):
    chosen = df.sort_values("prob_gap", ascending=False).head(n)
    tile_w, tile_h = 220, 230
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    canvas = Image.new("RGB", (cols * tile_w, 45 + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    for i, row in enumerate(chosen.itertuples()):
        x, y = (i % cols) * tile_w, 45 + (i // cols) * tile_h
        image = Image.open(Path("train") / row.file_name).convert("L")
        image.thumbnail((200, 175))
        canvas.paste(image.convert("RGB"), (x + 10, y))
        text = f"{row.file_name} y={row.label}\nCNX={row.cnx_prob1:.2f} RES={row.res_prob1:.2f}"
        draw.multiline_text((x + 10, y + 180), text, fill="black", spacing=2)
    canvas.save(path)


def main():
    cnx = pd.read_csv(PRED / "exp46_convnext_tiny_oof_probs.csv").rename(
        columns={"prob_1": "cnx_prob1", "pred_label": "cnx_pred"}
    )
    res = pd.read_csv(PRED / "exp48_resnet18_oof_probs.csv").rename(
        columns={"prob_1": "res_prob1", "pred_label": "res_pred"}
    )
    eff = pd.read_csv(PRED / "exp50_efficientnet_b0_oof_probs.csv").rename(
        columns={"prob_1": "eff_prob1", "pred_label": "eff_pred"}
    )
    df = cnx[["file_name", "label", "fold", "cnx_prob1", "cnx_pred"]].merge(
        res[["file_name", "res_prob1", "res_pred"]], on="file_name", validate="one_to_one"
    ).merge(eff[["file_name", "eff_prob1", "eff_pred"]], on="file_name", validate="one_to_one")

    df["group"] = np.select(
        [df.cnx_pred.eq(0) & df.res_pred.eq(1), df.cnx_pred.eq(1) & df.res_pred.eq(0),
         df.cnx_pred.eq(0) & df.res_pred.eq(0)],
        ["cnx0_res1", "cnx1_res0", "agree0"], default="agree1",
    )
    df["prob_gap"] = (df.res_prob1 - df.cnx_prob1).abs()
    feats = pd.DataFrame([image_features(x) for x in df.file_name])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df.to_csv(PRED / "exp93_disagreement_image_features.csv", index=False)

    feature_cols = list(feats.columns)
    summary = df.groupby("group").agg(
        n=("file_name", "size"), positive_rate=("label", "mean"),
        cnx_accuracy=("cnx_pred", lambda x: float((x.to_numpy() == df.loc[x.index, "label"].to_numpy()).mean())),
        res_accuracy=("res_pred", lambda x: float((x.to_numpy() == df.loc[x.index, "label"].to_numpy()).mean())),
        cnx_prob_mean=("cnx_prob1", "mean"), res_prob_mean=("res_prob1", "mean"),
        eff_prob_mean=("eff_prob1", "mean"),
    ).reset_index()
    for feature in feature_cols:
        means = df.groupby("group")[feature].mean()
        summary[f"{feature}_mean"] = summary.group.map(means)
    summary.to_csv(PRED / "exp93_disagreement_group_summary.csv", index=False)

    dis = df[df.group.isin(["cnx0_res1", "cnx1_res0"])].copy()
    dis["direction"] = dis.group.eq("cnx0_res1").astype(int)
    effects = []
    for feature in feature_cols:
        a = dis.loc[dis.direction.eq(1), feature]
        b = dis.loc[dis.direction.eq(0), feature]
        effects.append({"feature": feature, "cnx0_res1_mean": a.mean(), "cnx1_res0_mean": b.mean(), "cohens_d": cohens_d(a, b)})
    effects = pd.DataFrame(effects).sort_values("cohens_d", key=lambda x: x.abs(), ascending=False)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(model, dis[feature_cols], dis.direction, cv=cv)
    model.fit(dis[feature_cols], dis.direction)
    coef = model[-1].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_cols, "logistic_coef": coef, "abs_coef": np.abs(coef)})
    effects = effects.merge(coef_df, on="feature").sort_values("abs_coef", ascending=False)
    effects.to_csv(PRED / "exp93_disagreement_feature_effects.csv", index=False)

    make_montage(dis[dis.group.eq("cnx0_res1")], REPORT / "cnx0_res1_representatives.png", "ConvNeXt=0, ResNet=1")
    make_montage(dis[dis.group.eq("cnx1_res0")], REPORT / "cnx1_res0_representatives.png", "ConvNeXt=1, ResNet=0")

    result = {
        "total": len(df), "disagreements": len(dis),
        "group_counts": df.group.value_counts().to_dict(),
        "direction_classifier_accuracy": accuracy_score(dis.direction, pred),
        "direction_classifier_balanced_accuracy": balanced_accuracy_score(dis.direction, pred),
        "top_effects": effects.head(8).to_dict(orient="records"),
    }
    (PRED / "exp93_disagreement_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
