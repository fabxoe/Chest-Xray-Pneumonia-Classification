from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PRED_DIR = Path("oof/preds")
REPORT_DIR = Path("reports/exp95_suspicious_cases")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_SUBMISSION = Path("oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv")
BEST_SCORES = PRED_DIR / "exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv"

PROTECTED = {
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
    "test_0141.png": 1,
    "test_0049.png": 0,
}

MODELS = {
    "cnx": PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv",
    "resnet": PRED_DIR / "exp48_resnet18_test_probs_mean.csv",
    "eff": PRED_DIR / "exp50_efficientnet_b0_test_probs_mean.csv",
    "cnx123": PRED_DIR / "exp54_convnext_tiny_seed123_test_probs_mean.csv",
    "cnx777": PRED_DIR / "exp72_convnext_tiny_seed777_test_probs_mean.csv",
    "cnx_lion": PRED_DIR / "exp78_convnext_tiny_lion_test_probs_mean.csv",
    "eff123": PRED_DIR / "exp81_efficientnet_b0_seed123_test_probs_mean.csv",
    "cnxv2": PRED_DIR / "exp65_convnextv2_tiny_stage3_test_probs_mean.csv",
    "regnet": PRED_DIR / "exp67_regnet_y_3_2gf_test_probs_mean.csv",
    "effv2": PRED_DIR / "exp70_efficientnet_v2_s_wide_test_probs_mean.csv",
}


def load_base():
    base = pd.read_csv(BEST_SUBMISSION).rename(columns={"label": "exp56_label"})
    scores = pd.read_csv(BEST_SCORES)[["file_name", "decision_score"]].rename(
        columns={"decision_score": "exp56_decision_score"}
    )
    base = base.merge(scores, on="file_name", validate="one_to_one")
    sample = pd.read_csv("sample_submission.csv")[["file_name"]]
    if not base["file_name"].equals(sample["file_name"]):
        raise ValueError("EXP-56 submission order does not match sample_submission.csv")
    return base


def add_model_probs(df):
    out = df.copy()
    used = []
    for name, path in MODELS.items():
        if not path.exists():
            continue
        cur = pd.read_csv(path)[["file_name", "prob_1", "pred_label"]].rename(
            columns={"prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"}
        )
        out = out.merge(cur, on="file_name", validate="one_to_one")
        used.append(name)
    if len(used) < 3:
        raise ValueError(f"Too few model probability files found: {used}")
    return out, used


def add_suspicion_features(df, model_names):
    prob_cols = [f"{name}_prob1" for name in model_names]
    pred_cols = [f"{name}_pred" for name in model_names]
    out = df.copy()
    probs = out[prob_cols]
    preds = out[pred_cols]

    out["prob_mean"] = probs.mean(axis=1)
    out["prob_std"] = probs.std(axis=1)
    out["prob_min"] = probs.min(axis=1)
    out["prob_max"] = probs.max(axis=1)
    out["prob_range"] = out["prob_max"] - out["prob_min"]
    out["vote_1_count"] = preds.sum(axis=1)
    out["vote_0_count"] = len(model_names) - out["vote_1_count"]
    out["vote_minority_count"] = out[["vote_0_count", "vote_1_count"]].min(axis=1)
    out["exp56_score_margin"] = out["exp56_decision_score"].abs()
    out["prob_mean_margin"] = (out["prob_mean"] - 0.5).abs()
    out["model_disagrees_with_exp56"] = preds.ne(out["exp56_label"], axis=0).sum(axis=1)
    out["protected_expected"] = out["file_name"].map(PROTECTED)
    out["is_protected"] = out["protected_expected"].notna()
    out["protected_matches_exp56"] = np.where(
        out["is_protected"], out["exp56_label"].eq(out["protected_expected"]), True
    )

    # High score means "worth reviewing", not "should flip".
    out["suspicious_score"] = (
        out["prob_std"].rank(pct=True) * 0.32
        + out["prob_range"].rank(pct=True) * 0.22
        + out["vote_minority_count"].rank(pct=True) * 0.18
        + (1.0 - out["exp56_score_margin"].rank(pct=True)) * 0.18
        + out["model_disagrees_with_exp56"].rank(pct=True) * 0.10
    )

    conditions = [
        out["is_protected"],
        (out["model_disagrees_with_exp56"] >= 6) & (out["exp56_score_margin"] < 0.25),
        (out["vote_minority_count"] >= 3) | (out["prob_std"] >= out["prob_std"].quantile(0.90)),
    ]
    choices = ["avoid_change_protected", "review_high_priority", "review"]
    out["recommended_action"] = np.select(conditions, choices, default="keep")
    return out.sort_values(["suspicious_score", "prob_std"], ascending=False).reset_index(drop=True)


def image_stats(file_name):
    arr = np.asarray(Image.open(Path("test") / file_name).convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=64, range=(0, 1))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {
        "img_mean": float(arr.mean()),
        "img_std": float(arr.std()),
        "img_p10": float(np.percentile(arr, 10)),
        "img_p90": float(np.percentile(arr, 90)),
        "img_entropy": entropy,
    }


def add_image_stats(df):
    rows = [image_stats(file_name) for file_name in df["file_name"]]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def make_panel(df, model_names, path, n=50):
    chosen = df.head(n)
    cols = 5
    tile_w, tile_h = 250, 250
    header_h = 50
    rows = int(np.ceil(len(chosen) / cols))
    canvas = Image.new("RGB", (cols * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "EXP-95 suspicious test cases: review candidates, not direct flip targets", fill="black")
    key_models = [name for name in ["cnx", "resnet", "eff", "cnx123", "eff123"] if name in model_names]
    for i, row in enumerate(chosen.itertuples(index=False)):
        x = (i % cols) * tile_w
        y = header_h + (i // cols) * tile_h
        image = Image.open(Path("test") / row.file_name).convert("L")
        image.thumbnail((220, 160))
        canvas.paste(image.convert("RGB"), (x + 15, y + 5))
        line1 = f"{row.file_name} exp56={row.exp56_label} s={row.suspicious_score:.3f}"
        line2 = f"std={row.prob_std:.2f} vote1={row.vote_1_count} dis={row.model_disagrees_with_exp56}"
        probs = " ".join([f"{m}:{getattr(row, m + '_prob1'):.2f}" for m in key_models])
        action = str(row.recommended_action).replace("_", " ")
        draw.multiline_text((x + 10, y + 170), f"{line1}\n{line2}\n{probs}\n{action}", fill="black", spacing=2)
    canvas.save(path)


def write_report(df, model_names):
    action_counts = df["recommended_action"].value_counts().to_dict()
    top_cols = [
        "file_name",
        "exp56_label",
        "suspicious_score",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "exp56_decision_score",
        "recommended_action",
    ]
    top = df.head(15)[top_cols].copy()
    for col in ["suspicious_score", "prob_std", "exp56_decision_score"]:
        top[col] = top[col].map(lambda x: f"{x:.4f}")

    header = "| " + " | ".join(top_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(top_cols)) + " |"
    body = [
        "| " + " | ".join(str(getattr(row, col)) for col in top_cols) + " |"
        for row in top.itertuples(index=False)
    ]
    top_markdown = "\n".join([header, sep] + body)
    report = [
        "# EXP-95 Suspicious Test Case Review Pack",
        "",
        "목적: 제출 라벨을 leaderboard probing 방식으로 뒤집기보다, 모델 간 불일치와 불확실성이 큰 test 이미지를 리뷰 대상으로 묶는다.",
        "",
        f"- 기준 제출: `{BEST_SUBMISSION}`",
        f"- 사용 모델 수: `{len(model_names)}`",
        f"- 사용 모델: `{', '.join(model_names)}`",
        f"- action counts: `{action_counts}`",
        "",
        "## 해석 원칙",
        "",
        "- `suspicious_score`가 높다는 것은 바로 라벨을 바꾸라는 뜻이 아니다.",
        "- `review_high_priority`는 모델 간 반대가 크고 EXP-56 decision score가 경계에 가까운 샘플이다.",
        "- `avoid_change_protected`는 이전 public 결과에서 방향성이 확인된 guardrail 샘플이므로 변경하지 않는 쪽이 우선이다.",
        "- 의료 이미지에서는 이미지 패널과 XAI 반응을 보더라도 실제 병변이라고 단정하지 않고, 모델 반응 위치와 예측 불확실성으로만 해석한다.",
        "",
        "## Top 15",
        "",
        top_markdown,
        "",
        "## Outputs",
        "",
        "- `reports/exp95_suspicious_cases/exp95_suspicious_cases.csv`",
        "- `reports/exp95_suspicious_cases/exp95_suspicious_top50.csv`",
        "- `reports/exp95_suspicious_cases/exp95_suspicious_panel_top50.png`",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    base = load_base()
    df, model_names = add_model_probs(base)
    df = add_suspicion_features(df, model_names)
    df = add_image_stats(df)

    full_path = REPORT_DIR / "exp95_suspicious_cases.csv"
    top_path = REPORT_DIR / "exp95_suspicious_top50.csv"
    panel_path = REPORT_DIR / "exp95_suspicious_panel_top50.png"
    df.to_csv(full_path, index=False)
    df.head(50).to_csv(top_path, index=False)
    make_panel(df, model_names, panel_path, n=50)
    write_report(df, model_names)

    print("EXP-95 suspicious review pack")
    print(f"models: {model_names}")
    print(f"rows: {len(df)}")
    print(df["recommended_action"].value_counts().to_string())
    print()
    print(df.head(20)[[
        "file_name",
        "exp56_label",
        "suspicious_score",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "exp56_decision_score",
        "recommended_action",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
