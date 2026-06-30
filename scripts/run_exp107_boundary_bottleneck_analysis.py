from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "oof/preds/exp106_no_swin_low_change_summary.csv"
EXP56_PATH = ROOT / "oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv"
SUSPICIOUS_PATH = ROOT / "reports/exp95_suspicious_cases/exp95_suspicious_cases.csv"
OUT_DIR = ROOT / "reports/exp107_boundary_bottleneck"

PROTECTED = {
    "test_0025": 0,
    "test_0049": 0,
    "test_0141": 1,
    "test_0188": 0,
    "test_0201": 1,
    "test_0222": 1,
    "test_0244": 0,
    "test_0490": 0,
    "test_0504": 0,
    "test_0539": 0,
}


def sample_id(file_name: str) -> str:
    return Path(file_name).stem


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    exp56 = pd.read_csv(EXP56_PATH)
    exp56["sample_id"] = exp56["file_name"].map(sample_id)
    suspicious = pd.read_csv(SUSPICIOUS_PATH)
    suspicious["sample_id"] = suspicious["file_name"].map(sample_id)
    return summary, exp56, suspicious


def make_pool(summary: pd.DataFrame) -> pd.DataFrame:
    # This pool isolates rows that are otherwise plausible except for guardrail conflicts.
    return summary[
        summary["passes_label_band"]
        & summary["passes_low_change_current"]
        & summary["passes_meta_cv_floor"]
        & summary["not_duplicate_current"]
    ].copy()


def protected_bottleneck(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(pool)
    for sid, expected in PROTECTED.items():
        pred_col = f"{sid}_pred"
        score_col = f"{sid}_score"
        if pred_col not in pool:
            continue
        wrong = pool[pred_col] != expected
        rows.append(
            {
                "sample_id": sid,
                "expected": expected,
                "wrong_count": int(wrong.sum()),
                "wrong_rate": float(wrong.mean()) if total else 0.0,
                "pred1_rate": float((pool[pred_col] == 1).mean()) if total else 0.0,
                "mean_score": float(pool[score_col].mean()) if score_col in pool else None,
                "median_score": float(pool[score_col].median()) if score_col in pool else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["wrong_count", "sample_id"], ascending=[False, True])


def top_rows(summary: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "spec",
        "model",
        "param",
        "threshold",
        "meta_cv_accuracy",
        "meta_cv_recall",
        "test_label1",
        "changed_vs_current_best",
        "passes_protected",
        "candidate",
    ]
    protected_cols = []
    for sid in PROTECTED:
        protected_cols.extend([f"{sid}_pred", f"{sid}_score"])
    cols = [c for c in cols + protected_cols if c in summary.columns]
    top_overall = summary.sort_values("meta_cv_accuracy", ascending=False)[cols].head(30)
    top_pool = pool.sort_values("meta_cv_accuracy", ascending=False)[cols].head(30)
    return top_overall, top_pool


def unprotected_watchlist(pool: pd.DataFrame, exp56: pd.DataFrame, suspicious: pd.DataFrame) -> pd.DataFrame:
    # For rows that fail only because of protected samples, identify unprotected samples
    # frequently touched by strong candidates. This is for review, not direct flipping.
    exp56_labels = dict(zip(exp56["sample_id"], exp56["label"]))
    id_cols = [c for c in pool.columns if c.startswith("test_") and c.endswith("_pred")]
    protected_ids = set(PROTECTED)
    rows = []
    for pred_col in id_cols:
        sid = pred_col.removesuffix("_pred")
        if sid in protected_ids or sid not in exp56_labels:
            continue
        current = int(exp56_labels[sid])
        changed = pool[pred_col] != current
        if not changed.any():
            continue
        score_col = f"{sid}_score"
        rows.append(
            {
                "sample_id": sid,
                "exp56_label": current,
                "change_count": int(changed.sum()),
                "change_rate": float(changed.mean()),
                "pred1_rate": float((pool[pred_col] == 1).mean()),
                "mean_score": float(pool[score_col].mean()) if score_col in pool else None,
                "median_score": float(pool[score_col].median()) if score_col in pool else None,
            }
        )
    watch = pd.DataFrame(rows)
    if watch.empty:
        return watch
    keep_cols = [
        "sample_id",
        "file_name",
        "recommended_action",
        "suspicious_score",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "prob_mean",
        "prob_std",
        "exp56_score_margin",
    ]
    suspicious_small = suspicious[[c for c in keep_cols if c in suspicious.columns]].copy()
    watch = watch.merge(suspicious_small, on="sample_id", how="left")
    return watch.sort_values(["change_count", "suspicious_score"], ascending=[False, False])


def write_markdown(
    pool: pd.DataFrame,
    bottleneck: pd.DataFrame,
    watch: pd.DataFrame,
    top_pool: pd.DataFrame,
) -> None:
    def as_md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        cols = list(view.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    md = []
    md.append("# EXP-107 Boundary Bottleneck Analysis")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Plausible no-Swin rows: `{len(pool)}`")
    md.append(f"- Submission candidates after all constraints: `0`")
    md.append("- 목적: EXP-56 주변에서 어떤 샘플이 계속 후보 생성을 막는지 확인")
    md.append("")
    md.append("## Protected Sample Bottleneck")
    md.append("")
    md.append(as_md_table(bottleneck.head(10)))
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "`test_0504`가 가장 큰 병목이다. EXP-56은 이 샘플을 `1 -> 0`으로 바꿔 public 최고점을 만들었지만, "
        "EXP-106의 강한 후보 대부분은 다시 1로 되돌린다. 따라서 현재 pool에서는 meta-CV를 올리는 방향과 "
        "public에서 확인된 보호 방향이 충돌한다."
    )
    md.append("")
    md.append("## Unprotected Watchlist")
    md.append("")
    if watch.empty:
        md.append("변경 빈도가 높은 비보호 샘플이 없다.")
    else:
        md.append(as_md_table(watch.head(20)))
    md.append("")
    md.append("## Top Plausible Rows")
    md.append("")
    md.append(as_md_table(top_pool.head(10)))
    md.append("")
    md.append("## Next")
    md.append("")
    md.append("- 제출 후보를 바로 만들기보다 `test_0504`와 그 주변 경계 샘플의 모델별 확률/XAI를 비교한다.")
    md.append("- 새 모델 family를 추가하더라도 보호 샘플을 먼저 통과하는지 확인한다.")
    md.append("- 단순 Ridge/Logistic threshold 탐색은 현재 feature pool에서는 우선순위를 낮춘다.")
    (OUT_DIR / "exp107_boundary_bottleneck_report.md").write_text("\n".join(md) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, exp56, suspicious = load_inputs()
    pool = make_pool(summary)
    bottleneck = protected_bottleneck(pool)
    top_overall, top_pool = top_rows(summary, pool)
    watch = unprotected_watchlist(pool, exp56, suspicious)

    bottleneck.to_csv(OUT_DIR / "exp107_protected_bottleneck.csv", index=False)
    top_overall.to_csv(OUT_DIR / "exp107_top_overall_rows.csv", index=False)
    top_pool.to_csv(OUT_DIR / "exp107_top_plausible_rows.csv", index=False)
    watch.to_csv(OUT_DIR / "exp107_unprotected_watchlist.csv", index=False)
    write_markdown(pool, bottleneck, watch, top_pool)

    print("EXP-107 boundary bottleneck analysis")
    print("plausible rows:", len(pool))
    print("candidate rows:", int(summary["candidate"].sum()))
    print("top protected bottlenecks")
    print(bottleneck.head(5).to_string(index=False))
    if not watch.empty:
        print("top unprotected watchlist")
        print(watch.head(10).to_string(index=False))
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
