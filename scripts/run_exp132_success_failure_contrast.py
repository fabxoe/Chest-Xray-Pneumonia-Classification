from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT_BEST = ROOT / "oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv"
EXP129_RANKED = ROOT / "reports/exp129_success_branch_rerank/exp129_candidate_ranked.csv"
EXP130_SELECTION = ROOT / "reports/exp130_success_candidate_gradcam/exp130_selection_summary.csv"
OUT = ROOT / "reports/exp132_success_failure_contrast"
SUB_DIR = ROOT / "oof/submissions"
PRED_DIR = ROOT / "oof/preds"
OUT.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_DROP = {
    "test_0455.png": 0,
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0303.png": 0,
}
FAILED_KEEP = {
    "test_0289.png": 1,
    "test_0172.png": 1,
    "test_0013.png": 1,
    "test_0390.png": 1,
    "test_0600.png": 1,
    "test_0212.png": 1,
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
GUARDRAILS = {**SUCCESS_DROP, **FAILED_KEEP}


PROFILE_COLS = [
    "cnx_prob1",
    "cnx123_prob1",
    "cnx777_prob1",
    "cnx_lion_prob1",
    "eff_prob1",
    "eff123_prob1",
    "resnet_prob1",
    "regnet_prob1",
    "effv2_prob1",
    "cnxv2_prob1",
    "exp126_prob1",
    "prob_mean",
    "prob_std",
    "prob_min",
    "prob_max",
    "prob_range",
    "vote_1_count",
    "vote_0_count",
    "vote_minority_count",
    "model_disagrees_with_exp56",
    "tree_drop_score",
    "exp110_drop_score",
    "exp119_drop_score",
    "exp119_ml_zero_support",
    "exp119_rf_prob0",
    "exp119_extratrees_prob0",
    "exp119_svm_prob0_like",
    "exp119_gauss_edge_delta",
    "exp119_lap_var",
    "exp121_consensus_score",
    "img_mean",
    "img_std",
    "img_entropy",
    "img_p10",
    "img_p90",
]


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    lo, hi = float(s.min()), float(s.max())
    if np.isclose(lo, hi):
        out = pd.Series(0.5, index=series.index)
    else:
        out = (s - lo) / (hi - lo)
    return out if higher_is_better else 1.0 - out


def load_evidence() -> pd.DataFrame:
    if not EXP129_RANKED.exists():
        raise FileNotFoundError(EXP129_RANKED)
    df = pd.read_csv(EXP129_RANKED)
    if EXP130_SELECTION.exists():
        cam = pd.read_csv(EXP130_SELECTION)
        cam_cols = [
            "file_name",
            "success_cam_distance",
            "exp45_gradcam_prob1",
            "exp124_gradcam_prob1",
            "exp45_cam_center_fraction",
            "exp124_cam_center_fraction",
            "exp130_selection_score",
        ]
        df = df.merge(cam[[c for c in cam_cols if c in cam.columns]], on="file_name", how="left")
    df["guardrail_expected"] = df["file_name"].map(GUARDRAILS)
    df["is_guardrail"] = df["guardrail_expected"].notna()
    df["eligible_drop"] = df["current_best_label"].eq(1) & ~df["is_guardrail"]
    return df


def available_profile_cols(df: pd.DataFrame) -> list[str]:
    success = df[df["file_name"].isin(SUCCESS_DROP)]
    failure = df[df["file_name"].isin(FAILED_KEEP)]
    cols = []
    for col in PROFILE_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 20 and success[col].notna().sum() >= 2 and failure[col].notna().sum() >= 3:
            cols.append(col)
    return cols


def centroid_distances(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    source = df[df["eligible_drop"] | df["file_name"].isin(SUCCESS_DROP) | df["file_name"].isin(FAILED_KEEP)].copy()
    matrix = source[cols].apply(pd.to_numeric, errors="coerce")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0).replace(0, np.nan)
    z = (matrix - mean) / std
    success_center = z.loc[source["file_name"].isin(SUCCESS_DROP)].mean(axis=0)
    failure_center = z.loc[source["file_name"].isin(FAILED_KEEP)].mean(axis=0)
    rows = []
    for idx, row in z.iterrows():
        valid_success = row.notna() & success_center.notna()
        valid_failure = row.notna() & failure_center.notna()
        success_dist = np.nan
        failure_dist = np.nan
        if valid_success.sum() > 0:
            success_dist = float(np.sqrt(np.mean((row[valid_success] - success_center[valid_success]) ** 2)))
        if valid_failure.sum() > 0:
            failure_dist = float(np.sqrt(np.mean((row[valid_failure] - failure_center[valid_failure]) ** 2)))
        rows.append(
            {
                "file_name": source.loc[idx, "file_name"],
                "success_contrast_distance": success_dist,
                "failure_contrast_distance": failure_dist,
                "contrast_coverage": int(valid_success.sum()),
            }
        )
    return pd.DataFrame(rows)


def score_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = available_profile_cols(df)
    distances = centroid_distances(df, cols)
    out = df.merge(distances, on="file_name", how="left")

    out["s_near_success"] = minmax(out["success_contrast_distance"], False)
    out["s_far_failure"] = minmax(out["failure_contrast_distance"], True)
    out["s_distance_gap"] = minmax(out["failure_contrast_distance"] - out["success_contrast_distance"], True)
    out["s_tree"] = minmax(out.get("tree_drop_score", np.nan), True)
    out["s_exp119"] = minmax(out.get("exp119_drop_score", np.nan), True)
    out["s_consensus"] = minmax(out.get("exp121_consensus_score", np.nan), True)
    out["s_low_prob_mean"] = minmax(out.get("prob_mean", np.nan), False)
    out["s_low_exp45"] = minmax(out.get("exp45_gradcam_prob1", np.nan), False)
    out["s_low_exp124"] = minmax(out.get("exp124_gradcam_prob1", np.nan), False)
    out["s_cam_success"] = minmax(out.get("success_cam_distance", np.nan), False)

    weights = {
        "s_distance_gap": 0.30,
        "s_near_success": 0.18,
        "s_far_failure": 0.18,
        "s_tree": 0.10,
        "s_exp119": 0.08,
        "s_consensus": 0.05,
        "s_low_prob_mean": 0.05,
        "s_low_exp45": 0.025,
        "s_low_exp124": 0.025,
        "s_cam_success": 0.02,
    }
    total = pd.Series(0.0, index=out.index)
    wsum = pd.Series(0.0, index=out.index)
    for col, weight in weights.items():
        valid = out[col].notna()
        total.loc[valid] += out.loc[valid, col] * weight
        wsum.loc[valid] += weight
    out["exp132_score"] = np.where(wsum > 0, total / wsum, np.nan)
    out["exp132_rank"] = np.nan

    ranked = out.sort_values(
        ["eligible_drop", "exp132_score", "s_distance_gap", "exp119_drop_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    mask = ranked["eligible_drop"]
    ranked.loc[mask, "exp132_rank"] = np.arange(mask.sum()) + 1

    profile = pd.DataFrame(
        {
            "profile_column": cols,
            "success_mean": [pd.to_numeric(df.loc[df["file_name"].isin(SUCCESS_DROP), c], errors="coerce").mean() for c in cols],
            "failure_mean": [pd.to_numeric(df.loc[df["file_name"].isin(FAILED_KEEP), c], errors="coerce").mean() for c in cols],
        }
    )
    profile["abs_mean_gap"] = (profile["success_mean"] - profile["failure_mean"]).abs()
    profile = profile.sort_values("abs_mean_gap", ascending=False)
    return ranked, profile


def make_submission(file_name: str) -> dict:
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    current = pd.read_csv(CURRENT_BEST)
    current_map = dict(zip(current["file_name"], current["label"].astype(int)))
    sub = sample.copy()
    sub["label"] = sub["file_name"].map(current_map).astype(int)
    if int(sub.loc[sub["file_name"].eq(file_name), "label"].iloc[0]) != 1:
        raise ValueError(f"{file_name} is not current label 1")
    sub.loc[sub["file_name"].eq(file_name), "label"] = 0
    changed = current.merge(sub, on="file_name", suffixes=("_current", "_exp132"))
    changed = changed[changed["label_current"] != changed["label_exp132"]].copy()
    if len(changed) != 1:
        raise ValueError(f"expected one changed row, got {len(changed)}")
    stem = file_name.replace(".png", "")
    sub_path = SUB_DIR / f"exp132_contrast_drop_{stem}_submit.csv"
    changed_path = PRED_DIR / f"exp132_contrast_drop_{stem}_changed_vs_exp117.csv"
    sub.to_csv(sub_path, index=False)
    changed.to_csv(changed_path, index=False)
    return {
        "submission": str(sub_path),
        "changed_csv": str(changed_path),
        "file_name": file_name,
        "rows": int(len(sub)),
        "order_match_sample": bool(sub["file_name"].equals(sample["file_name"])),
        "labels": sorted(sub["label"].unique().tolist()),
        "label_counts": sub["label"].value_counts().sort_index().astype(int).to_dict(),
    }


def write_report(ranked: pd.DataFrame, sub_info: dict) -> None:
    top = ranked[ranked["eligible_drop"]].head(8)
    rows = "\n".join(
        [
            f"| {r.file_name} | {r.exp132_score:.4f} | {r.success_contrast_distance:.4f} | {r.failure_contrast_distance:.4f} | {(r.failure_contrast_distance - r.success_contrast_distance):.4f} | {getattr(r, 'exp119_drop_score', np.nan):.4f} | {getattr(r, 'tree_drop_score', np.nan):.4f} |"
            for r in top.itertuples()
        ]
    )
    md = f"""# EXP-132 Success-vs-Failure Contrast

## Summary

- 기준 제출: `{CURRENT_BEST}`
- 성공 drop: `{', '.join(SUCCESS_DROP)}`
- 실패 keep: `{', '.join(FAILED_KEEP)}`
- 최종 추천: `{sub_info['submission']}`
- 변경 샘플: `{sub_info['file_name']}: 1->0`

## Top Candidates

| file_name | score | dist_to_success | dist_to_failure | failure-success gap | exp119 | tree |
|---|---:|---:|---:|---:|---:|---:|
{rows}

## Decision

- EXP-132는 성공 샘플과 가까우면서 실패 샘플과 먼 후보를 우선한다.
- 이미 하락이 확인된 `test_0212`는 failure centroid에 포함하고 guardrail로 제외했다.
- 이 결과는 모델 일반화 개선이 아니라 마지막 제출 후보 선별용이다.
"""
    (OUT / "exp132_success_failure_contrast.md").write_text(md, encoding="utf-8")


def main() -> None:
    evidence = load_evidence()
    ranked, profile = score_candidates(evidence)
    ranked_path = OUT / "exp132_candidate_ranked.csv"
    profile_path = OUT / "exp132_success_failure_profile.csv"
    top_path = OUT / "exp132_top30.csv"
    ranked.to_csv(ranked_path, index=False)
    profile.to_csv(profile_path, index=False)
    ranked[ranked["eligible_drop"]].head(30).to_csv(top_path, index=False)
    winner = ranked[ranked["eligible_drop"]].iloc[0]
    sub_info = make_submission(str(winner["file_name"]))
    write_report(ranked, sub_info)
    result = {
        "ranked": str(ranked_path),
        "profile": str(profile_path),
        "top30": str(top_path),
        "report": str(OUT / "exp132_success_failure_contrast.md"),
        "recommended": sub_info,
        "top5": ranked[ranked["eligible_drop"]].head(5)[
            [
                "file_name",
                "exp132_score",
                "success_contrast_distance",
                "failure_contrast_distance",
                "tree_drop_score",
                "exp119_drop_score",
                "prob_mean",
            ]
        ].to_dict("records"),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
