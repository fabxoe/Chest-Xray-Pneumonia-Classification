from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP130_SELECTION = ROOT / "reports/exp130_success_candidate_gradcam/exp130_selection_summary.csv"
CURRENT_BEST = ROOT / "oof/submissions/exp117_after0385_drop_test_0303_only_submit.csv"
OUT = ROOT / "reports/exp131_final_submit_decision"
SUB_DIR = ROOT / "oof/submissions"
PRED_DIR = ROOT / "oof/preds"
OUT.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


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


def scenario_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["s_branch"] = minmax(out["exp129_score_adj"], True)
    out["s_success_profile"] = minmax(out["success_profile_distance"], False)
    out["s_success_cam"] = minmax(out["success_cam_distance"], False)
    out["s_low_exp45"] = minmax(out["exp45_gradcam_prob1"], False)
    out["s_low_exp124"] = minmax(out["exp124_gradcam_prob1"], False)
    out["s_low_prob_mean"] = minmax(out["prob_mean"], False)
    out["s_low_center"] = minmax(
        (out["exp45_cam_center_fraction"] + out["exp124_cam_center_fraction"]) / 2,
        False,
    )
    out["s_tree"] = minmax(out["tree_drop_score"], True)
    out["s_exp119"] = minmax(out["exp119_drop_score"], True)
    out["s_consensus"] = minmax(out["exp121_consensus_score"], True)

    scenarios = {
        "balanced": {
            "s_branch": 0.30,
            "s_success_profile": 0.20,
            "s_success_cam": 0.20,
            "s_low_exp45": 0.08,
            "s_low_exp124": 0.08,
            "s_tree": 0.06,
            "s_exp119": 0.05,
            "s_consensus": 0.03,
        },
        "branch_similarity_first": {
            "s_branch": 0.42,
            "s_success_profile": 0.28,
            "s_success_cam": 0.12,
            "s_low_exp45": 0.05,
            "s_low_exp124": 0.05,
            "s_tree": 0.04,
            "s_exp119": 0.03,
            "s_consensus": 0.01,
        },
        "xai_probability_first": {
            "s_branch": 0.16,
            "s_success_cam": 0.22,
            "s_low_exp45": 0.22,
            "s_low_exp124": 0.20,
            "s_low_center": 0.10,
            "s_success_profile": 0.06,
            "s_exp119": 0.04,
        },
        "ml_evidence_first": {
            "s_branch": 0.18,
            "s_exp119": 0.26,
            "s_consensus": 0.18,
            "s_tree": 0.16,
            "s_success_profile": 0.12,
            "s_success_cam": 0.10,
        },
        "public_branch_conservative": {
            "s_branch": 0.36,
            "s_success_profile": 0.24,
            "s_success_cam": 0.16,
            "s_low_prob_mean": 0.10,
            "s_tree": 0.08,
            "s_exp119": 0.04,
            "s_consensus": 0.02,
        },
    }
    for name, weights in scenarios.items():
        total = pd.Series(0.0, index=out.index)
        wsum = pd.Series(0.0, index=out.index)
        for col, weight in weights.items():
            valid = out[col].notna()
            total.loc[valid] += out.loc[valid, col] * weight
            wsum.loc[valid] += weight
        out[f"scenario_{name}"] = np.where(wsum > 0, total / wsum, np.nan)

    scenario_cols = [c for c in out.columns if c.startswith("scenario_")]
    ranks = []
    for col in scenario_cols:
        rank = out[col].rank(ascending=False, method="min")
        ranks.append(rank.rename(col.replace("scenario_", "rank_")))
    rank_df = pd.concat(ranks, axis=1)
    out = pd.concat([out, rank_df], axis=1)
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["scenario_win_count"] = (out[rank_cols] == 1).sum(axis=1)
    out["scenario_avg_rank"] = out[rank_cols].mean(axis=1)
    out["exp131_final_score"] = (
        0.35 * out["scenario_balanced"]
        + 0.30 * out["scenario_public_branch_conservative"]
        + 0.20 * out["scenario_branch_similarity_first"]
        + 0.10 * out["scenario_xai_probability_first"]
        + 0.05 * out["scenario_ml_evidence_first"]
    )
    return out.sort_values(
        ["scenario_win_count", "exp131_final_score", "scenario_avg_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def make_submission(file_name: str) -> dict:
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    current = pd.read_csv(CURRENT_BEST)
    current_map = dict(zip(current["file_name"], current["label"].astype(int)))
    sub = sample.copy()
    sub["label"] = sub["file_name"].map(current_map).astype(int)
    if int(sub.loc[sub["file_name"].eq(file_name), "label"].iloc[0]) != 1:
        raise ValueError(f"{file_name} is not current label 1")
    sub.loc[sub["file_name"].eq(file_name), "label"] = 0

    changed = current.merge(sub, on="file_name", suffixes=("_current", "_exp131"))
    changed = changed[changed["label_current"] != changed["label_exp131"]].copy()
    if len(changed) != 1:
        raise ValueError(f"expected one changed row, got {len(changed)}")

    stem = file_name.replace(".png", "")
    sub_path = SUB_DIR / f"exp131_final_recheck_drop_{stem}_submit.csv"
    changed_path = PRED_DIR / f"exp131_final_recheck_drop_{stem}_changed_vs_exp117.csv"
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


def write_markdown(decision: pd.DataFrame, sub_info: dict) -> None:
    top = decision.head(5)
    rows = "\n".join(
        [
            f"| {r.file_name} | {r.exp131_final_score:.4f} | {int(r.scenario_win_count)} | {r.scenario_avg_rank:.2f} | {r.exp129_score_adj:.4f} | {r.success_cam_distance:.4f} | {r.exp45_gradcam_prob1:.4f} | {r.exp124_gradcam_prob1:.4f} |"
            for r in top.itertuples()
        ]
    )
    md = f"""# EXP-131 Final Submit Decision

## Summary

- 기준 제출: `{CURRENT_BEST}`
- 목적: 남은 제출 2회 중 첫 제출 후보를 `test_0212`로 확정할지 로컬에서 재검증
- 최종 추천: `{sub_info['submission']}`
- 변경 샘플: `{sub_info['file_name']}: 1->0`

## Scenario Ranking

| file_name | final_score | win_count | avg_rank | exp129_score | cam_distance | exp45_p1 | exp124_p1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Decision

- `test_0212`는 balanced / branch similarity / ML evidence / public-branch conservative 시나리오에서 가장 안정적이다.
- `test_0267`은 ConvNeXt 확률만 보면 강한 대안이지만, public branch와 ML evidence 근거가 `test_0212`보다 약하다.
- 따라서 남은 제출 2회 중 첫 제출은 `test_0212: 1->0` 후보가 가장 타당하다.
"""
    (OUT / "exp131_final_submit_decision.md").write_text(md, encoding="utf-8")


def main() -> None:
    if not EXP130_SELECTION.exists():
        raise FileNotFoundError(EXP130_SELECTION)
    df = pd.read_csv(EXP130_SELECTION)
    decision = scenario_scores(df)
    decision_path = OUT / "exp131_final_decision_table.csv"
    decision.to_csv(decision_path, index=False)

    winner = decision.iloc[0]
    sub_info = make_submission(str(winner["file_name"]))
    write_markdown(decision, sub_info)
    result = {
        "decision_table": str(decision_path),
        "report": str(OUT / "exp131_final_submit_decision.md"),
        "recommended": sub_info,
        "top5": decision.head(5)[
            [
                "file_name",
                "exp131_final_score",
                "scenario_win_count",
                "scenario_avg_rank",
                "exp129_score_adj",
                "success_cam_distance",
                "exp45_gradcam_prob1",
                "exp124_gradcam_prob1",
            ]
        ].to_dict("records"),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
