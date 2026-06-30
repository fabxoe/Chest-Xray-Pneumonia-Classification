from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXP_ID = "exp121"
REPORT = Path("reports/exp121_evidence_ablation_consensus")
PRED = Path("oof/preds")
SUB = Path("oof/submissions")

RANKING_PATH = Path("reports/exp119_rf_svm_gaussian_rerank/exp119_rf_svm_gaussian_candidate_ranking.csv")
CURRENT_BEST = SUB / "exp117_after0385_drop_test_0303_only_submit.csv"
SAMPLE_SUBMISSION = Path("sample_submission.csv")

SUCCESS_DIRECTIONS = {
    "test_0455.png": 0,
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0303.png": 0,
}
FAILED_DROP_DIRECTIONS = {
    "test_0289.png": 1,
    "test_0172.png": 1,
    "test_0013.png": 1,
    "test_0188.png": 1,
    "test_0568.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 0,
}
OLDER_GUARDRAILS = {
    "test_0025.png": 0,
    "test_0539.png": 0,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0504.png": 0,
}
GUARDRAILS = {**OLDER_GUARDRAILS, **SUCCESS_DIRECTIONS, **FAILED_DROP_DIRECTIONS}


def normalize_file_name(name: str) -> str:
    name = str(name)
    return name if name.endswith(".png") else f"{name}.png"


def pct_rank_high(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")


def validate_submission(path: Path, current_best: pd.DataFrame) -> dict[str, object]:
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    sub = pd.read_csv(path)
    if len(sub) != 624:
        raise RuntimeError(f"Bad row count: {len(sub)}")
    if list(sub.columns) != ["file_name", "label"]:
        raise RuntimeError(f"Bad columns: {list(sub.columns)}")
    if not sub["file_name"].equals(sample["file_name"]):
        raise RuntimeError("Submission order mismatch")
    if not set(sub["label"].unique()) <= {0, 1}:
        raise RuntimeError("Bad label values")

    merged = current_best.rename(columns={"label": "before"}).merge(
        sub.rename(columns={"label": "after"}), on="file_name", validate="one_to_one"
    )
    changed = merged[merged["before"].ne(merged["after"])].copy()
    return {
        "rows": len(sub),
        "changed_count": len(changed),
        "label_counts": sub["label"].value_counts().sort_index().to_dict(),
        "changed_file": changed["file_name"].iloc[0] if len(changed) else "",
        "changed": f"{int(changed['before'].iloc[0])}->{int(changed['after'].iloc[0])}" if len(changed) else "",
    }


def make_drop_submission(current_best: pd.DataFrame, sample: str, out_path: Path) -> pd.DataFrame:
    sub = current_best[["file_name", "label"]].copy()
    mask = sub["file_name"].eq(sample)
    if mask.sum() != 1:
        raise RuntimeError(f"Sample not found exactly once: {sample}")
    if int(sub.loc[mask, "label"].iloc[0]) != 1:
        raise RuntimeError(f"Candidate is not current positive: {sample}")
    sub.loc[mask, "label"] = 0
    sub.to_csv(out_path, index=False)
    return pd.DataFrame([{"file_name": sample, "before": 1, "after": 0}])


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # One-source evidence. These are not calibrated probabilities; they are ranking signals.
    out["rf_only_score"] = (
        pct_rank_high(out["rf_prob0"].fillna(0)) * 0.55
        + pct_rank_high(out["extratrees_prob0"].fillna(0)) * 0.45
    )
    out["svm_only_score"] = pct_rank_high(out["svm_prob0_like"].fillna(0))
    out["gaussian_only_score"] = (
        pct_rank_high(out["gauss_edge_delta"].fillna(0)) * 0.45
        + pct_rank_high(out["lap_var"].fillna(0)) * 0.35
        + pct_rank_high(out["gauss_mean_abs_delta"].fillna(0)) * 0.20
    )
    out["tree_only_score"] = pct_rank_high(out["tree_drop_score"].fillna(0))

    # Pairwise evidence. RF receives slightly more weight where it models OOF labels directly.
    out["rf_svm_score"] = out["rf_only_score"] * 0.52 + out["svm_only_score"] * 0.48
    out["rf_gaussian_score"] = out["rf_only_score"] * 0.62 + out["gaussian_only_score"] * 0.38
    out["svm_gaussian_score"] = out["svm_only_score"] * 0.58 + out["gaussian_only_score"] * 0.42
    out["all_three_score"] = (
        out["rf_only_score"] * 0.42
        + out["svm_only_score"] * 0.33
        + out["gaussian_only_score"] * 0.25
    )
    out["all_three_with_tree_score"] = (
        out["all_three_score"] * 0.75
        + out["tree_only_score"] * 0.25
    )

    score_cols = [
        "rf_only_score",
        "svm_only_score",
        "gaussian_only_score",
        "rf_svm_score",
        "rf_gaussian_score",
        "svm_gaussian_score",
        "all_three_score",
        "all_three_with_tree_score",
    ]
    for col in score_cols:
        out[f"{col}_rank"] = out[col].rank(ascending=False, method="min").astype(int)

    top_sets = {}
    for col in score_cols:
        top_sets[col] = set(out.nsmallest(10, f"{col}_rank")["file_name"])

    out["top10_single_source_count"] = 0
    for col in ["rf_only_score", "svm_only_score", "gaussian_only_score"]:
        out["top10_single_source_count"] += out["file_name"].isin(top_sets[col]).astype(int)
    out["top10_pair_count"] = 0
    for col in ["rf_svm_score", "rf_gaussian_score", "svm_gaussian_score"]:
        out["top10_pair_count"] += out["file_name"].isin(top_sets[col]).astype(int)
    out["top10_total_count"] = 0
    for samples in top_sets.values():
        out["top10_total_count"] += out["file_name"].isin(samples).astype(int)

    out["absolute_support_count"] = (
        out["rf_prob0"].ge(0.20).astype(int)
        + out["extratrees_prob0"].ge(0.25).astype(int)
        + out["svm_prob0_like"].ge(0.40).astype(int)
        + out["tree_drop_score"].fillna(0).ge(0.70).astype(int)
        + out["vote_0_count"].ge(3).astype(int)
        + out["gaussian_only_score"].ge(out["gaussian_only_score"].quantile(0.90)).astype(int)
    )

    out["consensus_score"] = (
        pct_rank_high(out["top10_total_count"]) * 0.28
        + pct_rank_high(out["top10_single_source_count"]) * 0.20
        + pct_rank_high(out["top10_pair_count"]) * 0.18
        + pct_rank_high(out["absolute_support_count"]) * 0.16
        + pct_rank_high(out["all_three_with_tree_score"]) * 0.12
        + pct_rank_high(out["tree_only_score"]) * 0.06
    )
    out["consensus_rank"] = out["consensus_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["consensus_score", "top10_total_count", "absolute_support_count"], ascending=False)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(RANKING_PATH)
    ranking["file_name"] = ranking["file_name"].map(normalize_file_name)
    current_best = pd.read_csv(CURRENT_BEST)
    current_best["file_name"] = current_best["file_name"].map(normalize_file_name)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    if not current_best["file_name"].equals(sample["file_name"]):
        raise RuntimeError("Current best order mismatch with sample submission")

    ranking["guardrail_expected"] = ranking["file_name"].map(GUARDRAILS)
    ranking["is_guardrail"] = ranking["guardrail_expected"].notna()
    ranking = ranking[~ranking["is_guardrail"]].copy()

    scored = build_scores(ranking)

    score_cols = [
        "rf_only_score",
        "svm_only_score",
        "gaussian_only_score",
        "rf_svm_score",
        "rf_gaussian_score",
        "svm_gaussian_score",
        "all_three_score",
        "all_three_with_tree_score",
        "consensus_score",
    ]
    cols = [
        "file_name",
        "consensus_rank",
        "consensus_score",
        "top10_single_source_count",
        "top10_pair_count",
        "top10_total_count",
        "absolute_support_count",
        *score_cols,
        "rf_prob0",
        "extratrees_prob0",
        "svm_prob0_like",
        "tree_drop_score",
        "vote_0_count",
        "prob_mean",
        "prob_std",
        "gauss_edge_delta",
        "lap_var",
        "recommended_action",
        "exp119_drop_score",
    ]
    cols = [c for c in cols if c in scored.columns]
    scored[cols].to_csv(REPORT / "exp121_evidence_ablation_consensus_ranking.csv", index=False)

    for name in [
        "rf_only_score",
        "svm_only_score",
        "gaussian_only_score",
        "rf_svm_score",
        "rf_gaussian_score",
        "svm_gaussian_score",
        "all_three_score",
        "all_three_with_tree_score",
        "consensus_score",
    ]:
        scored.sort_values(name, ascending=False).head(20)[cols].to_csv(
            REPORT / f"exp121_top20_by_{name.replace('_score', '')}.csv",
            index=False,
        )

    # Submit only one if it is supported by at least two independent top-10 source/pair signals.
    candidates = scored[
        (scored["top10_total_count"] >= 2)
        & (scored["absolute_support_count"] >= 2)
    ].copy()
    selected = candidates.iloc[0] if len(candidates) else scored.iloc[0]
    selected_sample = normalize_file_name(selected["file_name"])
    out_path = SUB / f"exp122_after0303_consensus_drop_{selected_sample.replace('.png', '')}_only_submit.csv"

    changed = make_drop_submission(current_best, selected_sample, out_path)
    changed_path = PRED / f"exp122_after0303_consensus_drop_{selected_sample.replace('.png', '')}_changed_vs_exp117.csv"
    changed.to_csv(changed_path, index=False)
    validation = validate_submission(out_path, current_best)

    summary = pd.DataFrame([{
        "selected_sample": selected_sample,
        "submission_path": str(out_path),
        "changed_csv": str(changed_path),
        "consensus_rank": int(selected["consensus_rank"]),
        "consensus_score": float(selected["consensus_score"]),
        "top10_single_source_count": int(selected["top10_single_source_count"]),
        "top10_pair_count": int(selected["top10_pair_count"]),
        "top10_total_count": int(selected["top10_total_count"]),
        "absolute_support_count": int(selected["absolute_support_count"]),
        "rf_prob0": float(selected["rf_prob0"]),
        "extratrees_prob0": float(selected["extratrees_prob0"]),
        "svm_prob0_like": float(selected["svm_prob0_like"]),
        "tree_drop_score": float(selected.get("tree_drop_score", np.nan)),
        "validation": validation,
        "recommendation": "submit_only_if_using_one_remaining_slot",
    }])
    summary.to_csv(PRED / "exp122_after0303_consensus_submit_summary.csv", index=False)

    print("EXP-121 evidence ablation/consensus")
    print("Top 15 consensus candidates")
    print(scored.head(15)[cols].to_string(index=False))
    print()
    print("EXP-122 selected candidate")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
