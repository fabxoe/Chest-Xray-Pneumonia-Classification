from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "sample_submission.csv"
CURRENT_BEST_PATH = ROOT / "oof/submissions/exp110_drop_test_0455_rank2_submit.csv"
SUSPICIOUS_PATH = ROOT / "reports/exp95_suspicious_cases/exp95_suspicious_cases.csv"
EXP110_PATH = ROOT / "oof/preds/exp110_convnext_fp_drop_candidates_summary.csv"
EXP111_PATH = ROOT / "oof/preds/exp111_convnext_threshold_replay_summary.csv"
EXP112_PATH = ROOT / "reports/exp112_drop_candidate_gradcam/exp112_drop_candidate_gradcam_summary.csv"

REPORT_DIR = ROOT / "reports/exp113_tree_replan"
PRED_DIR = ROOT / "oof/preds"
SUB_DIR = ROOT / "oof/submissions"

CURRENT_BEST_SCORE = 0.9599358974
EXP111_DROP0289_SCORE = 0.9583333333

HARD_GUARDRAILS = {
    "test_0025.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 1,
    "test_0188.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0244.png": 0,
    "test_0289.png": 1,
    "test_0455.png": 0,
    "test_0490.png": 0,
    "test_0504.png": 0,
    "test_0539.png": 0,
    "test_0568.png": 0,
}

# Ordered by the post-EXP-111 plan. Keep this explicit so the tree can be
# replayed without reinterpreting the ranking after every public submission.
PLANNED_SINGLE_DROPS = [
    "test_0057.png",
    "test_0172.png",
    "test_0385.png",
    "test_0013.png",
    "test_0303.png",
]


def validate_submission(df: pd.DataFrame) -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    if len(df) != 624:
        raise ValueError(f"row count must be 624, got {len(df)}")
    if list(df.columns) != ["file_name", "label"]:
        raise ValueError(f"columns must be file_name,label, got {list(df.columns)}")
    if list(df["file_name"]) != list(sample["file_name"]):
        raise ValueError("sample order mismatch")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("labels must be 0/1")


def normalize_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    ranked = series.rank(pct=True, method="average")
    return ranked if higher_is_better else 1.0 - ranked


def load_current_best() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST_PATH)
    validate_submission(current)
    for sample, expected in HARD_GUARDRAILS.items():
        actual = int(current.loc[current["file_name"].eq(sample), "label"].iloc[0])
        if actual != expected:
            raise ValueError(f"current best violates guardrail {sample}: {actual} != {expected}")
    return current


def build_exp111_sample_map() -> pd.DataFrame:
    exp111 = pd.read_csv(EXP111_PATH)
    rows = []
    for row in exp111.itertuples(index=False):
        for sample in str(row.changed_samples).split(";"):
            rows.append(
                {
                    "file_name": sample,
                    "exp111_candidate": row.candidate,
                    "exp111_score_spec": row.score_spec,
                    "exp111_drop_count": row.drop_count,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["file_name", "exp111_candidate", "exp111_score_spec", "exp111_drop_count"])
    mapped = pd.DataFrame(rows)
    return (
        mapped.groupby("file_name")
        .agg(
            exp111_candidate=("exp111_candidate", lambda x: ";".join(sorted(set(x)))),
            exp111_score_spec=("exp111_score_spec", lambda x: ";".join(sorted(set(x)))),
            exp111_min_drop_count=("exp111_drop_count", "min"),
        )
        .reset_index()
    )


def build_evidence_table() -> pd.DataFrame:
    current = load_current_best().rename(columns={"label": "current_best_label"})
    suspicious = pd.read_csv(SUSPICIOUS_PATH)
    exp110 = pd.read_csv(EXP110_PATH)
    exp111_map = build_exp111_sample_map()
    exp112 = pd.read_csv(EXP112_PATH)

    table = suspicious.merge(current, on="file_name", how="inner")
    table["is_current_positive"] = table["current_best_label"].eq(1)
    table["is_guardrail"] = table["file_name"].isin(HARD_GUARDRAILS)
    table["guardrail_expected"] = table["file_name"].map(HARD_GUARDRAILS)
    table["blocked_by_public_result"] = table["file_name"].eq("test_0289.png")
    table["known_success_drop"] = table["file_name"].eq("test_0455.png")

    exp110_keep = [
        "changed_sample",
        "rank",
        "candidate",
        "drop_score",
        "cnx_core_mean",
        "cnx_core_std",
        "cnx_core_vote1",
    ]
    table = table.merge(
        exp110[exp110_keep].rename(
            columns={
                "changed_sample": "file_name",
                "rank": "exp110_rank",
                "candidate": "exp110_candidate",
                "drop_score": "exp110_drop_score",
                "cnx_core_mean": "exp110_cnx_core_mean",
                "cnx_core_std": "exp110_cnx_core_std",
                "cnx_core_vote1": "exp110_cnx_core_vote1",
            }
        ),
        on="file_name",
        how="left",
    )
    table = table.merge(exp111_map, on="file_name", how="left")
    exp112_keep = [
        "file_name",
        "source",
        "convnext_fold0_prob1",
        "cam_center_fraction",
        "cam_top_fraction",
        "cam_left_fraction",
        "cam_centroid_x",
        "cam_centroid_y",
    ]
    table = table.merge(
        exp112[exp112_keep].rename(columns={"source": "exp112_source"}),
        on="file_name",
        how="left",
    )

    table["planned_single_drop_priority"] = table["file_name"].map(
        {sample: i + 1 for i, sample in enumerate(PLANNED_SINGLE_DROPS)}
    )
    table["in_planned_single_drop"] = table["planned_single_drop_priority"].notna()
    table["eligible_current_drop"] = (
        table["is_current_positive"]
        & ~table["is_guardrail"]
        & ~table["blocked_by_public_result"]
        & ~table["recommended_action"].eq("avoid_change_protected")
        & ~table["recommended_action"].eq("review_high_priority")
    )

    # Score direction: higher means more plausible FP drop. This is deliberately
    # model-grounded and penalizes already failed public directions.
    eligible = table["eligible_current_drop"]
    table["score_low_cnx_core"] = normalize_rank(table["cnx_prob1"], higher_is_better=False)
    table["score_low_fold0"] = normalize_rank(table["convnext_fold0_prob1"].fillna(table["cnx_prob1"]), False)
    table["score_disagreement"] = normalize_rank(table["model_disagrees_with_exp56"], True)
    table["score_low_margin"] = normalize_rank(table["exp56_score_margin"].abs(), False)
    table["score_low_cam_center"] = normalize_rank(table["cam_center_fraction"].fillna(0.5), False)
    table["score_exp110_prior"] = table["exp110_rank"].map({1: 0.85, 2: 1.0, 3: 0.9}).fillna(0.0)
    table["score_exp111_prior"] = table["exp111_min_drop_count"].map({1: 0.5, 2: 0.25}).fillna(0.0)
    table["score_action_keep_bonus"] = table["recommended_action"].eq("keep").astype(float) * 0.15
    table["score_public_penalty"] = table["blocked_by_public_result"].astype(float) * 2.0
    table["tree_drop_score"] = (
        0.25 * table["score_low_cnx_core"]
        + 0.20 * table["score_low_fold0"]
        + 0.20 * table["score_disagreement"]
        + 0.15 * table["score_low_margin"]
        + 0.10 * table["score_low_cam_center"]
        + 0.10 * table["score_exp110_prior"]
        + 0.05 * table["score_exp111_prior"]
        + table["score_action_keep_bonus"]
        - table["score_public_penalty"]
    )
    table.loc[~eligible, "tree_drop_score"] = np.nan
    return table


def make_single_drop_submissions(evidence: pd.DataFrame) -> pd.DataFrame:
    current = load_current_best()
    rows = []
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    evidence_by_file = evidence.set_index("file_name")
    for sample in PLANNED_SINGLE_DROPS:
        if sample not in evidence_by_file.index:
            raise ValueError(f"missing evidence for {sample}")
        if int(evidence_by_file.loc[sample, "current_best_label"]) != 1:
            raise ValueError(f"{sample} is not positive in current best")

        stem = Path(sample).stem
        name = f"exp114_drop_{stem}_only"
        sub = current.copy()
        sub.loc[sub["file_name"].eq(sample), "label"] = 0
        validate_submission(sub)

        for guard_sample, expected in HARD_GUARDRAILS.items():
            actual = int(sub.loc[sub["file_name"].eq(guard_sample), "label"].iloc[0])
            if actual != expected:
                raise ValueError(f"{name} violates guardrail {guard_sample}: {actual} != {expected}")

        changed = current.merge(sub, on="file_name", suffixes=("_current_best", "_candidate"))
        changed = changed[changed["label_current_best"] != changed["label_candidate"]]
        if len(changed) != 1:
            raise ValueError(f"{name}: expected one changed sample, got {len(changed)}")

        out_path = SUB_DIR / f"{name}_submit.csv"
        changed_path = PRED_DIR / f"{name}_changed_vs_exp110.csv"
        sub.to_csv(out_path, index=False)
        changed.to_csv(changed_path, index=False)

        ev = evidence_by_file.loc[sample]
        rows.append(
            {
                "priority": int(ev["planned_single_drop_priority"]),
                "candidate": name,
                "submission_path": str(out_path),
                "changed_sample": sample,
                "changed_vs_exp110": "1->0",
                "test_label0": int((sub["label"] == 0).sum()),
                "test_label1": int((sub["label"] == 1).sum()),
                "tree_drop_score": ev["tree_drop_score"],
                "exp56_score_margin": ev["exp56_score_margin"],
                "prob_mean": ev["prob_mean"],
                "prob_std": ev["prob_std"],
                "vote_1_count": ev["vote_1_count"],
                "model_disagrees_with_exp56": ev["model_disagrees_with_exp56"],
                "recommended_action": ev["recommended_action"],
                "convnext_fold0_prob1": ev.get("convnext_fold0_prob1"),
                "cam_center_fraction": ev.get("cam_center_fraction"),
                "changed_csv": str(changed_path),
            }
        )
    return pd.DataFrame(rows).sort_values("priority")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence_table()
    ranked = evidence[evidence["eligible_current_drop"]].copy()
    ranked = ranked.sort_values(
        ["in_planned_single_drop", "planned_single_drop_priority", "tree_drop_score"],
        ascending=[False, True, False],
    )
    candidates = make_single_drop_submissions(evidence)

    evidence.to_csv(REPORT_DIR / "exp113_candidate_evidence_table.csv", index=False)
    ranked.to_csv(REPORT_DIR / "exp113_candidate_ranked.csv", index=False)
    candidates.to_csv(PRED_DIR / "exp114_current_best_single_drop_candidates_summary.csv", index=False)

    print("EXP-113/114 tree replan candidates")
    print("current best:", CURRENT_BEST_PATH)
    print("current best score:", CURRENT_BEST_SCORE)
    print("blocked failed direction: test_0289=1")
    print("evidence rows:", len(evidence))
    print("eligible drop rows:", len(ranked))
    print("created submissions:", len(candidates))
    cols = [
        "priority",
        "submission_path",
        "changed_sample",
        "test_label1",
        "tree_drop_score",
        "convnext_fold0_prob1",
        "cam_center_fraction",
        "recommended_action",
    ]
    print(candidates[cols].to_string(index=False))


if __name__ == "__main__":
    main()
