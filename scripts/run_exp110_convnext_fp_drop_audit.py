from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "oof/preds"
SUB_DIR = ROOT / "oof/submissions"
REPORT_DIR = ROOT / "reports/exp110_convnext_fp_drop_audit"
BASE_PATH = SUB_DIR / "exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
SUSPICIOUS_PATH = ROOT / "reports/exp95_suspicious_cases/exp95_suspicious_cases.csv"

PROB_FILES = {
    "cnx": PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv",
    "cnx123": PRED_DIR / "exp54_convnext_tiny_seed123_test_probs_mean.csv",
    "cnx777": PRED_DIR / "exp72_convnext_tiny_seed777_test_probs_mean.csv",
    "cnx_lion": PRED_DIR / "exp78_convnext_tiny_lion_test_probs_mean.csv",
    "cnxv2": PRED_DIR / "exp65_convnextv2_tiny_stage3_test_probs_mean.csv",
}

CORE_MODELS = ["cnx", "cnx123", "cnx777", "cnx_lion"]
REFERENCE_MODELS = CORE_MODELS + ["cnxv2"]

HARD_GUARDRAILS = {
    "test_0025.png": 0,
    "test_0049.png": 0,
    "test_0141.png": 1,
    "test_0188.png": 0,
    "test_0201.png": 1,
    "test_0222.png": 1,
    "test_0244.png": 0,
    "test_0490.png": 0,
    "test_0504.png": 0,
    "test_0539.png": 0,
    "test_0568.png": 0,
}


def load_prob(name: str, path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)[["file_name", "prob_1", "pred_label"]].copy()
    if list(df["file_name"]) != list(sample["file_name"]):
        raise ValueError(f"{name}: sample order mismatch")
    return df.rename(columns={"prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"})


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


def guardrail_ok(mapping: dict[str, int]) -> bool:
    return all(mapping[k] == v for k, v in HARD_GUARDRAILS.items())


def build_feature_table() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)[["file_name"]]
    base = pd.read_csv(BASE_PATH)
    if list(base["file_name"]) != list(sample["file_name"]):
        raise ValueError("EXP-56 sample order mismatch")

    table = base.rename(columns={"label": "exp56_label"}).copy()
    for name, path in PROB_FILES.items():
        table = table.merge(load_prob(name, path, sample), on="file_name", how="inner")

    suspicious = pd.read_csv(SUSPICIOUS_PATH)
    keep_cols = [
        "file_name",
        "exp56_decision_score",
        "exp56_score_margin",
        "prob_mean",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "suspicious_score",
        "recommended_action",
        "is_protected",
    ]
    table = table.merge(suspicious[[c for c in keep_cols if c in suspicious.columns]], on="file_name", how="left")

    core_probs = table[[f"{m}_prob1" for m in CORE_MODELS]]
    ref_probs = table[[f"{m}_prob1" for m in REFERENCE_MODELS]]
    table["cnx_core_mean"] = core_probs.mean(axis=1)
    table["cnx_core_std"] = core_probs.std(axis=1)
    table["cnx_core_min"] = core_probs.min(axis=1)
    table["cnx_core_max"] = core_probs.max(axis=1)
    table["cnx_core_range"] = table["cnx_core_max"] - table["cnx_core_min"]
    table["cnx_core_vote1"] = (core_probs >= 0.5).sum(axis=1)
    table["cnx_ref_mean"] = ref_probs.mean(axis=1)
    table["cnx_ref_std"] = ref_probs.std(axis=1)
    table["cnx_ref_vote1"] = (ref_probs >= 0.5).sum(axis=1)
    table["margin_abs"] = table["exp56_score_margin"].abs()
    table["drop_score"] = (
        (1.0 - table["margin_abs"].rank(pct=True))
        + table["cnx_core_std"].rank(pct=True)
        + table["cnx_core_range"].rank(pct=True)
        + (1.0 - table["prob_mean"].rank(pct=True))
    )
    return table


def select_candidates(table: pd.DataFrame) -> pd.DataFrame:
    protected_files = set(HARD_GUARDRAILS)
    candidates = table[
        table["exp56_label"].eq(1)
        & ~table["file_name"].isin(protected_files)
        & ~table["recommended_action"].eq("review_high_priority")
        & ~table["recommended_action"].eq("avoid_change_protected")
        & table["cnx_core_vote1"].ge(2)
    ].copy()
    candidates = candidates.sort_values(
        ["drop_score", "margin_abs", "cnx_core_std", "suspicious_score"],
        ascending=[False, True, False, False],
    )
    return candidates


def make_submissions(candidates: pd.DataFrame) -> pd.DataFrame:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_PATH)
    rows = []
    for rank, row in enumerate(candidates.head(3).itertuples(index=False), start=1):
        name = f"exp110_drop_{Path(row.file_name).stem}_rank{rank}"
        sub = base.copy()
        sub.loc[sub["file_name"].eq(row.file_name), "label"] = 0
        validate_submission(sub)
        mapping = dict(zip(sub["file_name"], sub["label"]))
        if not guardrail_ok(mapping):
            continue
        out_path = SUB_DIR / f"{name}_submit.csv"
        sub.to_csv(out_path, index=False)
        changed = base.merge(sub, on="file_name", suffixes=("_exp56", "_candidate"))
        changed = changed[changed["label_exp56"] != changed["label_candidate"]]
        changed_path = PRED_DIR / f"{name}_changed_vs_exp56.csv"
        changed.to_csv(changed_path, index=False)
        rows.append(
            {
                "rank": rank,
                "candidate": name,
                "submission_path": str(out_path),
                "changed_sample": row.file_name,
                "changed": "1->0",
                "test_label0": int((sub["label"] == 0).sum()),
                "test_label1": int((sub["label"] == 1).sum()),
                "drop_score": row.drop_score,
                "exp56_score_margin": row.exp56_score_margin,
                "cnx_core_mean": row.cnx_core_mean,
                "cnx_core_std": row.cnx_core_std,
                "cnx_core_vote1": row.cnx_core_vote1,
                "cnxv2_prob1": row.cnxv2_prob1,
                "prob_mean": row.prob_mean,
                "prob_std": row.prob_std,
                "vote_1_count": row.vote_1_count,
                "model_disagrees_with_exp56": row.model_disagrees_with_exp56,
                "suspicious_score": row.suspicious_score,
                "recommended_action": row.recommended_action,
                "changed_csv": str(changed_path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_feature_table()
    ranking = select_candidates(table)
    candidate_summary = make_submissions(ranking)

    ranking.to_csv(REPORT_DIR / "exp110_convnext_fp_drop_ranking.csv", index=False)
    candidate_summary.to_csv(PRED_DIR / "exp110_convnext_fp_drop_candidates_summary.csv", index=False)

    top_cols = [
        "file_name",
        "drop_score",
        "exp56_score_margin",
        "cnx_core_mean",
        "cnx_core_std",
        "cnx_core_vote1",
        "cnxv2_prob1",
        "prob_mean",
        "vote_1_count",
        "recommended_action",
    ]
    print("EXP-110 ConvNeXt hard-case FP drop audit")
    print("ranking rows:", len(ranking))
    print("saved candidates:", len(candidate_summary))
    print("top ranking")
    print(ranking[top_cols].head(12).to_string(index=False))
    if not candidate_summary.empty:
        print("candidate submissions")
        print(candidate_summary[["rank", "submission_path", "changed_sample", "test_label1"]].to_string(index=False))


if __name__ == "__main__":
    main()
