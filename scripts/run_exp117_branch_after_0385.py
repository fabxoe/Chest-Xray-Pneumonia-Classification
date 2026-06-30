from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "sample_submission.csv"
CURRENT_BEST_PATH = ROOT / "oof/submissions/exp116_after0057_drop_test_0385_only_submit.csv"
EXP114_SUMMARY_PATH = ROOT / "oof/preds/exp114_current_best_single_drop_candidates_summary.csv"
SUB_DIR = ROOT / "oof/submissions"
PRED_DIR = ROOT / "oof/preds"
REPORT_DIR = ROOT / "reports/exp117_branch_after_0385"

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
    "test_0057.png": 0,
    "test_0385.png": 0,
    "test_0490.png": 0,
    "test_0504.png": 0,
    "test_0539.png": 0,
    "test_0568.png": 0,
}

BRANCH_DROPS = [
    "test_0172.png",
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


def load_current_best() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BEST_PATH)
    validate_submission(current)
    for sample, expected in HARD_GUARDRAILS.items():
        actual = int(current.loc[current["file_name"].eq(sample), "label"].iloc[0])
        if actual != expected:
            raise ValueError(f"current best violates guardrail {sample}: {actual} != {expected}")
    return current


def main() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    current = load_current_best()
    summary = pd.read_csv(EXP114_SUMMARY_PATH).set_index("changed_sample")

    rows = []
    for sample in BRANCH_DROPS:
        if sample not in summary.index:
            raise ValueError(f"missing EXP-114 evidence for {sample}")
        if int(current.loc[current["file_name"].eq(sample), "label"].iloc[0]) != 1:
            raise ValueError(f"{sample} is not positive in current best")

        stem = Path(sample).stem
        name = f"exp117_after0385_drop_{stem}_only"
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
        changed_path = PRED_DIR / f"{name}_changed_vs_exp116.csv"
        sub.to_csv(out_path, index=False)
        changed.to_csv(changed_path, index=False)

        ev = summary.loc[sample]
        rows.append(
            {
                "priority": len(rows) + 1,
                "candidate": name,
                "submission_path": str(out_path),
                "changed_sample": sample,
                "changed_vs_exp116": "1->0",
                "test_label0": int((sub["label"] == 0).sum()),
                "test_label1": int((sub["label"] == 1).sum()),
                "tree_drop_score_from_exp114": ev["tree_drop_score"],
                "convnext_fold0_prob1": ev["convnext_fold0_prob1"],
                "cam_center_fraction": ev["cam_center_fraction"],
                "recommended_action": ev["recommended_action"],
                "changed_csv": str(changed_path),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(PRED_DIR / "exp117_after0385_branch_candidates_summary.csv", index=False)
    out.to_csv(REPORT_DIR / "exp117_after0385_branch_candidates_summary.csv", index=False)

    print("EXP-117 branch candidates after test_0385 success")
    print("current best:", CURRENT_BEST_PATH)
    print("created submissions:", len(out))
    print(out[["priority", "submission_path", "changed_sample", "test_label1", "tree_drop_score_from_exp114"]].to_string(index=False))


if __name__ == "__main__":
    main()
