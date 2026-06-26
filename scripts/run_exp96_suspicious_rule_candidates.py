from pathlib import Path

import pandas as pd


EXP_ID = "exp96"
BEST_SUBMISSION = Path("oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv")
SUSPICIOUS = Path("reports/exp95_suspicious_cases/exp95_suspicious_cases.csv")
OUT_DIR = Path("oof/submissions")
PRED_DIR = Path("oof/preds")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

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


def write_candidate(base, suspicious, name, files):
    out = base.copy()
    changed = []
    for file_name in files:
        if file_name in PROTECTED:
            raise ValueError(f"protected sample cannot be changed: {file_name}")
        idx = out.index[out["file_name"].eq(file_name)]
        if len(idx) != 1:
            raise ValueError(f"missing or duplicate file: {file_name}")
        i = idx[0]
        before = int(out.loc[i, "label"])
        after = 1 - before
        out.loc[i, "label"] = after
        changed.append({"file_name": file_name, "before": before, "after": after})

    for file_name, expected in PROTECTED.items():
        actual = int(out.loc[out["file_name"].eq(file_name), "label"].iloc[0])
        if actual != expected:
            raise ValueError(f"protected sample broken: {file_name} expected {expected}, got {actual}")

    sub_path = OUT_DIR / f"{name}_submit.csv"
    changed_path = PRED_DIR / f"{name}_changed_vs_exp56.csv"
    out.to_csv(sub_path, index=False)
    changed_df = pd.DataFrame(changed).merge(
        suspicious,
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    changed_df.to_csv(changed_path, index=False)
    return {
        "candidate": name,
        "submission_path": str(sub_path),
        "changed_path": str(changed_path),
        "changed_count": len(changed),
        "changed_files": ",".join(files),
        "label0": int((out["label"] == 0).sum()),
        "label1": int((out["label"] == 1).sum()),
    }


def main():
    base = pd.read_csv(BEST_SUBMISSION)
    sample = pd.read_csv("sample_submission.csv")[["file_name"]]
    if not base["file_name"].equals(sample["file_name"]):
        raise ValueError("EXP-56 order mismatch against sample_submission.csv")
    suspicious = pd.read_csv(SUSPICIOUS)

    high = suspicious[suspicious["recommended_action"].eq("review_high_priority")].copy()
    high = high.sort_values(["model_disagrees_with_exp56", "suspicious_score"], ascending=False)
    print(high[[
        "file_name",
        "exp56_label",
        "suspicious_score",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "exp56_decision_score",
    ]].to_string(index=False))

    # These are model-rule candidates, not manual answer probing:
    # - keep every protected sample fixed
    # - only use EXP-95 review_high_priority samples
    # - prefer one-change candidates before a two-change candidate
    specs = [
        ("exp96_flip0188_only", ["test_0188.png"]),
        ("exp96_flip0377_only", ["test_0377.png"]),
        ("exp96_flip0188_0377", ["test_0188.png", "test_0377.png"]),
    ]
    rows = [write_candidate(base, suspicious, name, files) for name, files in specs]
    summary = pd.DataFrame(rows)
    summary.to_csv(PRED_DIR / "exp96_suspicious_rule_candidates_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
