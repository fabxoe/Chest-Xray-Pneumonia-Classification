from pathlib import Path

import pandas as pd


EXP_ID = "exp97"
BEST = Path("oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv")
SUSPICIOUS = Path("reports/exp95_suspicious_cases/exp95_suspicious_cases.csv")
OUT_DIR = Path("oof/preds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_DIRS = [Path("oof/submissions"), Path("submissions")]

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
}

CONDITIONAL_FORBIDDEN = [
    ("test_0188.png", 1, "test_0568.png", 1, "avoid test_0188=1 and test_0568=1 together"),
]

LABEL1_SOFT_RANGE = (395, 397)
LABEL1_HARD_RANGE = (392, 402)
MAX_HIGH_PRIORITY_CHANGES = 0


def load_submission(path):
    df = pd.read_csv(path)
    if list(df.columns) != ["file_name", "label"]:
        raise ValueError(f"{path}: expected columns file_name,label, got {list(df.columns)}")
    if len(df) != 624:
        raise ValueError(f"{path}: expected 624 rows, got {len(df)}")
    if sorted(df["label"].unique().tolist()) not in ([0], [1], [0, 1]):
        raise ValueError(f"{path}: label must be 0/1")
    return df


def collect_submission_paths():
    paths = []
    for directory in SUBMISSION_DIRS:
        if directory.exists():
            paths.extend(directory.glob("*submit.csv"))
    return sorted(set(paths), key=lambda p: str(p))


def evaluate(path, base, sample_order, suspicious):
    row = {
        "path": str(path),
        "file_name": path.name,
        "source_dir": str(path.parent),
        "valid_format": False,
        "pass_guardrail": False,
        "risk_score": 0,
        "fail_reasons": "",
        "warnings": "",
    }
    fail = []
    warnings = []
    try:
        df = load_submission(path)
        row["valid_format"] = True
    except Exception as exc:
        row["fail_reasons"] = str(exc)
        row["risk_score"] = 999
        return row

    if not df["file_name"].equals(sample_order["file_name"]):
        fail.append("sample_order_mismatch")
        df = df.sort_values("file_name").reset_index(drop=True)
        base_eval = base.sort_values("file_name").reset_index(drop=True)
    else:
        base_eval = base

    changed = df["label"].to_numpy() != base_eval["label"].to_numpy()
    changed_files = df.loc[changed, "file_name"].tolist()
    row["changed_vs_exp56"] = int(changed.sum())
    row["changed_files"] = ",".join(changed_files[:30])
    row["label0"] = int((df["label"] == 0).sum())
    row["label1"] = int((df["label"] == 1).sum())

    for file_name, expected in HARD_GUARDRAILS.items():
        actual = int(df.loc[df["file_name"].eq(file_name), "label"].iloc[0])
        row[f"{file_name.replace('.png', '')}_label"] = actual
        if actual != expected:
            fail.append(f"{file_name}={actual}, expected {expected}")

    for a, a_val, b, b_val, reason in CONDITIONAL_FORBIDDEN:
        a_actual = int(df.loc[df["file_name"].eq(a), "label"].iloc[0])
        b_actual = int(df.loc[df["file_name"].eq(b), "label"].iloc[0])
        if a_actual == a_val and b_actual == b_val:
            fail.append(reason)

    low_hard, high_hard = LABEL1_HARD_RANGE
    low_soft, high_soft = LABEL1_SOFT_RANGE
    if not (low_hard <= row["label1"] <= high_hard):
        fail.append(f"label1_outside_hard_range:{row['label1']}")
    elif not (low_soft <= row["label1"] <= high_soft):
        warnings.append(f"label1_outside_soft_range:{row['label1']}")

    sus_changed = suspicious[suspicious["file_name"].isin(changed_files)].copy()
    high_priority_changed = sus_changed["recommended_action"].eq("review_high_priority").sum()
    protected_changed = sus_changed["recommended_action"].eq("avoid_change_protected").sum()
    review_changed = sus_changed["recommended_action"].eq("review").sum()
    row["high_priority_changed"] = int(high_priority_changed)
    row["protected_changed"] = int(protected_changed)
    row["review_changed"] = int(review_changed)
    row["mean_suspicious_score_changed"] = float(sus_changed["suspicious_score"].mean()) if len(sus_changed) else 0.0

    if high_priority_changed > MAX_HIGH_PRIORITY_CHANGES:
        fail.append(f"high_priority_changed:{high_priority_changed}")
    if protected_changed > 0:
        fail.append(f"avoid_change_protected_changed:{protected_changed}")
    if review_changed >= 3:
        warnings.append(f"many_review_samples_changed:{review_changed}")

    risk = 0
    risk += 100 * len(fail)
    risk += 15 * len(warnings)
    risk += min(row["changed_vs_exp56"], 50)
    risk += 20 * int(protected_changed)
    risk += 10 * int(high_priority_changed)
    risk += 3 * int(review_changed)
    risk += int(round(10 * row["mean_suspicious_score_changed"]))
    row["risk_score"] = int(risk)
    row["fail_reasons"] = "; ".join(fail)
    row["warnings"] = "; ".join(warnings)
    row["pass_guardrail"] = len(fail) == 0
    return row


def decision_tree_text():
    hard = "\n".join([f"  - {k} must be {v}" for k, v in HARD_GUARDRAILS.items()])
    return f"""# EXP-97 Guardrail Tree Filter

Purpose: rank existing submission files by public-informed guardrails without using leaderboard probing as a label search loop.

Decision tree:

1. Validate format.
   - 624 rows
   - columns: file_name,label
   - sample_submission order
   - labels in {{0,1}}
2. Apply hard guardrails:
{hard}
3. Apply conditional guardrails:
   - do not set both test_0188=1 and test_0568=1
4. Apply label distribution checks:
   - hard range: label1 in [{LABEL1_HARD_RANGE[0]}, {LABEL1_HARD_RANGE[1]}]
   - soft range: label1 in [{LABEL1_SOFT_RANGE[0]}, {LABEL1_SOFT_RANGE[1]}]
5. Apply suspicious-case checks:
   - avoid changing EXP-95 avoid_change_protected samples
   - avoid changing review_high_priority samples unless there is new model evidence
   - warn if 3 or more EXP-95 review samples change

Interpretation:

- pass_guardrail=True does not mean the file should be submitted.
- pass_guardrail=False means it conflicts with public-informed safety rules.
- risk_score is for prioritization only, not a validation metric.
"""


def main():
    sample_order = pd.read_csv("sample_submission.csv")[["file_name"]]
    base = load_submission(BEST)
    if not base["file_name"].equals(sample_order["file_name"]):
        raise ValueError("EXP-56 order mismatch")
    suspicious = pd.read_csv(SUSPICIOUS)

    rows = [evaluate(path, base, sample_order, suspicious) for path in collect_submission_paths()]
    report = pd.DataFrame(rows).sort_values(["pass_guardrail", "risk_score", "changed_vs_exp56"], ascending=[False, True, True])
    report.to_csv(OUT_DIR / "exp97_guardrail_tree_report.csv", index=False)

    passed = report[report["pass_guardrail"]].copy()
    passed.to_csv(OUT_DIR / "exp97_guardrail_tree_passed.csv", index=False)
    (OUT_DIR / "exp97_guardrail_tree.md").write_text(decision_tree_text(), encoding="utf-8")

    print("EXP-97 guardrail tree filter")
    print(f"evaluated: {len(report)}")
    print(f"passed: {int(report['pass_guardrail'].sum())}")
    print(f"failed: {int((~report['pass_guardrail']).sum())}")
    print()
    cols = ["path", "pass_guardrail", "risk_score", "changed_vs_exp56", "label1", "fail_reasons", "warnings"]
    print(report[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
