from __future__ import annotations

from pathlib import Path

import pandas as pd


EXP_ID = "exp109"
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
    "test_0568.png": 0,
}

LABEL1_SOFT_RANGE = (395, 397)
LABEL1_HARD_RANGE = (392, 402)
MAX_HIGH_PRIORITY_CHANGES = 0


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["file_name", "label"]:
        raise ValueError(f"{path}: expected columns file_name,label, got {list(df.columns)}")
    if len(df) != 624:
        raise ValueError(f"{path}: expected 624 rows, got {len(df)}")
    if sorted(df["label"].unique().tolist()) not in ([0], [1], [0, 1]):
        raise ValueError(f"{path}: label must be 0/1")
    return df


def collect_submission_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in SUBMISSION_DIRS:
        if directory.exists():
            paths.extend(directory.glob("*submit.csv"))
    return sorted(set(paths), key=lambda p: str(p))


def evaluate(path: Path, base: pd.DataFrame, sample_order: pd.DataFrame, suspicious: pd.DataFrame) -> dict:
    row = {
        "path": str(path),
        "file_name": path.name,
        "source_dir": str(path.parent),
        "valid_format": False,
        "pass_guardrail": False,
        "pass_and_nonduplicate": False,
        "risk_score": 0,
        "fail_reasons": "",
        "warnings": "",
    }
    fail: list[str] = []
    warnings: list[str] = []
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
    row["changed_files"] = ",".join(changed_files[:50])
    row["is_duplicate_exp56"] = row["changed_vs_exp56"] == 0
    row["label0"] = int((df["label"] == 0).sum())
    row["label1"] = int((df["label"] == 1).sum())

    for file_name, expected in HARD_GUARDRAILS.items():
        actual = int(df.loc[df["file_name"].eq(file_name), "label"].iloc[0])
        row[f"{file_name.replace('.png', '')}_label"] = actual
        if actual != expected:
            fail.append(f"{file_name}={actual}, expected {expected}")

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
    keep_changed = sus_changed["recommended_action"].eq("keep").sum()
    row["high_priority_changed"] = int(high_priority_changed)
    row["protected_changed"] = int(protected_changed)
    row["review_changed"] = int(review_changed)
    row["keep_changed"] = int(keep_changed)
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
    row["pass_and_nonduplicate"] = row["pass_guardrail"] and not row["is_duplicate_exp56"]
    return row


def write_decision_tree(report: pd.DataFrame) -> None:
    hard = "\n".join([f"  - {k} must be {v}" for k, v in HARD_GUARDRAILS.items()])
    text = f"""# EXP-109 Extended Guardrail Filter

Purpose: re-evaluate every existing submission candidate after EXP-108 showed that `test_0568=1` also lowers public score.

Evaluated files: {len(report)}
Passed guardrail: {int(report['pass_guardrail'].sum())}
Passed and non-duplicate against EXP-56: {int(report['pass_and_nonduplicate'].sum())}

Hard guardrails:
{hard}

Label distribution:
  - hard label1 range: [{LABEL1_HARD_RANGE[0]}, {LABEL1_HARD_RANGE[1]}]
  - soft label1 range: [{LABEL1_SOFT_RANGE[0]}, {LABEL1_SOFT_RANGE[1]}]

Suspicious-case rules:
  - do not change `avoid_change_protected`
  - do not change `review_high_priority`
  - warn if 3 or more `review` samples change

Interpretation:
  - `pass_guardrail=True` means the file does not violate current public-informed guardrails.
  - `pass_and_nonduplicate=True` is the only group worth considering as a new submission candidate.
  - This is a filtering analysis, not evidence that a candidate will improve public score.
"""
    (OUT_DIR / f"{EXP_ID}_extended_guardrail_tree.md").write_text(text, encoding="utf-8")


def main() -> None:
    sample_order = pd.read_csv("sample_submission.csv")[["file_name"]]
    base = load_submission(BEST)
    if not base["file_name"].equals(sample_order["file_name"]):
        raise ValueError("EXP-56 order mismatch")
    suspicious = pd.read_csv(SUSPICIOUS)

    rows = [evaluate(path, base, sample_order, suspicious) for path in collect_submission_paths()]
    report = pd.DataFrame(rows).sort_values(
        ["pass_and_nonduplicate", "pass_guardrail", "risk_score", "changed_vs_exp56"],
        ascending=[False, False, True, True],
    )
    report.to_csv(OUT_DIR / f"{EXP_ID}_extended_guardrail_report.csv", index=False)

    passed = report[report["pass_guardrail"]].copy()
    passed.to_csv(OUT_DIR / f"{EXP_ID}_extended_guardrail_passed.csv", index=False)

    candidates = report[report["pass_and_nonduplicate"]].copy()
    candidates.to_csv(OUT_DIR / f"{EXP_ID}_extended_guardrail_candidates.csv", index=False)
    write_decision_tree(report)

    print("EXP-109 extended guardrail filter")
    print(f"evaluated: {len(report)}")
    print(f"passed: {int(report['pass_guardrail'].sum())}")
    print(f"passed non-duplicate: {int(report['pass_and_nonduplicate'].sum())}")
    print()
    cols = [
        "path",
        "pass_guardrail",
        "pass_and_nonduplicate",
        "risk_score",
        "changed_vs_exp56",
        "label1",
        "fail_reasons",
        "warnings",
    ]
    print(report[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
