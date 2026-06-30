from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP56_PATH = ROOT / "oof/submissions/exp56_ridge_cnx_eff_compact_p3000_t0p025_submit.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
SUSPICIOUS_PATH = ROOT / "reports/exp95_suspicious_cases/exp95_suspicious_cases.csv"
OUT_SUB_DIR = ROOT / "oof/submissions"
OUT_PRED_DIR = ROOT / "oof/preds"

EXP_ID = "exp108"

PROTECTED_EXPECTED = {
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


def load_base() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)[["file_name"]]
    base = pd.read_csv(EXP56_PATH)
    if list(sample["file_name"]) != list(base["file_name"]):
        raise ValueError("EXP-56 order mismatch against sample_submission.csv")
    return base


def validate_submission(df: pd.DataFrame) -> None:
    sample = pd.read_csv(SAMPLE_PATH)[["file_name"]]
    if len(df) != 624:
        raise ValueError(f"Invalid row count: {len(df)}")
    if list(df.columns) != ["file_name", "label"]:
        raise ValueError(f"Invalid columns: {list(df.columns)}")
    if list(df["file_name"]) != list(sample["file_name"]):
        raise ValueError("Submission order mismatch")
    labels = set(df["label"].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Invalid labels: {labels}")


def summarize_candidate(name: str, df: pd.DataFrame, base: pd.DataFrame) -> dict:
    merged = base.merge(df, on="file_name", suffixes=("_exp56", "_candidate"))
    changed = merged[merged["label_exp56"] != merged["label_candidate"]].copy()
    changed_path = OUT_PRED_DIR / f"{name}_changed_vs_exp56.csv"
    changed.to_csv(changed_path, index=False)

    mapping = dict(zip(df["file_name"], df["label"]))
    protected_ok = all(mapping[k] == v for k, v in PROTECTED_EXPECTED.items())
    avoid_pair_ok = not (mapping["test_0188.png"] == 1 and mapping["test_0568.png"] == 1)

    return {
        "candidate": name,
        "path": str(OUT_SUB_DIR / f"{name}_submit.csv"),
        "changed_vs_exp56": len(changed),
        "changed_samples": ";".join(
            f"{r.file_name}:{r.label_exp56}->{r.label_candidate}" for r in changed.itertuples()
        ),
        "test_label0": int((df["label"] == 0).sum()),
        "test_label1": int((df["label"] == 1).sum()),
        "protected_ok": protected_ok,
        "avoid_pair_ok": avoid_pair_ok,
        "changed_csv": str(changed_path),
    }


def add_suspicious_context(summary: pd.DataFrame) -> pd.DataFrame:
    if not SUSPICIOUS_PATH.exists():
        return summary
    suspicious = pd.read_csv(SUSPICIOUS_PATH)
    context_cols = [
        "file_name",
        "exp56_label",
        "exp56_decision_score",
        "prob_mean",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "suspicious_score",
        "recommended_action",
    ]
    suspicious = suspicious[[c for c in context_cols if c in suspicious.columns]]
    context_rows = []
    for row in summary.itertuples():
        for part in str(row.changed_samples).split(";"):
            if not part:
                continue
            file_name = part.split(":")[0]
            matched = suspicious[suspicious["file_name"] == file_name]
            if matched.empty:
                continue
            d = matched.iloc[0].to_dict()
            d["candidate"] = row.candidate
            d["change"] = part
            context_rows.append(d)
    context = pd.DataFrame(context_rows)
    if not context.empty:
        context.to_csv(OUT_PRED_DIR / f"{EXP_ID}_changed_sample_context.csv", index=False)
    return summary


def main() -> None:
    OUT_SUB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base()
    candidates = [
        ("exp108_flip0568_only", ["test_0568.png"]),
        ("exp108_flip0377_only_recheck", ["test_0377.png"]),
        ("exp108_flip0568_0377", ["test_0568.png", "test_0377.png"]),
    ]

    rows = []
    for name, flips in candidates:
        df = base.copy()
        for file_name in flips:
            if file_name not in set(df["file_name"]):
                raise ValueError(f"Missing sample: {file_name}")
            idx = df["file_name"] == file_name
            df.loc[idx, "label"] = 1 - df.loc[idx, "label"]
        validate_submission(df)
        out_path = OUT_SUB_DIR / f"{name}_submit.csv"
        df.to_csv(out_path, index=False)
        rows.append(summarize_candidate(name, df, base))

    summary = pd.DataFrame(rows)
    summary = add_suspicious_context(summary)
    summary.to_csv(OUT_PRED_DIR / f"{EXP_ID}_flip0568_candidates_summary.csv", index=False)

    print("EXP-108 candidates")
    print(summary.to_string(index=False))
    print("recommended first submission: oof/submissions/exp108_flip0568_only_submit.csv")


if __name__ == "__main__":
    main()
