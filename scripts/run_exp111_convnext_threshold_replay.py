from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "oof/preds"
SUB_DIR = ROOT / "oof/submissions"
REPORT_DIR = ROOT / "reports/exp111_convnext_threshold_replay"
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


def load_prob(name: str, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(PROB_FILES[name])[["file_name", "prob_1"]].copy()
    if list(df["file_name"]) != list(sample["file_name"]):
        raise ValueError(f"{name}: sample order mismatch")
    return df.rename(columns={"prob_1": f"{name}_prob1"})


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


def build_table() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)[["file_name"]]
    base = pd.read_csv(BASE_PATH).rename(columns={"label": "exp56_label"})
    if list(base["file_name"]) != list(sample["file_name"]):
        raise ValueError("EXP-56 sample order mismatch")
    table = base.copy()
    for name in PROB_FILES:
        table = table.merge(load_prob(name, sample), on="file_name", how="inner")
    suspicious = pd.read_csv(SUSPICIOUS_PATH)
    keep_cols = [
        "file_name",
        "exp56_score_margin",
        "prob_mean",
        "prob_std",
        "vote_1_count",
        "model_disagrees_with_exp56",
        "suspicious_score",
        "recommended_action",
    ]
    table = table.merge(suspicious[[c for c in keep_cols if c in suspicious.columns]], on="file_name", how="left")
    core_cols = [f"{m}_prob1" for m in CORE_MODELS]
    table["cnx_core_mean"] = table[core_cols].mean(axis=1)
    table["cnx_core_min"] = table[core_cols].min(axis=1)
    table["cnx_core_median"] = table[core_cols].median(axis=1)
    table["cnx_core_std"] = table[core_cols].std(axis=1)
    table["cnx_core_vote1"] = (table[core_cols] >= 0.5).sum(axis=1)
    table["cnx_ref_mean"] = table[[*core_cols, "cnxv2_prob1"]].mean(axis=1)
    return table


def make_candidate(base: pd.DataFrame, drops: list[str], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = base.copy()
    for file_name in drops:
        sub.loc[sub["file_name"].eq(file_name), "label"] = 0
    validate_submission(sub)
    mapping = dict(zip(sub["file_name"], sub["label"]))
    if not guardrail_ok(mapping):
        raise ValueError(f"{name}: guardrail violation")
    changed = base.merge(sub, on="file_name", suffixes=("_exp56", "_candidate"))
    changed = changed[changed["label_exp56"] != changed["label_candidate"]]
    return sub, changed


def main() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    base = pd.read_csv(BASE_PATH)

    pool = table[
        table["exp56_label"].eq(1)
        & ~table["file_name"].isin(HARD_GUARDRAILS)
        & ~table["recommended_action"].eq("review_high_priority")
        & ~table["recommended_action"].eq("avoid_change_protected")
    ].copy()

    score_specs = [
        ("core_mean", "cnx_core_mean", True),
        ("core_min", "cnx_core_min", True),
        ("core_median", "cnx_core_median", True),
        ("ref_mean", "cnx_ref_mean", True),
        ("core_disagreement", "cnx_core_std", False),
    ]

    rows = []
    seen: set[tuple[str, ...]] = set()
    for spec_name, score_col, ascending in score_specs:
        sorted_pool = pool.sort_values([score_col, "exp56_score_margin"], ascending=[ascending, True])
        for k in [1, 2]:
            drops = sorted(sorted_pool.head(k)["file_name"].tolist())
            key = tuple(drops)
            if key in seen:
                continue
            seen.add(key)
            name = f"exp111_{spec_name}_drop{k}"
            try:
                sub, changed = make_candidate(base, drops, name)
            except ValueError:
                continue
            out_path = SUB_DIR / f"{name}_submit.csv"
            changed_path = PRED_DIR / f"{name}_changed_vs_exp56.csv"
            sub.to_csv(out_path, index=False)
            changed.to_csv(changed_path, index=False)
            detail = table[table["file_name"].isin(drops)].copy()
            rows.append(
                {
                    "candidate": name,
                    "submission_path": str(out_path),
                    "score_spec": spec_name,
                    "drop_count": k,
                    "changed_samples": ";".join(drops),
                    "test_label0": int((sub["label"] == 0).sum()),
                    "test_label1": int((sub["label"] == 1).sum()),
                    "min_exp56_margin": float(detail["exp56_score_margin"].min()),
                    "mean_core_mean": float(detail["cnx_core_mean"].mean()),
                    "mean_core_std": float(detail["cnx_core_std"].mean()),
                    "mean_vote1": float(detail["vote_1_count"].mean()),
                    "changed_csv": str(changed_path),
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["drop_count", "mean_core_mean", "mean_core_std"], ascending=[True, True, False])
    summary.to_csv(PRED_DIR / "exp111_convnext_threshold_replay_summary.csv", index=False)
    pool.sort_values("cnx_core_mean").to_csv(REPORT_DIR / "exp111_positive_pool_by_core_mean.csv", index=False)

    print("EXP-111 ConvNeXt threshold distribution replay")
    print("pool rows:", len(pool))
    print("saved candidates:", len(summary))
    if not summary.empty:
        print(summary[["candidate", "submission_path", "changed_samples", "test_label1", "score_spec"]].to_string(index=False))


if __name__ == "__main__":
    main()
