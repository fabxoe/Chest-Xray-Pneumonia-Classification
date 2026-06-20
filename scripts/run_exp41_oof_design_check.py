from pathlib import Path

import pandas as pd


FOLD_PATH = Path("oof/folds_seed42.csv")
SUMMARY_PATH = Path("oof/folds_seed42_summary.csv")


def main():
    folds = pd.read_csv(FOLD_PATH)
    assert len(folds) == 5216, f"unexpected train rows: {len(folds)}"
    assert folds["file_name"].is_unique, "file_name must be unique"
    assert set(folds["fold"].unique()) == {0, 1, 2, 3, 4}, folds["fold"].unique()
    assert set(folds["label"].unique()) == {0, 1}, folds["label"].unique()

    summary = folds.groupby(["fold", "label"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary["label0_ratio"] = summary[0] / summary["total"]
    summary["label1_ratio"] = summary[1] / summary["total"]
    summary.to_csv(SUMMARY_PATH)

    print("OOF fold design check passed")
    print(summary)
    print("saved:", FOLD_PATH)
    print("saved:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
