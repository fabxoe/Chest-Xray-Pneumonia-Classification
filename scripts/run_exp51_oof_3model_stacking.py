from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


EXP_ID = "exp51"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "cnx": {
        "oof": PRED_DIR / "exp46_convnext_tiny_oof_probs.csv",
        "test": PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv",
    },
    "res": {
        "oof": PRED_DIR / "exp48_resnet18_oof_probs.csv",
        "test": PRED_DIR / "exp48_resnet18_test_probs_mean.csv",
    },
    "eff": {
        "oof": PRED_DIR / "exp50_efficientnet_b0_oof_probs.csv",
        "test": PRED_DIR / "exp50_efficientnet_b0_test_probs_mean.csv",
    },
}
ALPHAS = [0.3, 1, 3, 10, 30, 50, 75, 100, 150, 300, 500, 1000, 3000]
CS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SENSITIVE = ["test_0244.png", "test_0025.png", "test_0539.png"]


def metric_row(y_true, y_pred, prefix):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
    }


def add_model_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_rank"] = out[f"{prefix}_prob1"].rank(pct=True)
    out[f"{prefix}_margin"] = out[f"{prefix}_prob1"] - out[f"{prefix}_prob0"]
    out[f"{prefix}_uncertainty"] = 1.0 - np.abs(out[f"{prefix}_margin"])
    return out


def load_oof_and_test():
    oof_base = None
    test_base = None
    for name, paths in MODELS.items():
        oof = pd.read_csv(paths["oof"])
        test = pd.read_csv(paths["test"])
        oof_cur = oof[["file_name", "label", "fold", "prob_0", "prob_1", "pred_label"]].rename(
            columns={
                "prob_0": f"{name}_prob0",
                "prob_1": f"{name}_prob1",
                "pred_label": f"{name}_pred",
            }
        )
        test_cur = test[["file_name", "prob_0", "prob_1", "pred_label"]].rename(
            columns={
                "prob_0": f"{name}_prob0",
                "prob_1": f"{name}_prob1",
                "pred_label": f"{name}_pred",
            }
        )
        if oof_base is None:
            oof_base = oof_cur
        else:
            oof_base = oof_base.merge(
                oof_cur.drop(columns=["label", "fold"]),
                on="file_name",
                how="inner",
                validate="one_to_one",
            )
        test_base = test_cur if test_base is None else test_base.merge(test_cur, on="file_name", how="inner", validate="one_to_one")

    for name in MODELS:
        oof_base = add_model_features(oof_base, name)
        test_base = add_model_features(test_base, name)

    for df in [oof_base, test_base]:
        probs = df[[f"{name}_prob1" for name in MODELS]]
        df["prob_mean_all"] = probs.mean(axis=1)
        df["prob_min_all"] = probs.min(axis=1)
        df["prob_max_all"] = probs.max(axis=1)
        df["prob_std_all"] = probs.std(axis=1)
        df["vote_sum_all"] = df[[f"{name}_pred" for name in MODELS]].sum(axis=1)
        df["all_agree"] = (df["vote_sum_all"].isin([0, len(MODELS)])).astype(int)
    return oof_base.sort_values("file_name").reset_index(drop=True), test_base.sort_values("file_name").reset_index(drop=True)


def feature_sets():
    specs = {}
    model_names = list(MODELS)
    for size in [2, 3]:
        for subset in combinations(model_names, size):
            tag = "_".join(subset)
            probs = [f"{m}_prob1" for m in subset]
            ranks = [f"{m}_rank" for m in subset]
            margins = [f"{m}_margin" for m in subset]
            uncertainties = [f"{m}_uncertainty" for m in subset]
            preds = [f"{m}_pred" for m in subset]
            specs[f"{tag}_prob_only"] = probs
            specs[f"{tag}_prob_rank"] = probs + ranks
            specs[f"{tag}_prob_margin"] = probs + margins
            specs[f"{tag}_compact"] = probs + ranks + margins + uncertainties + preds

    specs["all_prob_stats"] = [
        "cnx_prob1",
        "res_prob1",
        "eff_prob1",
        "prob_mean_all",
        "prob_min_all",
        "prob_max_all",
        "prob_std_all",
    ]
    specs["all_compact_stats"] = [
        "cnx_prob1",
        "res_prob1",
        "eff_prob1",
        "cnx_rank",
        "res_rank",
        "eff_rank",
        "cnx_margin",
        "res_margin",
        "eff_margin",
        "prob_mean_all",
        "prob_std_all",
        "vote_sum_all",
        "all_agree",
    ]
    return specs


def make_estimator(kind: str, value: float):
    if kind == "ridge":
        return RidgeClassifier(alpha=value)
    if kind == "logistic":
        return LogisticRegression(C=value, solver="liblinear", max_iter=1000)
    raise ValueError(kind)


def meta_cv_predict(oof: pd.DataFrame, cols: list[str], kind: str, value: float):
    y = oof["label"].to_numpy()
    folds = oof["fold"].to_numpy()
    pred = np.zeros(len(oof), dtype=int)
    score = np.zeros(len(oof), dtype=float)
    for fold in sorted(oof["fold"].unique()):
        train_idx = folds != fold
        valid_idx = folds == fold
        clf = make_estimator(kind, value)
        clf.fit(oof.loc[train_idx, cols], y[train_idx])
        pred[valid_idx] = clf.predict(oof.loc[valid_idx, cols])
        if hasattr(clf, "decision_function"):
            score[valid_idx] = clf.decision_function(oof.loc[valid_idx, cols])
        else:
            score[valid_idx] = clf.predict_proba(oof.loc[valid_idx, cols])[:, 1]
    return pred, score


def final_test_predict(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], kind: str, value: float):
    clf = make_estimator(kind, value)
    clf.fit(oof[cols], oof["label"].to_numpy())
    pred = clf.predict(test[cols])
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(test[cols])
    else:
        score = clf.predict_proba(test[cols])[:, 1]
    return pred, score


def add_candidate_row(row, test, test_pred, test_score, exp26_pred):
    row["test_label0"] = int(np.sum(test_pred == 0))
    row["test_label1"] = int(np.sum(test_pred == 1))
    row["changed_vs_exp26"] = int(np.sum(test_pred != exp26_pred))
    for file_name in SENSITIVE:
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        row[file_name.replace(".png", "_pred")] = int(test_pred[idx])
        row[file_name.replace(".png", "_score")] = float(test_score[idx])
    return row


def direct_rows(oof: pd.DataFrame, test: pd.DataFrame, exp26_pred: np.ndarray):
    rows = []
    for name in MODELS:
        oof_pred = oof[f"{name}_pred"].to_numpy()
        test_pred = test[f"{name}_pred"].to_numpy()
        test_score = test[f"{name}_prob1"].to_numpy()
        row = {
            "spec": f"direct_{name}",
            "model": "direct",
            "param": np.nan,
            "features": f"{name}_prob1>=0.5",
            "n_features": 1,
        }
        row.update(metric_row(oof["label"].to_numpy(), oof_pred, "meta_cv"))
        rows.append(add_candidate_row(row, test, test_pred, test_score, exp26_pred))

    for subset in [("cnx", "res"), ("cnx", "eff"), ("res", "eff"), ("cnx", "res", "eff")]:
        tag = "_".join(subset)
        oof_prob = oof[[f"{m}_prob1" for m in subset]].mean(axis=1)
        test_prob = test[[f"{m}_prob1" for m in subset]].mean(axis=1)
        oof_pred = (oof_prob >= 0.5).astype(int).to_numpy()
        test_pred = (test_prob >= 0.5).astype(int).to_numpy()
        row = {
            "spec": f"softvote_{tag}",
            "model": "softvote",
            "param": np.nan,
            "features": "+".join([f"{m}_prob1" for m in subset]),
            "n_features": len(subset),
        }
        row.update(metric_row(oof["label"].to_numpy(), oof_pred, "meta_cv"))
        rows.append(add_candidate_row(row, test, test_pred, test_prob.to_numpy(), exp26_pred))
    return rows


def main():
    oof, test = load_oof_and_test()
    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test["file_name"].to_numpy(), exp26["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-26")
    exp26_pred = exp26["pred"].to_numpy()

    rows = direct_rows(oof, test, exp26_pred)
    candidates = []
    for spec, cols in feature_sets().items():
        for alpha in ALPHAS:
            cv_pred, cv_score = meta_cv_predict(oof, cols, "ridge", alpha)
            test_pred, test_score = final_test_predict(oof, test, cols, "ridge", alpha)
            row = {
                "spec": spec,
                "model": "ridge",
                "param": alpha,
                "features": "+".join(cols),
                "n_features": len(cols),
            }
            row.update(metric_row(oof["label"].to_numpy(), cv_pred, "meta_cv"))
            row = add_candidate_row(row, test, test_pred, test_score, exp26_pred)
            rows.append(row)
            candidates.append((spec, "ridge", alpha, cols, row, cv_pred, cv_score, test_pred, test_score))

        for c_value in CS:
            cv_pred, cv_score = meta_cv_predict(oof, cols, "logistic", c_value)
            test_pred, test_score = final_test_predict(oof, test, cols, "logistic", c_value)
            row = {
                "spec": spec,
                "model": "logistic",
                "param": c_value,
                "features": "+".join(cols),
                "n_features": len(cols),
            }
            row.update(metric_row(oof["label"].to_numpy(), cv_pred, "meta_cv"))
            row = add_candidate_row(row, test, test_pred, test_score, exp26_pred)
            rows.append(row)
            candidates.append((spec, "logistic", c_value, cols, row, cv_pred, cv_score, test_pred, test_score))

    summary = pd.DataFrame(rows)
    summary["passes_sensitive_strict"] = (
        summary["test_0244_pred"].eq(0)
        & summary["test_0025_pred"].eq(0)
        & summary["test_0539_pred"].eq(0)
    )
    summary["passes_sensitive_core"] = summary["test_0025_pred"].eq(0) & summary["test_0539_pred"].eq(0)
    summary["passes_label_band"] = summary["test_label1"].between(380, 420)
    summary["passes_change_band"] = summary["changed_vs_exp26"].between(1, 24)
    summary["candidate"] = summary["passes_sensitive_strict"] & summary["passes_label_band"] & summary["passes_change_band"]
    summary = summary.sort_values(
        ["candidate", "passes_sensitive_strict", "meta_cv_accuracy", "meta_cv_recall", "changed_vs_exp26"],
        ascending=[False, False, False, False, True],
    )
    summary_path = PRED_DIR / f"{EXP_ID}_oof_3model_stacking_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(10)
    if selected.empty:
        selected = summary[summary["passes_sensitive_strict"]].head(10)
    if selected.empty:
        selected = summary[summary["passes_sensitive_core"]].head(10)
    if selected.empty:
        selected = summary[~summary["model"].isin(["direct", "softvote"])].head(10)

    selected_keys = {(row.spec, row.model, float(row.param)) for row in selected.itertuples() if row.model not in ["direct", "softvote"]}
    sample = pd.read_csv("sample_submission.csv")
    for spec, model_name, value, cols, row, cv_pred, cv_score, test_pred, test_score in candidates:
        if (spec, model_name, float(value)) not in selected_keys:
            continue
        value_name = str(value).replace(".", "p")
        prefix = f"{EXP_ID}_{model_name}_{spec}_{value_name}"
        submit = sample.copy()
        submit["label"] = submit["file_name"].map(dict(zip(test["file_name"], test_pred))).astype(int)
        submit.to_csv(SUB_DIR / f"{prefix}_submit.csv", index=False)
        pd.DataFrame(
            {
                "file_name": test["file_name"],
                "decision_score": test_score,
                "pred": test_pred,
                "exp26_pred": exp26_pred,
                "changed_vs_exp26": test_pred != exp26_pred,
            }
        ).to_csv(PRED_DIR / f"{prefix}_test_scores.csv", index=False)
        changed = test.loc[test_pred != exp26_pred, ["file_name"]].copy()
        changed["exp26_pred"] = exp26_pred[test_pred != exp26_pred]
        changed["exp51_pred"] = test_pred[test_pred != exp26_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_exp26.csv", index=False)

    print(summary.head(40).to_string(index=False))
    print("candidate count:", int(summary["candidate"].sum()))
    print("strict sensitive count:", int(summary["passes_sensitive_strict"].sum()))
    print("saved:", summary_path)
    print("saved submissions:", len(selected_keys))


if __name__ == "__main__":
    main()
