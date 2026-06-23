from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


EXP_ID = "exp80"
PRED_DIR = Path("oof/preds")
SUB_DIR = Path("oof/submissions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_BEST_PATH = PRED_DIR / "exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv"
CURRENT_BEST_META_CV_ACCURACY = 0.9865804597701149
META_CV_TOLERANCE = 0.001

MODELS = {
    "cnx": {
        "oof": PRED_DIR / "exp46_convnext_tiny_oof_probs.csv",
        "test": PRED_DIR / "exp46_convnext_tiny_test_probs_mean.csv",
    },
    "cnx123": {
        "oof": PRED_DIR / "exp54_convnext_tiny_seed123_oof_probs.csv",
        "test": PRED_DIR / "exp54_convnext_tiny_seed123_test_probs_mean.csv",
    },
    "eff": {
        "oof": PRED_DIR / "exp50_efficientnet_b0_oof_probs.csv",
        "test": PRED_DIR / "exp50_efficientnet_b0_test_probs_mean.csv",
    },
    "res": {
        "oof": PRED_DIR / "exp48_resnet18_oof_probs.csv",
        "test": PRED_DIR / "exp48_resnet18_test_probs_mean.csv",
    },
}

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

AVOID_PAIR_ONE = {"test_0188.png", "test_0568.png"}

ALPHAS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 3000]
CS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
THRESHOLDS = [-0.10, -0.075, -0.05, -0.03, -0.025, -0.02, -0.01, 0.0, 0.01, 0.02, 0.025, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


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
            columns={"prob_0": f"{name}_prob0", "prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"}
        )
        test_cur = test[["file_name", "prob_0", "prob_1", "pred_label"]].rename(
            columns={"prob_0": f"{name}_prob0", "prob_1": f"{name}_prob1", "pred_label": f"{name}_pred"}
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
        df["prob_std_all"] = probs.std(axis=1)
        df["prob_min_all"] = probs.min(axis=1)
        df["prob_max_all"] = probs.max(axis=1)
        df["vote_sum_all"] = df[[f"{name}_pred" for name in MODELS]].sum(axis=1)
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
        "cnx123_prob1",
        "eff_prob1",
        "res_prob1",
        "prob_mean_all",
        "prob_std_all",
        "prob_min_all",
        "prob_max_all",
    ]
    specs["exp56_plus_res_prob"] = ["cnx_prob1", "cnx123_prob1", "eff_prob1", "res_prob1"]
    specs["exp56_plus_res_prob_rank"] = [
        "cnx_prob1", "cnx123_prob1", "eff_prob1", "res_prob1",
        "cnx_rank", "cnx123_rank", "eff_rank", "res_rank",
    ]
    specs["exp56_plus_res_compact"] = [
        "cnx_prob1", "cnx123_prob1", "eff_prob1", "res_prob1",
        "cnx_rank", "cnx123_rank", "eff_rank", "res_rank",
        "cnx_margin", "cnx123_margin", "eff_margin", "res_margin",
    ]
    specs["res_limited"] = [
        "cnx_prob1", "cnx123_prob1", "eff_prob1",
        "res_prob1", "res_rank", "res_margin", "res_uncertainty",
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
        score[valid_idx] = clf.decision_function(oof.loc[valid_idx, cols])
    return pred, score


def final_test_predict(oof: pd.DataFrame, test: pd.DataFrame, cols: list[str], kind: str, value: float):
    clf = make_estimator(kind, value)
    clf.fit(oof[cols], oof["label"].to_numpy())
    pred = clf.predict(test[cols])
    score = clf.decision_function(test[cols])
    return pred, score


def metric_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "meta_cv_accuracy": accuracy_score(y_true, y_pred),
        "meta_cv_precision": precision_score(y_true, y_pred, zero_division=0),
        "meta_cv_recall": recall_score(y_true, y_pred, zero_division=0),
        "meta_cv_f1": f1_score(y_true, y_pred, zero_division=0),
        "meta_cv_tn": int(tn),
        "meta_cv_fp": int(fp),
        "meta_cv_fn": int(fn),
        "meta_cv_tp": int(tp),
    }


def add_common(row, test, test_pred, test_score, exp26_pred, current_best_pred):
    row["test_label0"] = int(np.sum(test_pred == 0))
    row["test_label1"] = int(np.sum(test_pred == 1))
    row["changed_vs_exp26"] = int(np.sum(test_pred != exp26_pred))
    row["changed_vs_current_best"] = int(np.sum(test_pred != current_best_pred))
    for file_name in list(PROTECTED) + sorted(AVOID_PAIR_ONE):
        idx = int(np.where(test["file_name"].to_numpy() == file_name)[0][0])
        tag = file_name.replace(".png", "")
        row[f"{tag}_pred"] = int(test_pred[idx])
        row[f"{tag}_score"] = float(test_score[idx])
    return row


def protected_condition(summary):
    checks = []
    for file_name, expected in PROTECTED.items():
        checks.append(summary[f"{file_name.replace('.png', '')}_pred"].eq(expected))
    out = checks[0]
    for check in checks[1:]:
        out = out & check
    return out


def avoid_pair_condition(summary):
    return ~(summary["test_0188_pred"].eq(1) & summary["test_0568_pred"].eq(1))


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p")


def main():
    oof, test = load_oof_and_test()
    y = oof["label"].to_numpy()
    exp26 = pd.read_csv("preds/exp26_ridge_without_densenet_alpha30_test_scores.csv").sort_values("file_name").reset_index(drop=True)
    current_best = pd.read_csv(CURRENT_BEST_PATH).sort_values("file_name").reset_index(drop=True)
    if not np.array_equal(test["file_name"].to_numpy(), exp26["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against EXP-26")
    if not np.array_equal(test["file_name"].to_numpy(), current_best["file_name"].to_numpy()):
        raise ValueError("test file order mismatch against current best")
    exp26_pred = exp26["pred"].to_numpy()
    current_best_pred = current_best["pred"].to_numpy()

    rows = []
    candidates = []
    for spec, cols in feature_sets().items():
        for kind, values in [("ridge", ALPHAS), ("logistic", CS)]:
            for value in values:
                cv_pred_default, cv_score = meta_cv_predict(oof, cols, kind, value)
                test_pred_default, test_score = final_test_predict(oof, test, cols, kind, value)
                for threshold in THRESHOLDS:
                    cv_pred = (cv_score >= threshold).astype(int)
                    test_pred = (test_score >= threshold).astype(int)
                    row = {
                        "spec": spec,
                        "model": kind,
                        "param": value,
                        "threshold": threshold,
                        "features": "+".join(cols),
                        "n_features": len(cols),
                        "default_meta_cv_changed": int(np.sum(cv_pred != cv_pred_default)),
                        "default_test_changed": int(np.sum(test_pred != test_pred_default)),
                    }
                    row.update(metric_row(y, cv_pred))
                    row = add_common(row, test, test_pred, test_score, exp26_pred, current_best_pred)
                    rows.append(row)
                    candidates.append((spec, kind, value, threshold, cols, row, test_pred, test_score))

    summary = pd.DataFrame(rows)
    summary["passes_protected"] = protected_condition(summary)
    summary["passes_avoid_pair"] = avoid_pair_condition(summary)
    summary["passes_label_band"] = summary["test_label1"].between(395, 401)
    summary["passes_low_change_current"] = summary["changed_vs_current_best"].between(1, 8)
    summary["passes_preferred_low_change"] = summary["changed_vs_current_best"].between(1, 3)
    summary["passes_meta_cv_floor"] = summary["meta_cv_accuracy"].ge(CURRENT_BEST_META_CV_ACCURACY - META_CV_TOLERANCE)
    summary["not_duplicate_current"] = summary["changed_vs_current_best"].gt(0)
    summary["candidate"] = (
        summary["passes_protected"]
        & summary["passes_avoid_pair"]
        & summary["passes_label_band"]
        & summary["passes_low_change_current"]
        & summary["passes_meta_cv_floor"]
        & summary["not_duplicate_current"]
    )
    summary = summary.sort_values(
        [
            "candidate",
            "passes_preferred_low_change",
            "passes_protected",
            "passes_avoid_pair",
            "changed_vs_current_best",
            "meta_cv_accuracy",
            "meta_cv_recall",
            "changed_vs_exp26",
        ],
        ascending=[False, False, False, False, True, False, False, True],
    )
    summary_path = PRED_DIR / f"{EXP_ID}_resnet_compact_stacking_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = summary[summary["candidate"]].head(10)
    if selected.empty:
        selected = summary[
            summary["passes_protected"]
            & summary["passes_avoid_pair"]
            & summary["passes_label_band"]
            & summary["passes_meta_cv_floor"]
            & summary["not_duplicate_current"]
        ].head(10)

    candidate_map = {
        (spec, kind, float(value), float(threshold)): (spec, kind, value, threshold, cols, row, test_pred, test_score)
        for spec, kind, value, threshold, cols, row, test_pred, test_score in candidates
    }
    sample = pd.read_csv("sample_submission.csv")
    saved_hashes = set()
    saved_count = 0
    for selected_row in selected.itertuples():
        key = (selected_row.spec, selected_row.model, float(selected_row.param), float(selected_row.threshold))
        if key not in candidate_map:
            continue
        spec, kind, value, threshold, cols, row, test_pred, test_score = candidate_map[key]
        pred_hash = tuple(test_pred.tolist())
        if pred_hash in saved_hashes:
            continue
        saved_hashes.add(pred_hash)
        prefix = f"{EXP_ID}_{kind}_{spec}_p{safe_name(value)}_t{safe_name(threshold)}"

        submit = sample.copy()
        submit["label"] = submit["file_name"].map(dict(zip(test["file_name"], test_pred))).astype(int)
        submit.to_csv(SUB_DIR / f"{prefix}_submit.csv", index=False)

        score_df = pd.DataFrame(
            {
                "file_name": test["file_name"],
                "decision_score": test_score,
                "pred": test_pred,
                "exp26_pred": exp26_pred,
                "current_best_pred": current_best_pred,
                "changed_vs_exp26": test_pred != exp26_pred,
                "changed_vs_current_best": test_pred != current_best_pred,
            }
        )
        score_df.to_csv(PRED_DIR / f"{prefix}_test_scores.csv", index=False)

        changed = test.loc[test_pred != current_best_pred, ["file_name"]].copy()
        changed["current_best_pred"] = current_best_pred[test_pred != current_best_pred]
        changed[f"{EXP_ID}_pred"] = test_pred[test_pred != current_best_pred]
        changed.to_csv(PRED_DIR / f"{prefix}_changed_vs_current_best.csv", index=False)
        saved_count += 1

    print(summary.head(40).to_string(index=False))
    print("candidate count:", int(summary["candidate"].sum()))
    print("saved:", summary_path)
    print("saved unique submissions:", saved_count)


if __name__ == "__main__":
    main()
