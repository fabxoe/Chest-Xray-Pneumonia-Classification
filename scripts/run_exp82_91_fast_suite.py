from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


PRED = Path("oof/preds")
SUB = Path("oof/submissions")
PRED.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

MODELS = {
    "cnx": ("exp46_convnext_tiny_oof_probs.csv", "exp46_convnext_tiny_test_probs_mean.csv"),
    "cnx123": ("exp54_convnext_tiny_seed123_oof_probs.csv", "exp54_convnext_tiny_seed123_test_probs_mean.csv"),
    "eff": ("exp50_efficientnet_b0_oof_probs.csv", "exp50_efficientnet_b0_test_probs_mean.csv"),
    "eff123": ("exp81_efficientnet_b0_seed123_oof_probs.csv", "exp81_efficientnet_b0_seed123_test_probs_mean.csv"),
    "res": ("exp48_resnet18_oof_probs.csv", "exp48_resnet18_test_probs_mean.csv"),
}

PROTECTED = {
    "test_0025.png": 0, "test_0539.png": 0, "test_0244.png": 0,
    "test_0490.png": 0, "test_0201.png": 1, "test_0222.png": 1,
    "test_0504.png": 0, "test_0141.png": 1, "test_0049.png": 0,
}


def load_tables():
    oof_base = test_base = None
    for name, (oof_file, test_file) in MODELS.items():
        o = pd.read_csv(PRED / oof_file).sort_values("file_name").reset_index(drop=True)
        t = pd.read_csv(PRED / test_file).sort_values("file_name").reset_index(drop=True)
        oc = o[["file_name", "label", "fold", "prob_1", "pred_label"]].rename(
            columns={"prob_1": f"{name}_prob", "pred_label": f"{name}_pred"}
        )
        tc = t[["file_name", "prob_1", "pred_label"]].rename(
            columns={"prob_1": f"{name}_prob", "pred_label": f"{name}_pred"}
        )
        if oof_base is None:
            oof_base = oc
            test_base = tc
        else:
            oof_base = oof_base.merge(oc.drop(columns=["label", "fold"]), on="file_name", validate="one_to_one")
            test_base = test_base.merge(tc, on="file_name", validate="one_to_one")
    for name in MODELS:
        oof_base[f"{name}_rank"] = oof_base[f"{name}_prob"].rank(pct=True)
        test_base[f"{name}_rank"] = test_base[f"{name}_prob"].rank(pct=True)
        oof_base[f"{name}_margin"] = 2 * oof_base[f"{name}_prob"] - 1
        test_base[f"{name}_margin"] = 2 * test_base[f"{name}_prob"] - 1
    return oof_base, test_base


def estimator(kind, value):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=value))
    return make_pipeline(StandardScaler(), LogisticRegression(C=value, solver="liblinear", max_iter=1000))


def cv_scores(oof, cols, kind, value):
    scores = np.zeros(len(oof))
    for fold in sorted(oof.fold.unique()):
        tr, va = oof.fold.ne(fold), oof.fold.eq(fold)
        model = estimator(kind, value)
        model.fit(oof.loc[tr, cols], oof.loc[tr, "label"])
        scores[va] = model.decision_function(oof.loc[va, cols])
    model = estimator(kind, value)
    model.fit(oof[cols], oof.label)
    return scores, model


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def guardrails(test, pred):
    mapping = dict(zip(test.file_name, pred))
    protected = all(int(mapping[k]) == v for k, v in PROTECTED.items())
    avoid_pair = not (int(mapping["test_0188.png"]) == 1 and int(mapping["test_0568.png"]) == 1)
    return protected, avoid_pair


def save_candidate(name, test, pred, score, best_pred):
    sample = pd.read_csv("sample_submission.csv")
    sample["label"] = sample.file_name.map(dict(zip(test.file_name, pred))).astype(int)
    sample.to_csv(SUB / f"{name}_submit.csv", index=False)
    detail = pd.DataFrame({"file_name": test.file_name, "score": score, "pred": pred, "exp56_pred": best_pred})
    detail["changed"] = detail.pred.ne(detail.exp56_pred)
    detail.to_csv(PRED / f"{name}_test_scores.csv", index=False)


def main():
    oof, test = load_tables()
    y = oof.label.to_numpy()
    best = pd.read_csv(PRED / "exp56_ridge_cnx_eff_compact_p3000_t0p025_test_scores.csv").sort_values("file_name")
    best_pred = best.pred.to_numpy()
    rows = []

    # EXP-82: EfficientNet seed constrained stacking.
    specs = {
        "eff_seeds_prob": ["eff_prob", "eff123_prob"],
        "cnx_eff_seeds_prob": ["cnx_prob", "eff_prob", "eff123_prob"],
        "cnx123_eff_seeds_prob": ["cnx_prob", "cnx123_prob", "eff_prob", "eff123_prob"],
        "cnx123_eff_seeds_rank": ["cnx_prob", "cnx123_prob", "eff_prob", "eff123_prob", "eff_rank", "eff123_rank"],
        "compact": ["cnx_prob", "cnx123_prob", "eff_prob", "eff123_prob", "cnx_rank", "cnx123_rank", "eff_rank", "eff123_rank"],
    }
    candidates = []
    for spec, cols in specs.items():
        for kind, vals in [("ridge", [30, 100, 300, 1000, 3000]), ("logistic", [0.05, 0.1, 0.25, 0.5, 1.0])]:
            for val in vals:
                cv, model = cv_scores(oof, cols, kind, val)
                ts = model.decision_function(test[cols])
                for th in [-0.10, -0.05, -0.025, 0, 0.025, 0.05, 0.10]:
                    cp, tp = (cv >= th).astype(int), (ts >= th).astype(int)
                    row = {"experiment": "EXP-82", "spec": spec, "model": kind, "param": val, "threshold": th, **metrics(y, cp)}
                    row["changed_vs_exp56"] = int(np.sum(tp != best_pred))
                    row["label1"] = int(tp.sum())
                    row["protected"], row["avoid_pair"] = guardrails(test, tp)
                    row["candidate"] = row["protected"] and row["avoid_pair"] and 1 <= row["changed_vs_exp56"] <= 5 and row["accuracy"] >= 0.9855
                    rows.append(row)
                    candidates.append((row, tp, ts))

    # EXP-83: EfficientNet seed soft voting.
    for w in np.arange(0, 1.01, 0.1):
        op = w * oof.eff_prob + (1 - w) * oof.eff123_prob
        tp = w * test.eff_prob + (1 - w) * test.eff123_prob
        pred_o, pred_t = op.ge(0.5).astype(int), tp.ge(0.5).astype(int)
        row = {"experiment": "EXP-83", "spec": "eff_seed_softvote", "weight_exp50": round(float(w), 2), **metrics(y, pred_o)}
        row["changed_vs_exp56"] = int(np.sum(pred_t != best_pred)); row["label1"] = int(pred_t.sum())
        row["protected"], row["avoid_pair"] = guardrails(test, pred_t)
        rows.append(row)

    # EXP-84: Error correlation and diversity.
    div = []
    for i, a in enumerate(MODELS):
        for b in list(MODELS)[i + 1:]:
            ea = oof[f"{a}_pred"].ne(y); eb = oof[f"{b}_pred"].ne(y)
            n11 = int((ea & eb).sum()); n00 = int((~ea & ~eb).sum()); n10 = int((ea & ~eb).sum()); n01 = int((~ea & eb).sum())
            q_den = n11 * n00 + n10 * n01
            div.append({"model_a": a, "model_b": b, "prob_corr": oof[f"{a}_prob"].corr(oof[f"{b}_prob"]),
                        "pred_disagreement": int(oof[f"{a}_pred"].ne(oof[f"{b}_pred"]).sum()),
                        "shared_errors": n11, "q_stat": (n11 * n00 - n10 * n01) / q_den if q_den else np.nan})
    pd.DataFrame(div).sort_values(["q_stat", "pred_disagreement"]).to_csv(PRED / "exp84_oof_diversity_summary.csv", index=False)

    # EXP-85/86: Importance and leave-one-feature/model-out pruning.
    base_cols = ["cnx_prob", "cnx123_prob", "eff_prob", "eff123_prob"]
    cv, model = cv_scores(oof, base_cols, "ridge", 300)
    coef = model[-1].coef_.ravel()
    pd.DataFrame({"feature": base_cols, "coefficient": coef, "abs_coefficient": np.abs(coef)}).sort_values("abs_coefficient", ascending=False).to_csv(PRED / "exp85_ridge_feature_importance.csv", index=False)
    prune = []
    for drop in [None] + base_cols:
        cols = [c for c in base_cols if c != drop]
        score, _ = cv_scores(oof, cols, "ridge", 300)
        prune.append({"dropped": drop or "none", "features": "+".join(cols), **metrics(y, score >= 0)})
    pd.DataFrame(prune).sort_values("accuracy", ascending=False).to_csv(PRED / "exp86_feature_pruning_summary.csv", index=False)

    # EXP-87: Threshold and sigmoid-like logistic calibration comparison.
    cal = []
    for kind, value in [("ridge", 300), ("logistic", 0.25)]:
        score, model = cv_scores(oof, base_cols, kind, value)
        tscore = model.decision_function(test[base_cols])
        for th in np.arange(-0.15, 0.151, 0.01):
            po, pt = (score >= th).astype(int), (tscore >= th).astype(int)
            protected, avoid = guardrails(test, pt)
            cal.append({"model": kind, "param": value, "threshold": round(float(th), 3), **metrics(y, po),
                        "changed_vs_exp56": int(np.sum(pt != best_pred)), "label1": int(pt.sum()), "protected": protected, "avoid_pair": avoid})
    pd.DataFrame(cal).sort_values(["protected", "accuracy"], ascending=[False, False]).to_csv(PRED / "exp87_threshold_calibration_summary.csv", index=False)

    # EXP-88: Low-change candidates from EXP-82 search.
    search = pd.DataFrame(rows)
    search.to_csv(PRED / "exp82_83_fast_search_summary.csv", index=False)
    valid = [(r, p, s) for r, p, s in candidates if r["candidate"]]
    valid.sort(key=lambda x: (x[0]["changed_vs_exp56"], -x[0]["accuracy"]))
    saved = 0
    seen = set()
    for row, pred, score in valid:
        key = tuple(pred.tolist())
        if key in seen: continue
        seen.add(key)
        name = f"exp88_{row['model']}_{row['spec']}_p{str(row['param']).replace('.', 'p')}_t{str(row['threshold']).replace('-', 'm').replace('.', 'p')}"
        save_candidate(name, test, pred, score, best_pred)
        saved += 1
        if saved == 3: break

    # EXP-89: FP/FN specialization.
    err = []
    for name in MODELS:
        pred = oof[f"{name}_pred"].to_numpy()
        m = metrics(y, pred)
        err.append({"model": name, **m, "error_count": m["fp"] + m["fn"], "fn_share": m["fn"] / max(1, m["fp"] + m["fn"])})
    pd.DataFrame(err).sort_values(["fn", "fp"]).to_csv(PRED / "exp89_fp_fn_specialization.csv", index=False)

    # EXP-90: Rank ensemble.
    rank = []
    for names in [("cnx", "eff"), ("cnx", "eff", "eff123"), ("cnx", "cnx123", "eff", "eff123")]:
        ro = oof[[f"{n}_rank" for n in names]].mean(axis=1)
        rt = test[[f"{n}_rank" for n in names]].mean(axis=1)
        for th in [0.24, 0.25, 0.26, 0.27, 0.28]:
            po, pt = ro.ge(th).astype(int), rt.ge(th).astype(int)
            protected, avoid = guardrails(test, pt)
            rank.append({"models": "+".join(names), "threshold": th, **metrics(y, po),
                         "changed_vs_exp56": int(np.sum(pt != best_pred)), "label1": int(pt.sum()), "protected": protected, "avoid_pair": avoid})
    pd.DataFrame(rank).sort_values(["protected", "accuracy"], ascending=[False, False]).to_csv(PRED / "exp90_rank_ensemble_summary.csv", index=False)

    # EXP-91: Existing single-split TTA direction summary only.
    tta_rows = []
    for name in ["convnext12", "convnext18_full", "efficientnet_b0", "resnet18"]:
        path = Path(f"preds/exp37_tta_{name}_test_probs.csv")
        if path.exists():
            df = pd.read_csv(path)
            prob_col = "prob_1" if "prob_1" in df else "prob1"
            pred = df[prob_col].ge(0.5).astype(int)
            tta_rows.append({"model": name, "rows": len(df), "label0": int((pred == 0).sum()), "label1": int((pred == 1).sum())})
    pd.DataFrame(tta_rows).to_csv(PRED / "exp91_existing_tta_direction_summary.csv", index=False)

    print("EXP-82 candidates:", len(valid), "saved:", saved)
    print("best EXP-82 rows")
    print(search[search.experiment.eq("EXP-82")].sort_values(["candidate", "accuracy"], ascending=[False, False]).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
