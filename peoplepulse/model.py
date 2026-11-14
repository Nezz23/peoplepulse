"""Attrition models for *pattern discovery*, never for scoring people.

Everything returned here is aggregate: metrics, global SHAP importance, group-level calibration.
No function returns or stores a per-employee probability.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config

# Protected attributes are excluded from the features; they are only used afterwards to audit the model.
PROTECTED = ["age", "gender", "marital_status"]
TEXT = ["business_travel", "department", "education_field", "job_role", "over_time"]


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["attrition", "employee_number", *PROTECTED]), df["attrition"]


def _prep(x: pd.DataFrame, scale: bool) -> ColumnTransformer:
    numeric = [c for c in x.columns if c not in TEXT]
    return ColumnTransformer([
        ("text", OneHotEncoder(handle_unknown="ignore", sparse_output=False), TEXT),
        ("num", StandardScaler() if scale else "passthrough", numeric)], verbose_feature_names_out=False)


def models(y: pd.Series) -> dict[str, Pipeline]:
    """Fixed, untuned settings: with 237 leavers, tuning on the same data would mostly fit noise."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    return {
        "Logistic regression": lambda x: Pipeline([("prep", _prep(x, True)), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=5000))]),
        "Random forest": lambda x: Pipeline([("prep", _prep(x, False)), ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=config.SEED,
            n_jobs=-1))]),
        "XGBoost": lambda x: Pipeline([("prep", _prep(x, False)), ("clf", XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=config.SEED, eval_metric="logloss"))]),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(config.SEED)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def shap_importance(pipe: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original variable (one-hot columns summed back), over a whole set: a global ranking only."""
    xt = pd.DataFrame(pipe["prep"].transform(x), columns=pipe["prep"].get_feature_names_out())
    values = shap.TreeExplainer(pipe["clf"]).shap_values(xt)
    per_col = pd.Series(np.abs(values).mean(axis=0), index=xt.columns)
    source = {c: next((t for t in TEXT if c.startswith(t + "_")), c) for c in xt.columns}
    out = per_col.groupby(source).sum().sort_values(ascending=False)
    # direction: correlation between a numeric variable's value and its SHAP value (text variables have none)
    direction = {c: float(np.corrcoef(xt[c], values[:, i])[0, 1]) for i, c in enumerate(xt.columns)
                 if c not in TEXT and source[c] == c and xt[c].std() > 0}
    return pd.DataFrame({"variable": out.index, "mean_abs_shap": out.values,
                         "value_vs_shap_corr": [direction.get(v, np.nan) for v in out.index]})


def group_audit(df_test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """ROC-AUC per protected group on the holdout: does the model rank leavers equally well across groups?
    (Mean predicted rates are not compared: class weighting deliberately inflates them.) Groups under MIN_GROUP dropped."""
    d = df_test.assign(p=p, age_band=pd.cut(df_test["age"], [0, 29, 39, 100], labels=["18-29", "30-39", "40+"]))
    rows = []
    for col in ["gender", "age_band", "marital_status"]:
        for g, part in d.groupby(col, observed=True):
            if len(part) < config.MIN_GROUP:
                continue
            both = part["attrition"].value_counts().reindex([0, 1], fill_value=0).min() >= 10  # AUC on <10 leavers is noise
            rows.append({"attribute": col, "group": str(g), "employees": len(part),
                         "leavers": int(part["attrition"].sum()), "actual_rate_pct": 100 * part["attrition"].mean(),
                         "roc_auc": roc_auc_score(part["attrition"], part["p"]) if both else np.nan})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict:
    x, y = features(df)
    x_tr, x_te, y_tr, y_te, df_tr, df_te = train_test_split(x, y, df, test_size=0.2, stratify=y,
                                                            random_state=config.SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    rows, fitted, preds = [], {}, {}
    for name, make in models(y_tr).items():
        cv_auc = cross_val_score(make(x_tr), x_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe = make(x_tr).fit(x_tr, y_tr)
        p = pipe.predict_proba(x_te)[:, 1]
        lo, hi = bootstrap_auc_ci(y_te.to_numpy(), p)
        rows.append({"model": name, "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_sd": cv_auc.std(),
                     "holdout_roc_auc": roc_auc_score(y_te, p), "holdout_auc_ci_low": lo, "holdout_auc_ci_high": hi,
                     "holdout_pr_auc": average_precision_score(y_te, p), "holdout_brier": brier_score_loss(y_te, p)})
        fitted[name], preds[name] = pipe, p
    metrics = pd.DataFrame(rows)
    # chosen on cross-validation, so the holdout stays an honest final check
    best = metrics.loc[metrics["cv_roc_auc_mean"].idxmax(), "model"]
    return {
        "metrics": metrics, "best_model": best,
        "holdout_size": len(y_te), "holdout_leavers": int(y_te.sum()),
        "pr_auc_baseline": float(y_te.mean()),               # PR-AUC of a model that guesses at random
        "shap": shap_importance(fitted["XGBoost"], x_te),
        "audit": group_audit(df_te, preds[best]),
    }
