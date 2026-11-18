"""Cached data and the banner every page shows."""
import json

import pandas as pd
import streamlit as st

from . import config, data, stats

NAVY = "#0F1F3D"
BLUE = "#2A78D6"


@st.cache_data
def employees():
    df = data.load()
    return df, data.labelled(df)


@st.cache_data
def tests():
    df, _ = employees()
    return stats.univariate_tests(df), stats.adjusted_odds_ratios(df)


@st.cache_data
def models():
    """The model results `python -m peoplepulse` wrote to results/; training on the free host stalls the page."""
    out = config.RESULTS_DIR
    s = json.loads((out / "summary.json").read_text())
    return {"metrics": pd.read_csv(out / "model_metrics.csv"), "shap": pd.read_csv(out / "shap_importance.csv"),
            "audit": pd.read_csv(out / "group_audit.csv"), "best_model": s["best_model_by_cv"],
            "holdout_size": s["holdout_size"], "holdout_leavers": s["holdout_leavers"],
            "pr_auc_baseline": s["pr_auc_random_baseline"]}


def header(title: str) -> None:
    st.set_page_config(page_title=f"PeoplePulse | {title}", layout="wide")
    st.title(title)
    st.info("Findings are associations in fictional data, not causes. This app shows group patterns only and "
            "never scores individual employees.", icon="ℹ️")
