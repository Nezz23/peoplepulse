"""Cached data and the banner every page shows."""
import streamlit as st

from . import data, model, stats

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


@st.cache_resource(show_spinner="Training three models (about 10 seconds, once)...")
def models():
    df, _ = employees()
    return model.run(df)


def header(title: str) -> None:
    st.set_page_config(page_title=f"PeoplePulse | {title}", layout="wide")
    st.title(title)
    st.caption(f"Data: IBM's fictional HR attrition sample (1,470 made-up employees) · Mode: {data.mode()}")
    st.info("Findings are associations in fictional data, not causes. This app shows group patterns only and "
            "never scores individual employees.", icon="ℹ️")
