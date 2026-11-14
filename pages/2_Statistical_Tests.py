import plotly.graph_objects as go
import streamlit as st

from peoplepulse import ui

ui.header("Statistical tests")
uni, odds = ui.tests()
FORMATS = {c: st.column_config.NumberColumn(format="%.1e") for c in ["p_value", "p_adjusted"]} | {
    c: st.column_config.NumberColumn(format="%.3f") for c in ["effect_size", "odds_ratio", "ci_low", "ci_high"]}

st.subheader("One variable at a time")
st.markdown("Chi-square tests for categories, Mann-Whitney U for numbers. p-values are adjusted with "
            "Benjamini-Hochberg because 30 variables are tested at once. Sorted by effect size.")
st.dataframe(uni[["variable", "test", "effect_measure", "effect_size", "p_value", "p_adjusted",
                  "significant_fdr_5pct", "median_left", "median_stayed"]],
             hide_index=True, width="stretch", column_config=FORMATS)
st.caption("Rank-biserial r below 0: people who left tend to have lower values. Cramér's V: 0.1 small, 0.3 medium.")

st.subheader("Adjusted for each other (logistic regression)")
st.markdown(f"Odds ratios for work-condition variables, each holding the others fixed "
            f"(n = {odds.attrs['n']:,}, McFadden pseudo-R² = {odds.attrs['pseudo_r2']:.2f}). "
            "Above 1: higher odds of leaving. Age, gender and marital status are deliberately left out.")
o = odds.sort_values("odds_ratio")
fig = go.Figure(go.Scatter(
    x=o["odds_ratio"], y=o["term"], mode="markers", marker=dict(size=9, color=ui.BLUE),
    error_x=dict(type="data", symmetric=False, array=o["ci_high"] - o["odds_ratio"],
                 arrayminus=o["odds_ratio"] - o["ci_low"], color="#888"),
    hovertemplate="%{y}<br>odds ratio %{x:.2f}<extra></extra>"))
fig.add_vline(x=1, line_dash="dot", line_color=ui.NAVY)
fig.update_layout(xaxis_type="log", xaxis_title="Odds ratio (log scale, 95% CI)", height=520,
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.warning("These are associations in a cross-sectional, fictional dataset. An odds ratio of 5 for overtime does "
           "not mean removing overtime would cut attrition fivefold.")
with st.expander("Table view"):
    st.dataframe(odds, hide_index=True, width="stretch", column_config=FORMATS)
