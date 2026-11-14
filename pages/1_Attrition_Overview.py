import plotly.graph_objects as go
import streamlit as st

from peoplepulse import config, stats, ui

ui.header("Attrition overview")
df, lab = ui.employees()
overall = stats.rate(df)["rate_pct"]

DIMENSIONS = {"Department": "department", "Job role": "job_role", "Overtime": "over_time",
              "Business travel": "business_travel", "Job level": "job_level", "Tenure": "tenure_band",
              "Income quartile": "income_band", "Stock option level": "stock_option_level",
              "Job satisfaction": "job_satisfaction", "Environment satisfaction": "environment_satisfaction",
              "Work-life balance": "work_life_balance"}
choice = st.selectbox("Break attrition down by", list(DIMENSIONS))
col = DIMENSIONS[choice]
g = stats.group_rates(lab, col)

fig = go.Figure(go.Bar(
    x=g["rate_pct"], y=g[col].astype(str), orientation="h", marker_color=ui.BLUE,
    error_x=dict(type="data", symmetric=False, array=g["ci_high_pct"] - g["rate_pct"],
                 arrayminus=g["rate_pct"] - g["ci_low_pct"], color="#888"),
    customdata=g[["employees", "leavers", "ci_low_pct", "ci_high_pct"]],
    hovertemplate="%{y}<br>%{x:.1f}% left (%{customdata[1]} of %{customdata[0]})"
                  "<br>95% CI %{customdata[2]:.1f}% to %{customdata[3]:.1f}%<extra></extra>"))
fig.add_vline(x=overall, line_dash="dot", line_color=ui.NAVY,
              annotation_text=f"company {overall:.1f}%", annotation_position="top")
fig.update_layout(xaxis_title="Attrition rate (%)", yaxis_title=None, height=120 + 40 * len(g),
                  margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Bars show the share of each group that left; whiskers are 95% Wilson intervals. "
           "Overlapping whiskers mean the difference may be chance.")

hidden = lab[col].value_counts()
hidden = hidden[(hidden > 0) & (hidden < config.MIN_GROUP)]
if len(hidden):
    st.caption(f"{len(hidden)} group(s) hidden for having fewer than {config.MIN_GROUP} employees.")
with st.expander("Table view"):
    st.dataframe(g.round(1), hide_index=True, width="stretch")
