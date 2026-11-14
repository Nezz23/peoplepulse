import plotly.graph_objects as go
import streamlit as st

from peoplepulse import ui

ui.header("Model insights")
r = ui.models()
m = r["metrics"]

st.markdown(f"Three models, fixed settings, trained on 80% of employees. Chosen by 5-fold cross-validation; "
            f"the other 20% ({r['holdout_size']} employees, {r['holdout_leavers']} leavers) is only used for the "
            "final check. Age, gender and marital status are not model inputs; they are only used for the audit below.")
st.dataframe(m.round(3), hide_index=True, width="stretch")
st.caption(f"Best by cross-validation: **{r['best_model']}**. ROC-AUC 0.5 = random ranking. "
           f"PR-AUC of random guessing = {r['pr_auc_baseline']:.2f}. Brier scores of class-weighted models "
           "are inflated because their probabilities are not calibrated.")

st.subheader("What drives the XGBoost model (global SHAP)")
s = r["shap"].head(12).iloc[::-1]
fig = go.Figure(go.Bar(x=s["mean_abs_shap"], y=s["variable"], orientation="h", marker_color=ui.BLUE,
                       hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
fig.update_layout(xaxis_title="Mean |SHAP value| on the holdout (log-odds)", height=460,
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Averaged over all holdout employees: a ranking of what the model relies on, not an explanation of "
           "any one person. Direction column: correlation between a variable's value and its SHAP value "
           "(below 0 means higher values push predictions down).")
with st.expander("Table view"):
    st.dataframe(r["shap"].round(3), hide_index=True, width="stretch")

st.subheader(f"Does the {r['best_model'].lower()} work equally well across groups?")
st.dataframe(r["audit"].round(3), hide_index=True, width="stretch")
st.caption("ROC-AUC per group on the holdout. Blank when a group has fewer than 10 leavers or stayers. "
           "Holdout groups are small, so gaps of a few points are within noise.")
