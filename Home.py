import streamlit as st

from peoplepulse import config, stats, ui

ui.header("PeoplePulse: workforce attrition patterns")
df, lab = ui.employees()
r = stats.rate(df)

c1, c2, c3 = st.columns(3)
c1.metric("Employees", f"{r['employees']:,}")
c2.metric("Left the company", f"{r['leavers']:,}")
c3.metric("Attrition rate", f"{r['rate_pct']:.1f}%", help=f"95% CI {r['ci_low_pct']:.1f}% to {r['ci_high_pct']:.1f}%")

st.markdown(f"""
**What questions this answers**

| Page | Question |
|---|---|
| Attrition Overview | Which departments, roles and working conditions have higher attrition? |
| Statistical Tests | Which differences are larger than chance, after correcting for testing 30 variables? |
| Model Insights | How well can attrition be predicted, which factors drive the model, and does it work equally well across groups? |
| Responsible Use | What this analysis must not be used for, and its limits. |

Groups with fewer than {config.MIN_GROUP} employees are hidden everywhere so no small group can be singled out.
""")
