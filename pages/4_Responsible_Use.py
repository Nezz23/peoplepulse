import streamlit as st

from peoplepulse import config, ui

ui.header("Responsible use")
st.markdown(f"""
**What this app is for:** spotting group-level patterns (roles, working conditions, tenure) worth a closer,
human look, such as an overtime or travel policy review.

**What it must not be used for**
- **Scoring or flagging individual employees.** The app has no per-person risk score, by design, and the code
  never stores one. A flight-risk label can change how a manager treats someone and becomes self-fulfilling.
- **Decisions about hiring, pay, promotion or termination.**
- **Causal claims.** Every number is an association in cross-sectional data. People who work overtime leave more
  often here, but overtime could be a symptom of understaffing, not the cause.

**Safeguards built in**
- Age, gender and marital status are excluded from the model and used only to audit it.
- Groups under {config.MIN_GROUP} employees are hidden.
- 30 tests are corrected for multiple comparisons (Benjamini-Hochberg).
- The holdout set is used once, after the model is chosen by cross-validation.

**Limits of the data**
- It is **fictional**: IBM data scientists made it up to teach analytics. Nothing here describes a real company.
- One snapshot with no dates: no trends, no time to exit, no reason for leaving.
- 237 leavers, so estimates for small roles have wide intervals.
- DailyRate, HourlyRate and MonthlyRate have no documented meaning and look like noise; one still reaches
  nominal significance, a reminder that a small p-value is not proof.
""")
