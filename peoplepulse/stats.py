"""Group attrition rates and significance tests. Every result is an association in fictional data, not a cause."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from . import config

CATEGORICAL = ["business_travel", "department", "education", "education_field", "environment_satisfaction",
               "gender", "job_involvement", "job_level", "job_role", "job_satisfaction", "marital_status",
               "over_time", "performance_rating", "relationship_satisfaction", "stock_option_level",
               "work_life_balance"]
NUMERIC = ["age", "daily_rate", "distance_from_home", "hourly_rate", "monthly_income", "monthly_rate",
           "num_companies_worked", "percent_salary_hike", "total_working_years", "training_times_last_year",
           "years_at_company", "years_in_current_role", "years_since_last_promotion", "years_with_curr_manager"]


def rate(df: pd.DataFrame) -> dict:
    n, k = len(df), int(df["attrition"].sum())
    lo, hi = proportion_confint(k, n, method="wilson")
    return {"employees": n, "leavers": k, "rate_pct": 100 * k / n, "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi}


def group_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Attrition rate per group with a 95% Wilson interval. Groups under MIN_GROUP are dropped, not shown."""
    rows = [{col: g, **rate(part)} for g, part in df.groupby(col, observed=True)]
    out = pd.DataFrame(rows)
    return out[out["employees"] >= config.MIN_GROUP].reset_index(drop=True)


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def univariate_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square (categorical) or Mann-Whitney U (numeric) per variable, Benjamini-Hochberg across all of them."""
    rows = []
    for col in CATEGORICAL:
        table = pd.crosstab(df[col], df["attrition"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"variable": col, "test": "chi-square", "statistic": chi2, "p_value": p,
                     "effect_size": cramers_v(table, chi2), "effect_measure": "Cramer's V"})
    left, stayed = df[df["attrition"] == 1], df[df["attrition"] == 0]
    for col in NUMERIC:
        u, p = stats.mannwhitneyu(left[col], stayed[col])
        rows.append({"variable": col, "test": "Mann-Whitney U", "statistic": u, "p_value": p,
                     # rank-biserial: >0 means leavers tend to have higher values
                     "effect_size": 2 * u / (len(left) * len(stayed)) - 1, "effect_measure": "rank-biserial r",
                     "median_left": left[col].median(), "median_stayed": stayed[col].median()})
    out = pd.DataFrame(rows)
    out["p_adjusted"] = multipletests(out["p_value"], method="fdr_bh")[1]
    out["significant_fdr_5pct"] = out["p_adjusted"] < 0.05
    return out.sort_values("effect_size", key=abs, ascending=False).reset_index(drop=True)


# Adjusted model: work-condition variables only. Protected attributes (age, gender, marital status) are left out.
FORMULA = ("attrition ~ C(over_time) + C(business_travel, Treatment('Non-Travel')) + job_level + np.log2(monthly_income)"
           " + years_at_company + years_since_last_promotion + num_companies_worked + distance_from_home"
           " + job_satisfaction + environment_satisfaction + work_life_balance + C(stock_option_level)")


def readable_term(term: str) -> str:
    """'C(over_time)[T.Yes]' -> 'over_time = Yes (vs No)'; 'np.log2(monthly_income)' -> 'monthly_income (per doubling)'."""
    if term.startswith("C("):
        var, level = term[2:].split(")")[0].split(",")[0], term.split("[T.")[1].rstrip("]")
        base = {"over_time": "No", "business_travel": "Non-Travel", "stock_option_level": "0"}[var]
        return f"{var} = {level} (vs {base})"
    if term.startswith("np.log2("):
        return f"{term[8:-1]} (per doubling)"
    return f"{term} (per unit)"


def adjusted_odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression odds ratios, each holding the others fixed. Associations, not effects of an intervention."""
    fit = smf.logit(FORMULA, data=df).fit(disp=False)
    ci = fit.conf_int()
    out = pd.DataFrame({"term": fit.params.index, "odds_ratio": np.exp(fit.params.values),
                        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values), "p_value": fit.pvalues.values})
    out = out[out["term"] != "Intercept"].reset_index(drop=True)
    out["term"] = out["term"].map(readable_term)
    out.attrs["pseudo_r2"] = fit.prsquared
    out.attrs["n"] = int(fit.nobs)
    return out
