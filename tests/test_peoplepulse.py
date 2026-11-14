import os
import re

import numpy as np
import pandas as pd
import psycopg
import pytest
from streamlit.testing.v1 import AppTest

from peoplepulse import config, data, model, stats

TEST_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/peoplepulse_test")


@pytest.fixture(scope="module")
def df():
    return data.load()


@pytest.fixture(scope="module")
def results(df):
    return model.run(df)


def test_bundled_data_matches_the_known_source(df):
    assert len(df) == 1470 and df["attrition"].sum() == 237


def test_validate_rejects_bad_input():
    raw = data.read_csv()
    with pytest.raises(data.SchemaError):
        data.validate(raw.drop(columns=["OverTime"]))
    bad = raw.copy()
    bad.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(data.SchemaError):
        data.validate(bad)


def test_tidy_columns_match_the_postgres_table(df):
    ddl = (config.SQL_DIR / "schema.sql").read_text()
    sql_cols = re.findall(r"^\s{4}(\w+)\s+(?:integer|smallint|text)", ddl, flags=re.M)
    assert sorted(sql_cols) == sorted(df.columns)
    assert not set(data.CONSTANT) & set(df.columns)


def test_wilson_interval_and_rate():
    r = stats.rate(pd.DataFrame({"attrition": [1] + [0] * 9}))
    assert r["rate_pct"] == 10 and r["ci_low_pct"] == pytest.approx(1.79, abs=0.01)
    assert r["ci_high_pct"] == pytest.approx(40.42, abs=0.01)


def test_small_groups_are_suppressed():
    d = pd.DataFrame({"g": ["big"] * 30 + ["tiny"] * 5, "attrition": [0, 1] * 15 + [1] * 5})
    assert list(stats.group_rates(d, "g")["g"]) == ["big"]


def test_multiple_testing_correction_only_raises_p_values(df):
    t = stats.univariate_tests(df)
    assert len(t) == len(stats.CATEGORICAL) + len(stats.NUMERIC)
    assert (t["p_adjusted"] >= t["p_value"] - 1e-12).all()


def test_readable_terms():
    assert stats.readable_term("C(over_time)[T.Yes]") == "over_time = Yes (vs No)"
    assert stats.readable_term("np.log2(monthly_income)") == "monthly_income (per doubling)"


def test_protected_attributes_are_not_model_inputs(df):
    x, _ = model.features(df)
    assert not set(model.PROTECTED) & set(x.columns) and "employee_number" not in x.columns


def test_model_outputs_are_aggregate_only(results):
    """No per-employee scores leave model.run: every table is far smaller than the 294-person holdout."""
    tables = [v for v in results.values() if isinstance(v, pd.DataFrame)]
    assert tables and all(len(t) < 50 for t in tables)
    assert not any(isinstance(v, np.ndarray) for v in results.values())


def test_metrics_are_sane_and_best_is_chosen_by_cv(results):
    m = results["metrics"]
    assert m["holdout_roc_auc"].between(0.5, 1).all()
    assert (m["holdout_auc_ci_low"] <= m["holdout_roc_auc"]).all() and (m["holdout_roc_auc"] <= m["holdout_auc_ci_high"]).all()
    assert results["best_model"] == m.loc[m["cv_roc_auc_mean"].idxmax(), "model"]


def test_shap_sums_one_hot_columns_back_to_variables(results):
    s = results["shap"]
    assert s["variable"].is_unique and "over_time" in set(s["variable"])
    assert not any(v.startswith("job_role_") for v in s["variable"])


def test_postgres_round_trip_equals_csv(df, monkeypatch):
    from peoplepulse import db
    admin, name = TEST_URL.rsplit("/", 1)
    with psycopg.connect(admin + "/postgres", autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {name}")
    monkeypatch.setattr(config, "DATABASE_URL", TEST_URL)
    assert db.load_employees(df) == 1470
    back = data.load()
    pd.testing.assert_frame_equal(back[df.columns], df.sort_values("employee_number").reset_index(drop=True),
                                  check_dtype=False)


def test_postgres_rejects_impossible_rows(df, monkeypatch):
    from peoplepulse import db
    monkeypatch.setattr(config, "DATABASE_URL", TEST_URL)
    bad = df.head(3).copy()
    bad.loc[0, "years_at_company"] = bad.loc[0, "total_working_years"] + 1
    with pytest.raises(psycopg.errors.CheckViolation):
        db.load_employees(bad)


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Attrition_Overview.py", "pages/2_Statistical_Tests.py",
                                  "pages/3_Model_Insights.py", "pages/4_Responsible_Use.py"])
def test_every_page_runs_in_demo_mode(page, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", None)
    at = AppTest.from_file(str(config.ROOT / page), default_timeout=180).run()
    assert not at.exception
