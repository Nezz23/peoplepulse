"""Load, validate and label the IBM HR attrition data, from Postgres when DATABASE_URL is set, else the bundled CSV."""
import re

import pandas as pd

from . import config

# Column -> required dtype kind (i = integer, O = text) in the source file.
SCHEMA = {
    "Age": "i", "Attrition": "O", "BusinessTravel": "O", "DailyRate": "i", "Department": "O",
    "DistanceFromHome": "i", "Education": "i", "EducationField": "O", "EmployeeCount": "i", "EmployeeNumber": "i",
    "EnvironmentSatisfaction": "i", "Gender": "O", "HourlyRate": "i", "JobInvolvement": "i", "JobLevel": "i",
    "JobRole": "O", "JobSatisfaction": "i", "MaritalStatus": "O", "MonthlyIncome": "i", "MonthlyRate": "i",
    "NumCompaniesWorked": "i", "Over18": "O", "OverTime": "O", "PercentSalaryHike": "i", "PerformanceRating": "i",
    "RelationshipSatisfaction": "i", "StandardHours": "i", "StockOptionLevel": "i", "TotalWorkingYears": "i",
    "TrainingTimesLastYear": "i", "WorkLifeBalance": "i", "YearsAtCompany": "i", "YearsInCurrentRole": "i",
    "YearsSinceLastPromotion": "i", "YearsWithCurrManager": "i",
}

# The same value for every employee (1, "Y", 80): they carry no information.
CONSTANT = ["employee_count", "over18", "standard_hours"]

# Codes from IBM's data dictionary for the 1-4 / 1-5 survey and rating scales.
_SAT = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
LABELS = {
    "education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "environment_satisfaction": _SAT, "job_satisfaction": _SAT, "relationship_satisfaction": _SAT,
    "job_involvement": _SAT,
    "performance_rating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "work_life_balance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}


class SchemaError(ValueError):
    pass


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def read_csv(path=config.CSV) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")       # the source file starts with a BOM


def validate(raw: pd.DataFrame) -> None:
    missing = set(SCHEMA) - set(raw.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")
    wrong = [c for c, kind in SCHEMA.items() if raw[c].dtype.kind != kind and not
             (kind == "O" and pd.api.types.is_string_dtype(raw[c]))]
    if wrong:
        raise SchemaError(f"unexpected types: {wrong}")
    if raw.isna().any().any():
        raise SchemaError("null values present")
    if not raw["EmployeeNumber"].is_unique:
        raise SchemaError("EmployeeNumber is not unique")
    if not set(raw["Attrition"]) <= {"Yes", "No"}:
        raise SchemaError("Attrition must be Yes/No")


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """snake_case columns, 0/1 target, constant columns dropped. Ordinal codes stay numeric (models need them)."""
    df = raw.rename(columns=snake).drop(columns=CONSTANT)
    df["attrition"] = (df["attrition"] == "Yes").astype(int)
    return df


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Readable copy for charts: survey codes as words, plus tenure and income bands."""
    out = df.copy()
    for col, mapping in LABELS.items():
        out[col] = pd.Categorical(out[col].map(mapping), categories=list(mapping.values()), ordered=True)
    out["tenure_band"] = pd.cut(out["years_at_company"], [-1, 1, 3, 5, 10, 40],
                                labels=["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
    out["income_band"] = pd.qcut(out["monthly_income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    return out


def load() -> pd.DataFrame:
    """The tidy table. Postgres when DATABASE_URL is set, otherwise demo mode on the bundled CSV."""
    if config.DATABASE_URL:
        from . import db
        return db.read_employees()
    raw = read_csv()
    validate(raw)
    return tidy(raw)


def mode() -> str:
    return "PostgreSQL" if config.DATABASE_URL else "Demo (bundled CSV, no database)"
