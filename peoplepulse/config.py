import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "ibm_hr_attrition.csv"
SQL_DIR = ROOT / "sql"
RESULTS_DIR = ROOT / "results"
SOURCE_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"

# Unset -> demo mode: the app reads the bundled CSV and needs no database.
DATABASE_URL = os.environ.get("DATABASE_URL")

# Groups smaller than this are suppressed on every page, so no small group can be singled out.
MIN_GROUP = 20
SEED = 42
