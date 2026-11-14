"""Optional PostgreSQL storage. `python -m peoplepulse.db` loads the CSV; the app reads it when DATABASE_URL is set."""
import pandas as pd
import psycopg

from . import config, data


def connect() -> psycopg.Connection:
    return psycopg.connect(config.DATABASE_URL, autocommit=True)


def load_employees(df: pd.DataFrame) -> int:
    """Rebuild hr.employees from the tidy frame. The table's CHECK constraints reject impossible rows."""
    with connect() as conn:
        conn.execute((config.SQL_DIR / "schema.sql").read_text())
        cols = ", ".join(df.columns)
        with conn.cursor().copy(f"COPY hr.employees ({cols}) FROM STDIN") as copy:
            for row in df.itertuples(index=False):
                copy.write_row([v.item() if hasattr(v, "item") else v for v in row])
        return conn.execute("SELECT count(*) FROM hr.employees").fetchone()[0]


def read_employees() -> pd.DataFrame:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM hr.employees ORDER BY employee_number")
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


if __name__ == "__main__":
    raw = data.read_csv()
    data.validate(raw)
    print(f"loaded {load_employees(data.tidy(raw))} employees into {config.DATABASE_URL}")
