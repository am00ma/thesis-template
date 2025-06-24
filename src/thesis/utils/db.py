import sqlite3
from pathlib import Path
from typing import Literal

import pandas as pd


def save_db(
    df: pd.DataFrame,
    db_path: Path | str,
    table: str,
    if_exists: Literal["fail", "replace", "append"] = "replace",
    index: bool = False,
    verbose: bool = False,
) -> None:
    """Save DataFrame to sqlite3 db as specified table.
    NOTE : default behaviour is to replace the table if it exists."""
    conn = sqlite3.connect(str(db_path))

    if len(df.columns) == 0:
        print(f"Cannot save empty df ->\n{df}")
        return

    df.to_sql(table, conn, if_exists=if_exists, index=index)
    if verbose:
        print(df)
        print(df.columns)
        print(f"Saved to => {table} => {db_path}")


def load_db(
    db_path: Path | str,
    table: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read sqlite3 db, return all rows from specified table."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * from '{table}'", conn)
    if verbose:
        print(df)
        print(df.columns)
    return df
