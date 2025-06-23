"""
1. Read, write with sqlite3 databases and pandas DataFrames.
2. Check git status for config file
3. Helpers to load/save configs to/from json
"""

import json
import sqlite3
import subprocess
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


def check_git_status():
    """
    Check the Git status of the current repository.

    Returns:
        str:
            - The commit hash (if clean).
            - The commit hash prefixed with 'dirty-' (if unstaged changes exist).

    Raises:
        FileNotFoundError: If git is not installed.
        subprocess.CalledProcessError: If git commands fail.
        RuntimeError: If not in a Git repository.
    """
    try:
        # Check if the current directory is a git repository
        if not Path(".git").exists():
            raise RuntimeError("Not a git repository")

        # Get the current commit hash (regardless of dirty state)
        commit_hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_hash_result.stdout.strip()

        # Check if there are unstaged changes
        diff_result = subprocess.run(["git", "diff", "--quiet"], capture_output=True, text=True)

        # Return dirty-{hash} if unstaged changes exist (git diff returns 1)
        if diff_result.returncode == 1:
            return f"dirty-{commit_hash}"
        else:
            return commit_hash

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError("Git is not installed or not in PATH") from e


JSONType = dict[str, str | int | float | Path]


def save_json(
    path: str | Path,
    data: JSONType,
    overrides: JSONType | None = None,
    verbose: bool = False,
) -> None:
    """Save JSON dictionary to file, with overrides."""

    # New dict to prevent mutation of original
    new_data = {}

    # Append any extra data (also overrides info if duplicate keys)
    if overrides is not None:
        new_data = {**data, **overrides}

    # Convert paths to strings
    for k, v in new_data.items():
        if isinstance(v, Path):
            new_data[k] = str(v)
        else:
            new_data[k] = v

    # Save to json
    text = json.dumps(new_data, indent=4)
    Path(path).write_text(text)

    if verbose:
        print(f"{text}\nSaved to: {path}")


def load_json(
    path: str | Path,
    path_cols: list[str] | None = None,
    verbose: bool = False,
) -> JSONType:
    """Load JSON dictionary from file."""

    # Load from file
    with open(path, "r") as f:
        data = json.load(f)

    # Convert paths to strings
    if path_cols is not None:
        for name in path_cols:
            data[name] = Path(data[name])

    if verbose:
        print(f"{json.dumps(data, indent=4)}\nLoaded from: {path}")

    return data
