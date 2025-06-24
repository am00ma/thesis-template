import json
from pathlib import Path


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
