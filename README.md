# Thesis template

To follow along with [guidelines](https://manyids2x.nl).

## Installation

1. Clone the repo
    ```bash
    git clone https://github.com/am00ma/thesis-template
    ```

2. Follow instructions from https://docs.astral.sh/uv/getting-started/installation/ to install `uv`.

3. Install virtual environment:

    ```bash

    # Install the things mentioned in pyproject.toml
    uv sync

    # Activate the virtual environment (sh)
    source .venv/bin/activate
    ```

## Usage

1. Serve the site:
    ```bash
    mkdocs serve
    ```
2. Open the page in a browser: [link](http://localhost:4444)
3. Modify the content in the `docs` folder as needed (page will reload).
4. Commit all changes to git:
    ```bash
    git add .
    git commit -m "some commit message"
    ```
