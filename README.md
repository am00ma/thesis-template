# Thesis template

To follow along with [guidelines](https://manyids2x.nl).

## Installation

1. Clone the repo

   ```bash
   git clone https://github.com/am00ma/thesis-template
   ```

2. Follow instructions from [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

3. Install virtual environment:

   ```bash
   # Install dependencies listed in pyproject.toml
   uv sync

   # Activate the virtual environment (sh)
   source .venv/bin/activate
   ```

## Design notes

1. Composable configs
2. Saving history with git commit hash
3. Using sqlite3 as data format
4. Advantages of torch dataloader
5. Advantages of using pandas df to load dataset
6. Structuring of subdirs: data, config, experiments, outputs
7. Treating config, experiments, outputs as data
8. Running on Snellius
