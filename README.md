## Quickstart
```bash
# Windows PowerShell (Poetry required)
poetry install
poetry run python -m esports_quant.cli ingest_opendota --limit 1500
poetry run python -m esports_quant.cli train
poetry run python -m esports_quant.cli evaluate
poetry run pytest


![ci](https://github.com/Dylan-McG/esports_model_project/actions/workflows/ci.yml/badge.svg)
