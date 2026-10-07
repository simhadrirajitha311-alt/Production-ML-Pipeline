# Contributing

Thanks for contributing to the AutoML Production Pipeline project.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Pull request checklist

- Add or update tests for behavioral changes.
- Keep code idiomatic, documented, and type-hinted.
- Ensure model training and data validation remain leakage-safe.
- Document any configuration or architecture changes.
