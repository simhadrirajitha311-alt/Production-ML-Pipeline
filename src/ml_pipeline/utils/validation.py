from __future__ import annotations


def ensure_non_empty(value, name: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} cannot be empty.")
