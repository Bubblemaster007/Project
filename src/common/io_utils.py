from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config.yaml without requiring PyYAML when it uses JSON syntax."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{config_path} is not JSON-compatible YAML and PyYAML is not installed."
            ) from exc
        return yaml.safe_load(text)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out
