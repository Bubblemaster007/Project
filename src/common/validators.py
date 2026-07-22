from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要字段：{', '.join(missing)}")
