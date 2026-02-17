from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict

import pandas as pd
import zipfile
from uuid import uuid4


def new_dataset_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def dataset_dir(root: Path, dataset_id: str) -> Path:
    d = root / dataset_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_text(path: Path, text: str) -> None:
    path.write_text(text or "", encoding="utf-8")


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj or {}, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table_csv(root: Path, dataset_id: str, table_name: str, df: pd.DataFrame) -> None:
    d = dataset_dir(root, dataset_id)
    csv_path = d / f"{table_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")


def save_dataset_to_disk(
    root: Path,
    dataset_id: str,
    ddl_text: str,
    schema: dict,
    tables: Dict[str, pd.DataFrame],
    dataset_prompt: str,
) -> None:
    d = dataset_dir(root, dataset_id)
    _save_text(d / "ddl.sql", ddl_text or "")
    _save_json(d / "schema.json", schema or {})
    _save_json(
        d / "meta.json",
        {
            "dataset_id": dataset_id,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tables": list((tables or {}).keys()),
            "dataset_prompt": dataset_prompt or "",
        },
    )
    for tname, tdf in (tables or {}).items():
        save_table_csv(root, dataset_id, tname, tdf)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def tables_to_zip_bytes(tables: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in (tables or {}).items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    bio.seek(0)
    return bio.read()
