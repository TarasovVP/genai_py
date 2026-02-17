from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    default_rows_per_table: int
    default_seed: int

    vertex_project: str
    vertex_location: str
    vertex_model: str

    datasets_root: Path

    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str

    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str


def get_settings() -> Settings:
    default_rows_per_table = int(os.getenv("DEFAULT_ROWS_PER_TABLE", "10"))
    default_seed = int(os.getenv("DEFAULT_SEED", "0"))

    vertex_project = os.getenv("VERTEX_PROJECT", "gd-gcp-gridu-genai")
    vertex_location = os.getenv("VERTEX_LOCATION", "europe-west1")
    vertex_model = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")

    datasets_root = Path(os.getenv("DATASETS_ROOT", "datasets"))
    datasets_root.mkdir(parents=True, exist_ok=True)

    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = int(os.getenv("PG_PORT", "55432"))
    pg_db = os.getenv("PG_DB", "data_assistant")
    pg_user = os.getenv("PG_USER", "data_assistant")
    pg_password = os.getenv("PG_PASSWORD", "data_assistant")

    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    return Settings(
        default_rows_per_table=default_rows_per_table,
        default_seed=default_seed,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_model=vertex_model,
        datasets_root=datasets_root,
        pg_host=pg_host,
        pg_port=pg_port,
        pg_db=pg_db,
        pg_user=pg_user,
        pg_password=pg_password,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
    )
