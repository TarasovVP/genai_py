from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import psycopg


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


class PostgresClient:
    def __init__(self, cfg: PostgresConfig):
        self._cfg = cfg

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self._cfg.host,
            port=self._cfg.port,
            dbname=self._cfg.dbname,
            user=self._cfg.user,
            password=self._cfg.password,
        )

    def apply_ddl(self, ddl_text: str) -> None:
        statements = self._split_sql_statements(ddl_text)
        with self.connect() as conn:
            with conn:
                with conn.cursor() as cur:
                    for stmt in statements:
                        s = stmt.strip()
                        if not s:
                            continue
                        cur.execute(s)

    def reset_public_schema(self) -> None:
        with self.connect() as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DROP SCHEMA IF EXISTS public CASCADE;")
                    cur.execute("CREATE SCHEMA public;")
                    cur.execute("GRANT ALL ON SCHEMA public TO public;")

    def insert_df(self, table: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0

        with self.connect() as conn:
            with conn:
                inserted = self.insert_df_in_conn(conn, table, df)
            return inserted

    def insert_df_in_conn(self, conn: psycopg.Connection, table: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0

        cols = list(df.columns)
        placeholders = ", ".join(["%s"] * len(cols))
        col_sql = ", ".join([self._quote_ident(c) for c in cols])
        sql = f"INSERT INTO {self._quote_ident(table)} ({col_sql}) VALUES ({placeholders})"

        values = df.where(pd.notnull(df), None).itertuples(index=False, name=None)

        with conn.cursor() as cur:
            cur.executemany(sql, values)

        return len(df)

    def insert_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        if not tables:
            return result

        for name, df in tables.items():
            result[name] = self.insert_df(name, df)
        return result

    @staticmethod
    def _quote_ident(s: str) -> str:
        s = str(s)
        s = s.replace('"', '""')
        return f'"{s}"'

    @staticmethod
    def _split_sql_statements(sql: str) -> List[str]:
        out: List[str] = []
        buf: List[str] = []
        in_single = False
        in_double = False
        i = 0
        while i < len(sql):
            ch = sql[i]
            if ch == "'" and not in_double:
                if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_single = not in_single
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                buf.append(ch)
                i += 1
                continue
            if ch == ";" and not in_single and not in_double:
                out.append("".join(buf))
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
        return out
