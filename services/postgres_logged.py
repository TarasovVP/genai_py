from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd


class PostgresLogged:

    def __init__(
        self,
        pg: Any,
        tracer: Any,
        page: str,
        dataset_id: Optional[str],
    ):
        self.pg = pg
        self.tracer = tracer
        self.page = page
        self.dataset_id = dataset_id

    def _span(self, name: str, start_ts: float, end_ts: float, metadata: Optional[Dict[str, Any]], status: str):
        self.tracer.span(
            page=self.page,
            dataset_id=self.dataset_id,
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            metadata=metadata or {},
            status=status,
        )

    def full_reload(self, ddl_text: str, tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        t0 = time.time()
        status = "ok"
        err = None
        try:
            self.pg.reset_public_schema()
            self.pg.apply_ddl(ddl_text)
            inserted = self.pg.insert_tables(tables)
            return inserted
        except Exception as e:
            status = "error"
            err = str(e)
            raise
        finally:
            self._span(
                name="postgres_full_reload",
                start_ts=t0,
                end_ts=time.time(),
                metadata={
                    "tables": list((tables or {}).keys()),
                    "ddl_chars": len(ddl_text or ""),
                    "error": err or "",
                },
                status=status,
            )

    def reload_table(self, table_name: str, df: pd.DataFrame) -> int:
        t0 = time.time()
        status = "ok"
        err = None
        try:
            with self.pg.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;')
                conn.commit()
            return self.pg.insert_df(table_name, df)
        except Exception as e:
            status = "error"
            err = str(e)
            raise
        finally:
            self._span(
                name="postgres_reload_table",
                start_ts=t0,
                end_ts=time.time(),
                metadata={
                    "table": table_name,
                    "rows": int(len(df)) if df is not None else 0,
                    "cols": int(len(df.columns)) if df is not None else 0,
                    "error": err or "",
                },
                status=status,
            )

    def run_select_to_df(self, sql: str) -> pd.DataFrame:
        t0 = time.time()
        status = "ok"
        err = None
        try:
            with self.pg.connect() as conn:
                df = pd.read_sql_query(sql, conn)
            return df
        except Exception as e:
            status = "error"
            err = str(e)
            raise
        finally:
            self._span(
                name="postgres_select_query",
                start_ts=t0,
                end_ts=time.time(),
                metadata={
                    "sql": (sql or "")[:5000],
                    "error": err or "",
                },
                status=status,
            )
